import express from 'express';
import { readdir, stat } from 'fs/promises';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const app = express();
const PORT = 3040;

// Docker로 런처를 띄울 때: WORKSPACE=마운트 경로, HOST_PROJECT_PATH=호스트 절대경로(필수)
const IN_DOCKER = process.env.WORKSPACE != null;
const LIST_DIR = IN_DOCKER ? process.env.WORKSPACE : resolve(__dirname, '..');
const SLIDES_DIR = join(LIST_DIR, 'slides');
const COMPOSE_FILE = process.env.COMPOSE_FILE || resolve(LIST_DIR, 'compose.yml');
const CONTAINER_NAME = 'slidev-runner';
const SLIDEV_HOST = process.env.SLIDEV_HOST || CONTAINER_NAME;
const SLIDEV_PORT = 3030;

// 동시 클릭 직렬화: 동시 요청이 와도 순차적으로 처리 (Docker 컨테이너 이름 충돌 방지)
let runChain = Promise.resolve();
function serialize(fn) {
  const next = runChain.then(fn, fn);
  runChain = next.catch(() => {});
  return next;
}

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

/**
 * slides/ 폴더에서 슬라이드 덱 발견.
 * - slides/<name>.md           → name="<name>", path="slides/<name>.md"
 * - slides/<name>/index.md     → name="<name>", path="slides/<name>/index.md"
 */
async function listDecks() {
  let entries;
  try {
    entries = await readdir(SLIDES_DIR, { withFileTypes: true });
  } catch {
    return [];
  }
  const decks = [];
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith('.md') && !entry.name.startsWith('README')) {
      decks.push({
        name: entry.name.replace(/\.md$/, ''),
        path: `slides/${entry.name}`,
      });
    } else if (entry.isDirectory()) {
      try {
        const s = await stat(join(SLIDES_DIR, entry.name, 'index.md'));
        if (s.isFile()) {
          decks.push({
            name: entry.name,
            path: `slides/${entry.name}/index.md`,
          });
        }
      } catch {
        // index.md 없음 → 덱이 아님
      }
    }
  }
  return decks.sort((a, b) => a.name.localeCompare(b.name));
}

/** GET /api/decks — slides/ 내 슬라이드 덱 목록 */
app.get('/api/decks', async (_req, res) => {
  try {
    res.json({ decks: await listDecks() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/** Slidev 컨테이너의 포트가 열릴 때까지 active polling */
async function waitForSlidev(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const ac = new AbortController();
      const t = setTimeout(() => ac.abort(), 500);
      const res = await fetch(`http://${SLIDEV_HOST}:${SLIDEV_PORT}/`, { signal: ac.signal });
      clearTimeout(t);
      if (res.status < 500) return true;
    } catch {
      // 아직 준비 안 됨
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  return false;
}

/** POST /api/run — 선택한 path로 Docker Slidev 실행 (기존 컨테이너 중지 후 재실행) */
app.post('/api/run', async (req, res) => {
  const path = req.body?.path;
  if (!path || typeof path !== 'string') {
    return res.status(400).json({ error: 'path가 필요합니다.' });
  }
  // 경로 조작 방지: slides/ 아래로 한정
  if (path.includes('..') || path.startsWith('/') || !path.startsWith('slides/')) {
    return res.status(400).json({ error: '잘못된 경로입니다.' });
  }

  const run = (cmd, args, opts = {}) =>
    new Promise((resolve, reject) => {
      const p = spawn(cmd, args, { stdio: 'pipe', ...opts });
      let stdout = '';
      let stderr = '';
      p.stdout?.on('data', (d) => (stdout += d.toString()));
      p.stderr?.on('data', (d) => (stderr += d.toString()));
      p.on('close', (code) => {
        if (code === 0) return resolve({ stdout, stderr });
        const msg = [stderr, stdout].filter(Boolean).join('\n').trim() || `exit ${code}`;
        reject(new Error(msg));
      });
    });

  /** 명령 실행 후 stdout 반환 (실패 시 throw) */
  const runOut = async (cmd, args) => {
    const { stdout } = await run(cmd, args);
    return stdout.trim();
  };

  const compose = (...args) => ['compose', '-f', COMPOSE_FILE, ...args];

  try {
    const result = await serialize(async () => {
      // 기존 slidev 컨테이너 정리 (rm -f 가 stop+remove 를 같이 수행)
      try {
        await run('docker', ['rm', '-f', CONTAINER_NAME]);
        console.log(`[Slidev 런처] 기존 컨테이너 제거: ${CONTAINER_NAME}`);
      } catch {
        // 컨테이너가 없으면 정상
      }

      // 새 컨테이너 실행: compose.yml의 slidev 서비스 정의를 사용하고 명령만 오버라이드
      await run('docker', compose(
        'run', '-d', '--rm', '--name', CONTAINER_NAME, '--service-ports',
        '--quiet-pull',
        'slidev', 'slidev', path, '--remote',
      ));

      // Slidev 포트가 응답할 때까지 active polling
      const ready = await waitForSlidev();
      if (ready) {
        return { status: 200, body: { ok: true, url: 'http://localhost:3030' } };
      }

      // 준비 안 됨 → 컨테이너 상태 확인
      const psOut = await runOut('docker', ['ps', '-q', '-f', `name=${CONTAINER_NAME}`]).catch(() => '');
      let logs = '';
      try {
        logs = await runOut('docker', ['logs', CONTAINER_NAME]);
      } catch {
        logs = '(로그 없음)';
      }
      if (!psOut) {
        await run('docker', ['rm', '-f', CONTAINER_NAME]).catch(() => {});
        console.error('[Slidev 런처] 컨테이너가 바로 종료됨:', logs);
        return { status: 500, body: {
          error: '컨테이너가 바로 종료되었습니다. Slidev 실행에 실패한 것 같습니다.',
          detail: logs,
        }};
      }
      return { status: 504, body: {
        error: 'Slidev가 시간 안에 응답하지 않았습니다.',
        detail: logs,
      }};
    });

    res.status(result.status).json(result.body);
  } catch (err) {
    console.error('[Slidev 런처] Docker 실행 실패:', err.message);
    res.status(500).json({ error: err.message || 'Docker 실행 실패' });
  }
});

app.listen(PORT, () => {
  console.log(`Slidev 런처: http://localhost:${PORT}`);
  console.log(`슬라이드 경로: ${SLIDES_DIR}`);
  console.log(`compose 파일: ${COMPOSE_FILE}`);
});

// 종료 시 slidev-runner 컨테이너 정리 (없으면 compose down 시 네트워크 제거 실패)
let shuttingDown = false;
async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[Slidev 런처] ${signal} 수신, ${CONTAINER_NAME} 정리…`);
  await new Promise((res) => {
    const p = spawn('docker', ['rm', '-f', CONTAINER_NAME], { stdio: 'ignore' });
    p.on('close', () => res());
    p.on('error', () => res());
  });
  process.exit(0);
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

import express from 'express';
import { readdir, stat } from 'fs/promises';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const app = express();
const PORT = 3040;

// Docker로 런처를 띄울 때: WORKSPACE=마운트 경로, HOST_PROJECT_PATH=호스트 절대경로
const IN_DOCKER = process.env.WORKSPACE != null;
const LIST_DIR = IN_DOCKER ? process.env.WORKSPACE : resolve(__dirname, '..');
const SLIDES_DIR = join(LIST_DIR, 'slides');
const CONTAINER_NAME = 'slidev-runner';
const SLIDEV_HOST = process.env.SLIDEV_HOST || CONTAINER_NAME;
const SLIDEV_PORT = 3030;

// 동시 클릭 직렬화: kill → start 사이에 다른 요청이 끼어들어 포트 충돌 나는 것 방지
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

/** Slidev 포트가 열릴 때까지 active polling */
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
    await new Promise((r) => setTimeout(r, 100));
  }
  return false;
}

function dockerRun(args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn('docker', args, { stdio: 'pipe', ...opts });
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
}

/** POST /api/run — 선택한 path로 Slidev 실행 (이전 slidev 종료 후 새로 띄움) */
app.post('/api/run', async (req, res) => {
  const path = req.body?.path;
  if (!path || typeof path !== 'string') {
    return res.status(400).json({ error: 'path가 필요합니다.' });
  }
  // 경로 조작 방지: slides/ 아래로 한정
  if (path.includes('..') || path.startsWith('/') || !path.startsWith('slides/')) {
    return res.status(400).json({ error: '잘못된 경로입니다.' });
  }

  try {
    const result = await serialize(async () => {
      // 1. 이전 slidev 프로세스 종료. TERM 후 잠시 대기, 살아있으면 KILL.
      //    포트(3030)가 해제될 때까지 기다림.
      await dockerRun(['exec', CONTAINER_NAME, 'sh', '-c', `
        if pgrep -f "slidev" >/dev/null 2>&1; then
          pkill -TERM -f "slidev" 2>/dev/null || true
          for i in $(seq 1 30); do
            pgrep -f "slidev" >/dev/null 2>&1 || break
            sleep 0.05
          done
          pkill -KILL -f "slidev" 2>/dev/null || true
        fi
      `]).catch((e) => {
        console.warn('[Slidev 런처] 이전 프로세스 정리 중 경고:', e.message);
      });

      // 2. 새 slidev 프로세스 detach 실행. 컨테이너 안에서 백그라운드로 띄움.
      //    nohup + & 로 띄워서 docker exec가 바로 리턴하고, 출력은 컨테이너 stdout으로.
      await dockerRun(['exec', '-d', CONTAINER_NAME, 'sh', '-c',
        `cd /slidev && exec slidev "${path}" --remote`]);

      // 3. 포트가 응답할 때까지 active polling
      const ready = await waitForSlidev();
      if (ready) {
        return { status: 200, body: { ok: true, url: 'http://localhost:3030' } };
      }
      return { status: 504, body: {
        error: 'Slidev가 시간 안에 응답하지 않았습니다.',
      }};
    });

    res.status(result.status).json(result.body);
  } catch (err) {
    console.error('[Slidev 런처] 실행 실패:', err.message);
    res.status(500).json({ error: err.message || '실행 실패' });
  }
});

app.listen(PORT, () => {
  console.log(`Slidev 런처: http://localhost:${PORT}`);
  console.log(`슬라이드 경로: ${SLIDES_DIR}`);
});

// 종료 시 컨테이너 안의 slidev 프로세스 정리 (컨테이너 자체는 compose가 관리)
let shuttingDown = false;
async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[Slidev 런처] ${signal} 수신, slidev 프로세스 정리…`);
  try {
    await dockerRun(['exec', CONTAINER_NAME, 'sh', '-c', 'pkill -KILL -f slidev 2>/dev/null || true']);
  } catch {
    // 컨테이너가 이미 정지됐을 수 있음 — 무시
  }
  process.exit(0);
}
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

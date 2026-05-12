import express from 'express';
import { readdir } from 'fs/promises';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const app = express();
const PORT = 3040;

// Docker로 런처를 띄울 때: WORKSPACE=마운트 경로, HOST_PROJECT_PATH=호스트 절대경로(필수)
const IN_DOCKER = process.env.WORKSPACE != null;
const LIST_DIR = IN_DOCKER ? process.env.WORKSPACE : resolve(__dirname, '..');
const COMPOSE_FILE = process.env.COMPOSE_FILE || resolve(LIST_DIR, 'compose.yml');
const CONTAINER_NAME = 'slidev-runner';

// 동시 클릭 직렬화: 동시 요청이 와도 순차적으로 처리 (Docker 컨테이너 이름 충돌 방지)
let runChain = Promise.resolve();
function serialize(fn) {
  const next = runChain.then(fn, fn);
  runChain = next.catch(() => {});
  return next;
}

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

/** 슬라이드로 쓸 .md 파일만 필터 (Slides.md 패턴 또는 그냥 .md) */
function isSlideFile(name) {
  return name.endsWith('.md') && !name.startsWith('README');
}

/** GET /api/files — 슬라이드 .md 파일 목록 */
app.get('/api/files', async (_req, res) => {
  try {
    const names = await readdir(LIST_DIR);
    const files = names.filter(isSlideFile).sort();
    res.json({ files });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/** POST /api/run — 선택한 파일로 Docker Slidev 실행 (기존 컨테이너 중지 후 재실행) */
app.post('/api/run', async (req, res) => {
  const file = req.body?.file;
  if (!file || typeof file !== 'string') {
    return res.status(400).json({ error: 'file 이름이 필요합니다.' });
  }
  // 간단한 보안: 파일명에 경로 조작 방지
  if (file.includes('/') || file.includes('..')) {
    return res.status(400).json({ error: '잘못된 파일명입니다.' });
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
    // 동시 클릭이 와도 직렬화: 컨테이너 이름 충돌 방지
    const result = await serialize(async () => {
      // 기존 slidev 컨테이너 정리: kill → rm 순서로 확실히 제거
      try {
        await run('docker', ['kill', CONTAINER_NAME]);
      } catch {
        // 이미 멈춰 있거나 존재하지 않으면 무시
      }
      try {
        await run('docker', ['rm', '-f', CONTAINER_NAME]);
        console.log(`[Slidev 런처] 기존 컨테이너 제거: ${CONTAINER_NAME}`);
      } catch {
        // 컨테이너가 없으면 정상
      }

      // 새 컨테이너 실행: compose.yml의 slidev 서비스 정의를 사용하고 명령만 오버라이드
      // --service-ports: compose.yml에 정의된 포트 매핑 사용
      // --rm: 종료 시 컨테이너 자동 삭제
      // --quiet-pull: pull 진행 출력 억제 (pull_policy: build 와 함께 사용)
      await run('docker', compose(
        'run', '-d', '--rm', '--name', CONTAINER_NAME, '--service-ports',
        '--quiet-pull',
        'slidev', 'slidev', file, '--remote',
      ));

      // Slidev가 기동되어 포트가 열릴 시간을 잠시 대기
      await new Promise((r) => setTimeout(r, 3500));
      const psOut = await runOut('docker', ['ps', '-q', '-f', `name=${CONTAINER_NAME}`]).catch(() => '');

      if (!psOut) {
        // 컨테이너가 이미 종료됨 → 로그 확인 후 사용자에게 표시
        let logs = '';
        try {
          logs = await runOut('docker', ['logs', CONTAINER_NAME]);
        } catch {
          logs = '(로그 없음)';
        }
        await run('docker', ['rm', '-f', CONTAINER_NAME]).catch(() => {});
        console.error('[Slidev 런처] 컨테이너가 바로 종료됨:', logs);
        return { status: 500, body: {
          error: '컨테이너가 바로 종료되었습니다. Slidev 실행에 실패한 것 같습니다.',
          detail: logs,
        }};
      }

      return { status: 200, body: { ok: true, url: 'http://localhost:3030' } };
    });

    res.status(result.status).json(result.body);
  } catch (err) {
    console.error('[Slidev 런처] Docker 실행 실패:', err.message);
    res.status(500).json({ error: err.message || 'Docker 실행 실패' });
  }
});

app.listen(PORT, () => {
  console.log(`Slidev 런처: http://localhost:${PORT}`);
  console.log(`목록 경로: ${LIST_DIR}`);
  console.log(`compose 파일: ${COMPOSE_FILE}`);
});

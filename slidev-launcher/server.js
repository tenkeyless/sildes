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
const DOCKER_VOLUME_PATH = process.env.HOST_PROJECT_PATH || LIST_DIR;

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

  try {
    // 기존 slidev 컨테이너 중지 (없으면 무시)
    await run('docker', ['stop', 'slidev-m1']).catch(() => {});
    await run('docker', ['rm', 'slidev-m1']).catch(() => {});

    // 새 컨테이너 실행 (-t: TTY 유지, --rm: 종료 시 컨테이너 자동 삭제)
    await run('docker', [
      'run', '-d', '-t', '--rm', '--name', 'slidev-m1',
      '-v', `${DOCKER_VOLUME_PATH}:/slidev`,
      '-p', '3030:3030',
      'my-slidev:m1', 'slidev', file, '--remote',
    ], { cwd: LIST_DIR });

    // node_modules 마운트 방식이면 Slidev만 기동하면 됨 (2~3초면 충분)
    await new Promise((r) => setTimeout(r, 3500));
    const psOut = await runOut('docker', ['ps', '-q', '-f', 'name=slidev-m1']).catch(() => '');

    if (!psOut) {
      // 컨테이너가 이미 종료됨 → 로그 확인 후 사용자에게 표시
      let logs = '';
      try {
        logs = await runOut('docker', ['logs', 'slidev-m1']);
      } catch {
        logs = '(로그 없음)';
      }
      await run('docker', ['rm', '-f', 'slidev-m1']).catch(() => {});
      console.error('[Slidev 런처] 컨테이너가 바로 종료됨:', logs);
      return res.status(500).json({
        error: '컨테이너가 바로 종료되었습니다. Slidev 실행에 실패한 것 같습니다.',
        detail: logs,
      });
    }

    res.json({ ok: true, url: 'http://localhost:3030' });
  } catch (err) {
    console.error('[Slidev 런처] Docker 실행 실패:', err.message);
    res.status(500).json({ error: err.message || 'Docker 실행 실패' });
  }
});

app.listen(PORT, () => {
  console.log(`Slidev 런처: http://localhost:${PORT}`);
  console.log(`목록 경로: ${LIST_DIR}${IN_DOCKER ? ` (호스트 마운트: ${DOCKER_VOLUME_PATH})` : ''}`);
});

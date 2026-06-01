#!/bin/sh
set -e

# 호스트에서 마운트되는 /slidev 에 node_modules 가 없으면 자동 설치.
# 이미지에는 @slidev/cli 만 글로벌로 들어가 있고, 프로젝트 의존성(테마 등)은
# 호스트 package.json 기준으로 여기서 설치한다. 최초 기동 시 1회만 실행.
if [ -f /slidev/package.json ] && [ ! -d /slidev/node_modules ]; then
  echo "[slidev] node_modules 가 없습니다. 의존성 설치 중 (최초 1회, 약 15-30초)..."
  if ! (cd /slidev && npm install --no-audit --no-fund); then
    echo "[slidev] npm install 실패" >&2
    exit 1
  fi
  echo "[slidev] 의존성 설치 완료."
fi

# healthcheck 가 감지할 ready 마커. install 이 성공해야 여기 도달.
touch /tmp/slidev-ready

exec "$@"

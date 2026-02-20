#!/usr/bin/env bash
# 웹 런처를 Docker로 실행 (Node/npm 설치 불필요)
# 사용법: ./run-launcher.sh  (메인 폴더 slides/ 에서 실행)

set -e
cd "$(dirname "$0")"

# 테마 의존성: node_modules 없으면 일회성 컨테이너로 npm install (호스트에 Node 불필요)
if [ ! -d node_modules/prism-theme-vars ]; then
  echo "테마 의존성 설치 중 (최초 1회)..."
  docker run --rm -v "$(pwd):/app" -w /app node:20-slim npm install --omit=dev
fi

# Slidev 이미지가 없으면 빌드
docker image inspect my-slidev:m1 &>/dev/null || docker build -t my-slidev:m1 .
# 런처 이미지가 없으면 빌드
docker image inspect slidev-launcher &>/dev/null || docker build -t slidev-launcher slidev-launcher/

echo "런처: http://localhost:3040"
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd):/workspace" \
  -e WORKSPACE=/workspace \
  -e HOST_PROJECT_PATH="$(pwd)" \
  -p 3040:3040 \
  slidev-launcher

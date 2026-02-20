# 런처 전용 이미지: Node + Docker CLI (호스트 Docker 소켓 마운트로 사용)
FROM node:20-slim

# 호스트 Docker 데몬(API 1.44+)과 맞추기 위해 Docker 공식 저장소에서 CLI 설치
RUN apt-get update && apt-get install -y ca-certificates curl \
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
  && chmod a+r /etc/apt/keyrings/docker.asc \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
  && apt-get update && apt-get install -y docker-ce-cli \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY server.js ./
COPY public ./public

ENV WORKSPACE=/workspace
EXPOSE 3040

CMD ["node", "server.js"]

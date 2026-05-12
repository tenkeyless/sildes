# Apple Silicon(arm64) 지원 Node.js 이미지
FROM node:24-slim

WORKDIR /slidev

# procps: launcher가 docker exec로 이전 slidev 프로세스를 종료할 때 pkill 사용
RUN apt-get update \
  && apt-get install -y --no-install-recommends procps \
  && rm -rf /var/lib/apt/lists/*

# Slidev CLI 설치 (실제 슬라이드는 호스트에서 마운트되는 /slidev 에서 실행)
RUN npm install -g @slidev/cli @antfu/utils

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 3030

# compose.yml에서 command를 sleep infinity로 오버라이드하여 idle 컨테이너로 사용.
# launcher가 docker exec로 그 안에서 slidev를 띄움.
CMD ["slidev", "--remote"]


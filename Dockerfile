# 1. Apple Silicon(arm64) 지원 Node.js 이미지 사용
FROM node:24-slim

# 2. 작업 디렉토리 설정
WORKDIR /slidev

# 3. Slidev 및 관련 종속성 설치 (M1 네이티브 빌드)
# playwright 등의 브라우저는 무거우므로 제외하거나 필요시 추가
RUN npm install -g @slidev/cli @antfu/utils

# 4. 컨테이너 시작 시 /slidev에서 npm install (theme-cnu 등 로컬 테마 의존성)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# 5. 포트 설정
EXPOSE 3030

# 6. 실행 명령 (실제 파일명은 런처/run 시 인자로 전달됨)
CMD ["slidev", "--remote"]


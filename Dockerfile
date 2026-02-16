# 1. Apple Silicon(arm64) 지원 Node.js 이미지 사용
FROM node:20-slim

# 2. 작업 디렉토리 설정
WORKDIR /slidev

# 3. Slidev 및 관련 종속성 설치 (M1 네이티브 빌드)
# playwright 등의 브라우저는 무거우므로 제외하거나 필요시 추가
RUN npm install -g @slidev/cli @antfu/utils

# 4. 포트 설정
EXPOSE 3030

# 5. 실행 명령
# --remote 옵션은 Docker 내부에서 실행될 때 외부 접속을 허용하기 위해 필요합니다.
#ENTRYPOINT ["slidev", "--remote"]
CMD ["slidev", "--remote"]


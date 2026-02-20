# Slidev 슬라이드 (충남대 딥러닝 프로젝트 강의)

[Docker](https://www.docker.com/) 기반 Slidev 슬라이드 실행 환경입니다.

## 요구 사항

- Docker (Apple Silicon M1 호환 이미지 사용)

## Docker 이미지 빌드

이미지가 없다면 먼저 빌드합니다.

```bash
docker build -t my-slidev:m1 .
```

## 슬라이드 실행

### 방법 1: 웹 런처 사용 (권장)

브라우저에서 슬라이드 파일 목록을 보고, 클릭 한 번으로 해당 파일을 Docker로 띄울 수 있습니다. **메인 폴더(slides/)에서** 실행합니다.

```bash
npm run launcher
```

브라우저에서 **http://localhost:3040** 을 열고, 원하는 슬라이드 파일을 클릭하면 Slidev가 실행되며 **http://localhost:3030** 이 새 탭에서 열립니다. (런처 서버는 3040, Slidev는 3030 포트 사용)

### 방법 2: 스크립트 사용 (선택)

웹 런처 대신 터미널에서 파일명만 넘겨서 띄우고 싶을 때 사용합니다. 한 번만 실행 권한을 부여합니다.

```bash
chmod +x run-slidev.sh
./run-slidev.sh Day1-3_BERT_Slides.md
```

### 방법 3: docker run 직접 사용

```bash
docker run --name slidev-m1 --rm -it \
    -v "$(pwd):/slidev" \
    -p 3030:3030 \
    my-slidev:m1 slidev <슬라이드파일.md> --remote
```

실행 후 브라우저에서 **http://localhost:3030** 으로 접속합니다.

## 슬라이드 파일

| 파일명 | 설명 |
|--------|------|
| `Day1-2_TF-IDF_Slides.md` | TF-IDF 슬라이드 |
| `Day1-3_BERT_Slides.md` | BERT 슬라이드 |
| `Day2-1_Seq2Seq_Slides.md` | Seq2Seq 슬라이드 |

## 옵션 설명

- `--name slidev-m1` : 컨테이너 이름
- `--rm` : 종료 시 컨테이너 자동 삭제
- `-it` : 터미널 인터랙티브
- `-v "$(pwd):/slidev"` : 현재 디렉터리를 컨테이너 `/slidev`에 마운트
- `-p 3030:3030` : 호스트 3030 포트로 접속
- `--remote` : Docker 내부에서 실행할 때 외부(호스트)에서 접속 가능하도록 바인딩

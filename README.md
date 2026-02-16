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

프로젝트 루트에서 다음 명령으로 슬라이드를 띄웁니다. **슬라이드 파일명**만 바꿔서 사용하면 됩니다.

```bash
docker run --name slidev-m1 --rm -it \
    -v "$(pwd):/slidev" \
    -p 3030:3030 \
    my-slidev:m1 slidev <슬라이드파일.md> --remote
```

### 예시

- **BERT 슬라이드**
  ```bash
  docker run --name slidev-m1 --rm -it \
      -v "$(pwd):/slidev" \
      -p 3030:3030 \
      my-slidev:m1 slidev Day1-3_BERT_Slides.md --remote
  ```

- **TF-IDF 슬라이드**
  ```bash
  docker run --name slidev-m1 --rm -it \
      -v "$(pwd):/slidev" \
      -p 3030:3030 \
      my-slidev:m1 slidev Day1-2_TF-IDF_Slides.md --remote
  ```

실행 후 브라우저에서 **http://localhost:3030** 으로 접속합니다.

## 슬라이드 파일

| 파일명 | 설명 |
|--------|------|
| `Day1-2_TF-IDF_Slides.md` | TF-IDF 슬라이드 |
| `Day1-3_BERT_Slides.md` | BERT 슬라이드 |

## 옵션 설명

- `--name slidev-m1` : 컨테이너 이름
- `--rm` : 종료 시 컨테이너 자동 삭제
- `-it` : 터미널 인터랙티브
- `-v "$(pwd):/slidev"` : 현재 디렉터리를 컨테이너 `/slidev`에 마운트
- `-p 3030:3030` : 호스트 3030 포트로 접속
- `--remote` : Docker 내부에서 실행할 때 외부(호스트)에서 접속 가능하도록 바인딩

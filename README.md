# Slidev 슬라이드 템플릿

[Slidev](https://sli.dev) 기반 마크다운 슬라이드 템플릿입니다. Docker로 바로 실행할 수 있는 웹 런처와 커스텀 테마(`theme-cnu`)를 포함하고 있습니다.

이 저장소를 **Use this template** 버튼으로 복제하거나 클론한 뒤, [example.md](example.md)를 복사해 새 슬라이드 덱을 만드세요.

## 요구 사항

- **Docker만** 있으면 됩니다 (Node/npm 설치 불필요, Apple Silicon 호환)
- Node가 이미 설치되어 있다면 `npm` 경로로도 실행 가능

## 빠른 시작

### 1) 웹 런처 (권장)

메인 폴더에서 실행하면 `.md` 파일 목록이 브라우저에 떠서 클릭으로 슬라이드를 띄울 수 있습니다.

```bash
chmod +x run-launcher.sh
./run-launcher.sh
```

- 런처: <http://localhost:3040>
- 선택한 슬라이드: <http://localhost:3030>

처음 한 번은 Docker 이미지 빌드와 `npm install`이 자동으로 실행됩니다.

### 2) 단일 파일 실행

특정 파일만 띄우고 싶을 때:

```bash
chmod +x run-slidev.sh
./run-slidev.sh example.md
```

### 3) npm 사용 (Node가 설치된 경우)

```bash
npm run launcher
```

### 4) Docker 이미지 수동 빌드

```bash
docker build -t my-slidev:m1 .
docker build -t slidev-launcher slidev-launcher/
```

## 새 슬라이드 만들기

1. [example.md](example.md)를 복사해 원하는 이름의 `.md` 파일을 만듭니다 (예: `my-deck.md`).
2. 프론트매터에서 `title`과 `info`를 수정합니다.
3. `theme: ./theme-cnu`는 그대로 두면 커스텀 테마가 적용됩니다.
4. `---`로 슬라이드를 구분하고, 슬라이드 앞에 `layout:`을 명시합니다.

웹 런처를 켜둔 상태에서 새 `.md` 파일을 만들면 목록에 자동으로 노출됩니다 (`README*.md`는 제외).

### 자주 쓰는 레이아웃

| 레이아웃 | 용도 |
| :--- | :--- |
| `default` | 일반 제목 + 본문 |
| `center` + `class: text-center` | 섹션 구분용 (제목만 가운데) |
| `two-cols-header` | 좌우 두 칸 (`::left::`, `::right::`) |
| `top_img-bottom_text` | 위 이미지/다이어그램, 아래 텍스트 |
| `two_row_two_column` | 2x2 그리드 |
| `img_caption` | 이미지 + 캡션 |

커스텀 레이아웃 정의는 [theme-cnu/layouts/](theme-cnu/layouts/)에서 확인할 수 있습니다.

### Mermaid 다이어그램

코드 블록을 ` ```mermaid ` 로 열면 자동으로 렌더링됩니다.

````markdown
```mermaid
graph LR
    A[입력] --> B[처리] --> C[출력]
```
````

크기는 ` ```mermaid {scale: 0.8} ` 로 조절할 수 있습니다.

## 디렉터리 구조

```text
.
├── example.md             # 새 슬라이드의 출발점이 되는 예시 파일
├── theme-cnu/             # 커스텀 테마 (폰트, 레이아웃, 스타일)
│   ├── layouts/           # 추가 Slidev 레이아웃 (.vue)
│   └── styles/
├── slidev-launcher/       # 슬라이드 목록 웹 런처 (Node + Express)
├── Dockerfile             # Slidev 실행용 이미지
├── docker-entrypoint.sh
├── run-launcher.sh        # 런처를 Docker로 실행
├── run-slidev.sh          # 단일 슬라이드 파일을 Docker로 실행
└── package.json           # 테마 의존성 (Slidev 자체는 Docker 이미지에 포함)
```

## 라이선스

원하시는 라이선스를 추가하세요.

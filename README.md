# Slidev 슬라이드 템플릿

[Slidev](https://sli.dev) 기반 마크다운 슬라이드 템플릿입니다. Docker로 바로 실행할 수 있는 웹 런처를 포함하고 있습니다. 기본 테마는 [slidev-theme-codecompose](https://www.npmjs.com/package/slidev-theme-codecompose)이며, `package.json`에서 다른 테마로 교체할 수 있습니다.

이 저장소를 **Use this template** 버튼으로 복제하거나 클론한 뒤, [slides/example/](slides/example/)을 복사해 새 슬라이드 덱을 만드세요.

## 요구 사항

- **Docker + Docker Compose v2** (Docker Desktop 또는 `docker compose` 플러그인 포함된 Docker Engine)
- Node/npm 설치 불필요, Apple Silicon 호환

## 빠른 시작

### 1) 최초 빌드 (한 번만)

```bash
docker compose build
```

런처 이미지와 Slidev 이미지를 모두 빌드합니다.

### 2) 기동 (권장)

```bash
docker compose up -d
```

런처(`slidev-launcher`)와 상시 가동되는 Slidev 컨테이너(`slidev-runner`)가 함께 올라옵니다.

> **최초 1회**는 Slidev 컨테이너가 `node_modules/` 가 비어있는 것을 감지해 `npm install` 을 자동 실행하므로 30초~1분 정도 걸립니다. healthcheck가 통과한 뒤에 launcher가 떠서 클릭 가능 상태가 되므로, `docker compose up -d` 가 리턴되면 바로 사용해도 됩니다. 다음 기동부터는 즉시 올라옵니다.

- 브라우저에서 <http://localhost:3040> 을 열고 슬라이드를 클릭
- Slidev는 <http://localhost:3030> 에서 실행
- 다른 슬라이드를 클릭하면 같은 컨테이너 안에서 Slidev 프로세스만 재시작 (컨테이너 자체는 유지 → Vite 캐시 보존으로 두 번째 클릭부터 더 빠름)

### 3) 정리

```bash
docker compose down
```

런처와 Slidev 컨테이너, 네트워크까지 한 번에 정리됩니다.

## 새 슬라이드 만들기

`slides/` 폴더 안에 두 가지 형태 중 하나로 추가합니다.

**(A) 단일 파일** — 이미지/리소스가 거의 없을 때:

```text
slides/my-deck.md
```

**(B) 폴더 + index.md** — 이미지/컴포넌트 등 자산을 함께 두고 싶을 때 (권장):

```text
slides/my-deck/
├── index.md
└── public/           # Slidev가 정적 자산 폴더로 인식
    └── img/foo.png   # 슬라이드에서는 /img/foo.png 로 참조
```

작성 순서:

1. [slides/example/](slides/example/)를 복사해서 원하는 이름의 폴더(또는 파일)로 만듭니다.
2. 프론트매터에서 `title` 등 메타데이터를 수정합니다.
3. `theme: codecompose`를 그대로 두거나, 원하는 [Slidev 테마](https://sli.dev/themes/gallery)로 교체합니다 (`package.json`의 의존성도 함께 교체).
4. `---`로 슬라이드를 구분하고, 필요 시 슬라이드 앞에 `layout:`을 명시합니다.

웹 런처를 켜둔 상태에서 새 파일/폴더를 추가하면 목록에 자동으로 노출됩니다.

### 자주 쓰는 레이아웃 (codecompose)

| 레이아웃 | 용도 |
| :--- | :--- |
| `cover` | 표지 슬라이드 (`coverAuthor`, `coverDate` 등 메타 사용) |
| `table-of-contents` | 자동 목차 (`hideInToc`로 항목 제외) |
| `section` | 파트/섹션 구분 |
| `cols` | 좌우 두 칸 (`::left::`, `::right::`) |
| `bullets` | 불릿 강조 |
| `statement` | 큰 글씨 한 줄/링크 강조 |
| `outro` | 마무리 슬라이드 |
| 생략 | `default` (일반 제목 + 본문) |

다른 Slidev 빌트인 레이아웃(`center`, `two-cols`, `image-right` 등)도 그대로 쓸 수 있습니다. 테마를 교체하면 사용 가능한 레이아웃이 달라지니 해당 테마 문서를 확인하세요.

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
├── slides/                # 슬라이드 덱이 모이는 폴더
│   └── example/           # 예시 덱 (폴더 + index.md 패턴)
│       └── index.md
├── compose.yml            # launcher + slidev 서비스 정의
├── slidev-launcher/       # 슬라이드 목록 웹 런처 (Node + Express)
│   └── Dockerfile         # 런처용 이미지 (Node + docker CLI + compose plugin)
├── Dockerfile             # Slidev 실행용 이미지
├── docker-entrypoint.sh
└── package.json           # Slidev CLI + 테마 의존성
```

런처(`slidev-launcher`)는 사용자가 슬라이드를 클릭하면 호스트 Docker 소켓을 통해 상시 가동 중인 `slidev-runner` 컨테이너에 `docker exec` 로 Slidev 프로세스를 띄우는 구조입니다 ([slidev-launcher/server.js](slidev-launcher/server.js)). 컨테이너를 살린 채 프로세스만 교체하므로 Vite의 dep pre-bundle 캐시가 유지되어, 같은 세션에서 두 번째 클릭부터는 startup이 빨라집니다.

## 라이선스

[MIT](LICENSE)

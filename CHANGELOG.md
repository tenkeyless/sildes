# 변경 이력

이 프로젝트의 모든 주요 변경사항은 이 파일에 기록됩니다.

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따르며,
이 프로젝트는 [Semantic Versioning](https://semver.org/lang/ko/)을 준수합니다.

## 1.0.0 - 2026-05-12

### 추가됨 1.0.0

- **Docker Compose 기반 슬라이드 실행 환경**
  - `docker compose up -d` 한 번으로 launcher + Slidev 컨테이너 동시 기동
  - `docker compose down` 한 번으로 컨테이너·네트워크 일괄 정리
  - Node/npm 호스트 설치 불필요 (Docker만 있으면 동작)
  - Apple Silicon(arm64) 호환
- **슬라이드 목록 웹 런처 (Node + Express)**
  - 브라우저에서 슬라이드 클릭으로 실행 (`http://localhost:3040`)
  - `slides/` 폴더 자동 스캔하여 덱 목록 표시
  - 동시 클릭 직렬화 (요청 race condition 방지)
  - 슬라이드 포트(3030) active polling으로 준비 즉시 응답
- **슬라이드 폴더 구조**
  - `slides/<name>.md` — 단일 파일 패턴
  - `slides/<name>/index.md` — 폴더 패턴 (같은 폴더 `public/` 을 자산 루트로 사용해 이미지·HTML·데모를 함께 보관)
  - 런처 UI에는 덱 이름만 노출 (확장자/경로 숨김)
- **고성능 startup**
  - 상시 가동 Slidev 컨테이너(`slidev-runner`) + `docker exec` 로 프로세스만 교체하는 구조
  - Vite의 dep pre-bundle 캐시가 컨테이너 안에 유지되어 두 번째 클릭부터 ~1.3–1.5초 (첫 클릭 ~2.2초)
  - 클린한 종료 처리 (SIGTERM/SIGINT 핸들러로 slidev 프로세스 정리)
- **slidev-theme-codecompose 테마 기본 적용**
  - 레이아웃: `cover`, `table-of-contents`, `section`, `cols`, `bullets`, `statement`, `outro`
  - `themeConfig`: pagination 위치, brandColor, sectionNav, logo 등 커스터마이징
  - 다른 Slidev 빌트인 레이아웃(`center`, `two-cols`, `image-right` 등)과 호환
- **Example deck (35 슬라이드)**
  - 기본 마크다운 (제목/부제 패턴, 리스트, 표 정렬, 코드 블록, 인용·강조)
  - 레이아웃 데모 (default/cols/bullets/statement/outro)
  - Mermaid 다이어그램 (flowchart, sequenceDiagram, subgraph, `{scale: 0.8}` 옵션, 노드 스타일링)
  - LaTeX 수식 (inline `$...$` / block `$$...$$`, `aligned` 환경)
  - 이미지·iframe 자산 참조 패턴 및 폴더 구조 설명
  - 덱 frontmatter / 슬라이드별 옵션 / 인라인 HTML·CSS 데모
- **MIT 라이선스**

### 기술 사항 1.0.0

- Slidev CLI `^51.0.0`
- slidev-theme-codecompose `^1.0.1`
- Node 24-slim 기반 Docker 이미지
  - `procps` 패키지 (컨테이너 내 프로세스 관리: `pkill`/`pgrep`)
  - `docker-compose-plugin` (런처 컨테이너에서 호스트 daemon 제어)
- Docker Compose v2 필요 (`docker compose` 플러그인)
- 호스트 Docker 소켓 마운트 (`/var/run/docker.sock`) 로 launcher → daemon 제어
- compose 프로젝트명 `slides` 로 고정 (top-level `name:` 필드) — 호스트와 launcher 안에서 같은 네트워크(`slides_default`) 공유

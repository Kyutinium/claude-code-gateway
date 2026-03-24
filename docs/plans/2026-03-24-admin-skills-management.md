# Admin Panel Skills Management Feature

## 개요

`CLAUDE_CWD/.claude/skills/` 디렉토리에 있는 스킬들을 admin 패널에서 전용 UI로 관리하는 기능 추가.

현재 상태: Workspace 탭의 범용 파일 에디터로 `.claude/skills/` 파일 CRUD가 가능하지만, 스킬에 특화된 UX가 없음.

## 현재 인프라 (이미 구현됨)

- `_ALLOWED_DIRS`에 `.claude/skills/` 포함 → 경로 검증 통과
- `/admin/api/files` 엔드포인트로 스킬 파일 읽기/쓰기 가능
- ETag 기반 낙관적 동시성 제어
- 원자적 파일 쓰기 (temp + rename)
- `Skill` 도구가 `DEFAULT_ALLOWED_TOOLS`에 포함

## 스킬 파일 구조

```
.claude/skills/
  └── {skill-name}/
      └── SKILL.md          # YAML frontmatter + Markdown body
```

```yaml
---
name: my-skill
description: 한줄 설명
compatibility: claude-code  # or generic
metadata:
  author: jinyoung
  version: 1.0.0
---

# Skill Documentation (Markdown)
스킬 사용법, 예제 등...
```

## 구현 계획

### Phase 1: Backend — Skills 전용 API (src/admin_service.py)

신규 함수 추가:

| 함수 | 설명 |
|------|------|
| `list_skills()` | `.claude/skills/*/SKILL.md` 스캔, frontmatter 파싱, 메타데이터 목록 반환 |
| `get_skill(name)` | 단일 스킬의 전체 내용 + 파싱된 메타데이터 반환 |
| `create_skill(name, content)` | 디렉토리 생성 + SKILL.md 작성, 이름 중복/유효성 검증 |
| `delete_skill(name)` | 스킬 디렉토리 삭제 (확인 필수) |

구현 세부:
- frontmatter 파싱: `---` 블록을 직접 파싱 (yaml 의존성은 이미 사용 중)
- 스킬 이름 검증: `^[a-z0-9][a-z0-9-]*$` (소문자, 숫자, 하이픈만)
- 기존 `validate_file_path()` 보안 로직 그대로 활용
- 삭제 시 디렉토리 내 모든 파일 제거 (SKILL.md 외 부속 파일 포함)

### Phase 2: Routes — Skills API 엔드포인트 (src/routes/admin.py)

| Method | Endpoint | 동작 |
|--------|----------|------|
| GET | `/admin/api/skills` | 스킬 목록 (name, description, version, author) |
| GET | `/admin/api/skills/{name}` | 스킬 상세 (content + metadata + ETag) |
| PUT | `/admin/api/skills/{name}` | 스킬 생성/수정 (body: content, If-Match 지원) |
| DELETE | `/admin/api/skills/{name}` | 스킬 삭제 |

모두 `require_admin` 의존성 적용.

### Phase 3: Frontend — Skills 탭 UI (src/admin_page.py)

기존 탭 바에 **Skills** 탭 추가 (Workspace와 Sessions 사이):

```
Dashboard | Logs | Rate Limits | Workspace | Skills | Sessions | Config
```

#### UI 구성요소

**스킬 목록 (좌측 사이드바)**
- 카드 형태로 각 스킬 표시: 이름, 설명, 버전
- "새 스킬 만들기" 버튼
- 클릭 시 우측 편집기로 이동

**스킬 편집기 (우측 메인)**
- 상단: 메타데이터 폼
  - 이름 (생성 시만 편집 가능)
  - 설명 (text input)
  - 버전 (text input)
  - 작성자 (text input)
- 하단: SKILL.md 본문 편집 (CodeMirror, markdown 모드)
- 저장/삭제 버튼
- dirty indicator (미저장 변경 표시)

**새 스킬 생성 모달/폼**
- 스킬 이름 입력 (자동 유효성 검사)
- 기본 템플릿으로 SKILL.md 초기화
- 생성 후 편집기로 자동 전환

#### Alpine.js 상태

```javascript
skills: [],           // 스킬 목록
selectedSkill: null,  // 현재 선택된 스킬 이름
skillContent: '',     // 편집 중인 내용
skillEtag: null,      // 낙관적 동시성용
skillDirty: false,    // 변경 감지
skillMeta: {},        // 파싱된 메타데이터
```

### Phase 4: 테스트

| 테스트 파일 | 범위 |
|------------|------|
| `tests/test_admin_skills_unit.py` | list/get/create/delete 로직, frontmatter 파싱, 이름 검증 |
| `tests/test_admin_skills_api.py` | 엔드포인트 통합 테스트, 인증, ETag, 에러 케이스 |

주요 테스트 케이스:
- 빈 디렉토리 → 빈 목록 반환
- frontmatter 없는 SKILL.md → graceful fallback
- 잘못된 스킬 이름 → 400 에러
- ETag 불일치 → 409 Conflict
- 존재하지 않는 스킬 삭제 → 404
- 경로 순회 시도 → 거부

## 변경 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `src/admin_service.py` | `list_skills()`, `get_skill()`, `create_skill()`, `delete_skill()` 추가 |
| `src/routes/admin.py` | `/admin/api/skills` CRUD 엔드포인트 4개 추가 |
| `src/admin_page.py` | Skills 탭 UI (목록 + 편집기 + 생성 폼) |
| `tests/test_admin_skills_unit.py` | 신규 — 서비스 로직 단위 테스트 |
| `tests/test_admin_skills_api.py` | 신규 — API 엔드포인트 통합 테스트 |

## 설계 판단

1. **별도 API vs 기존 files API 재사용**: 별도 API 채택. files API는 범용이라 스킬 메타데이터 파싱/검증 로직을 넣기 어려움.
2. **frontmatter 파싱 라이브러리**: 외부 의존성 없이 직접 파싱. `---` 구분자 + `yaml.safe_load()`로 충분.
3. **스킬 활성화/비활성화**: Phase 1에서는 제외. 파일 존재 여부가 곧 활성 상태. 향후 필요 시 메타데이터에 `enabled: false` 추가 가능.
4. **스킬 내 부속 파일**: SKILL.md만 편집 가능. 부속 파일(예: 스크립트)은 Workspace 탭에서 관리.

## 구현 순서

Phase 1 → Phase 2 → Phase 4 (백엔드 테스트) → Phase 3 → Phase 4 (UI 테스트 보완)

백엔드를 먼저 완성하고 테스트한 뒤, UI를 마지막에 붙이는 전략.

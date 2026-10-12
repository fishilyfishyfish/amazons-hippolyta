# CLAUDE.md — Amazons (Hippolyta)

아마존 게임(Game of the Amazons) 엔진과 분석 사이트. 모든 세션은 이 문서를 기준으로 작업한다.

## 규칙

- 10×10 판. 백 아마존 a4, d1, g1, j4 / 흑 a7, d10, g10, j7. 백 선.
- 한 수 = 아마존 하나를 퀸처럼 이동(빈 칸으로만, 말이나 화살은 통과 불가)
  + 도착 칸에서 퀸처럼 화살 발사(같은 제약, 방금 떠난 칸도 가능).
- 자기 차례에 둘 수 없으면 패배. 무승부 없음.

## 구조

- `engine/`: Rust. 판은 u128 비트보드(칸 번호 = 행×10+열). wasm-bindgen으로 WASM 빌드.
  - `src/board.rs`: 비트보드, 합법수 생성, 수 적용/되돌리기, 종국 판정, 기보 표기, perft
  - `src/naive.rs`: 교차검증용 단순 배열 구현 (비트보드 코드와 공유하는 것은 `Move` 타입뿐)
  - `src/rng.rs`: SplitMix64 (랜덤 AI, 테스트용)
  - `src/wasm.rs`: JS 바인딩 (`Game` 클래스)
  - `tests/movegen.rs`: 규칙/수 생성 테스트, perft 교차검증
  - `examples/bench.rs`: 수 생성 벤치마크
- `web/`: 바닐라 HTML/CSS/JS. 엔진은 Web Worker(`worker.js`)에서만 호출해서 UI가 멈추지 않게.
  - `main.js`: 렌더링과 입력만. 규칙 판단은 워커가 준 합법수 목록으로만 한다.
  - `web/pkg/`: 빌드 산출물(커밋 안 함).
- `scripts/build-wasm.sh`: WASM 빌드 → `web/pkg/`
- `.github/workflows/pages.yml`: 테스트 → WASM 빌드 → GitHub Pages 배포 (main 푸시 시)

## 기보 표기

`출발-도착(화살)`. 예: `d1-d7(g7)`. 파일 a–j, 랭크 1–10 (1이 백 쪽). 파서는 `Move::from_str`.

## 단계

1. 규칙, 합법수 생성, 테스트 — **완료**
2. 최소 웹 UI(사람 vs 랜덤 AI), Pages 배포 — **완료** (Pages 활성화는 수동 필요, 아래 참고)
3. 영역 평가(퀸 거리)와 히트맵
4. MCTS 엔진, 버전 간 자동 대국으로 강도 측정
5. 분석 기능: 평가 그래프, 수순 트리, 추천 수, 기보 입출력
6. 알파제로식 신경망 평가

## 원칙

- 단계마다 동작하는 상태로 커밋. 테스트 통과 전엔 커밋 금지.
- 성능 수치(수 생성 속도, 대국 승률)는 이 문서에 기록.
- main에 바로 커밋/푸시.

## 자주 쓰는 명령

```sh
cd engine && cargo test                          # 빠른 테스트 (~2초)
cd engine && cargo test --release -- --ignored   # perft(3) 교차검증 (~2분)
cd engine && cargo run --release --example bench [-- depth3]
./scripts/build-wasm.sh                          # rustup target add wasm32-unknown-unknown,
                                                 # cargo install wasm-bindgen-cli --version 0.2.129
cd web && python3 -m http.server                 # 로컬 확인 (module worker라 file:// 불가)
```

## 설계 결정

- **칸 번호**: `sq = row*10 + col`, row 0 = 랭크 1, col 0 = 파일 a. 비트 `sq`가 그 칸. 비트 100–127은 항상 0.
- **슬라이딩 생성**: 칸×8방향 레이 테이블(컴파일 타임 `const fn`)을 쓰고, 레이와 점유를 AND해서
  가장 가까운 차단 칸을 증가 방향은 `trailing_zeros`, 감소 방향은 `leading_zeros`로 찾은 뒤
  그 칸 이후 레이를 XOR로 뺀다(클래식 차단 방식). 시프트 방식보다 가장자리 마스크가 필요 없고
  10열 판에서 줄 넘김 버그 여지가 없다.
- **화살 생성**: 출발 칸을 점유에서 뺀 `occ_wo`로 이동 칸과 화살 칸을 모두 계산한다.
  도착 칸은 자기 레이 위에 없으므로 따로 넣을 필요가 없고, 떠난 칸으로 화살을 쏠 수 있다.
- **Move**: `{from, to, arrow}: u8` 3바이트 구조체. 되돌리기는 수 자체만으로 가능하므로
  별도 undo 정보를 저장하지 않는다(`unmake_move(m)`).
- **종국 판정**: "인접 빈 칸이 있는 아마존이 하나라도 있으면 둘 수 있다"로 O(4) 판정.
  인접 칸으로 옮긴 뒤 떠난 칸에 화살을 쏘면 항상 합법이고, 모든 이동은 인접 칸을 지나가기 때문.
- **perft**: 마지막 깊이는 수를 만들지 않고 popcount로 센다(`count_moves`).
- **교차검증**: `naive.rs`는 2차원 배열 + 방향 스텝 루프로, 아마존을 실제로 옮겨 놓고 화살을 찾는다.
  초기 위치 수 집합 비교, perft(2) 비교, 무작위 대국 12판의 매 수 수 집합/수 개수/종국 판정 비교,
  일부 위치에서 perft(2) 비교. perft(3)은 `#[ignore]`로 별도 실행.
- **WASM 경계**: 합법수는 `[from,to,arrow,...]` 평탄 `Uint8Array`(초기 위치 6.5KB)로 넘기고,
  UI가 from→to→arrow 트리를 만들어 하이라이트한다. 매 입력 단계마다 워커를 왕복하지 않아도 된다.
  수를 둘 때는 엔진이 `is_legal`로 다시 검증한다.
- **워커**: 엔진 상태(`Game`)는 워커 하나만 소유. 메인은 id 붙인 요청 → 결과+스냅샷 응답.
  module worker(`type: 'module'`) + `wasm-bindgen --target web` 사용 (번들러 없음).
- **wasm-bindgen 버전 고정**: `=0.2.129`. CLI 버전이 크레이트와 정확히 같아야 하므로
  `Cargo.toml`, `pages.yml`, 위 설치 명령을 같이 바꿔야 한다.
- **무르기**: AI 대국에서는 마지막 사람 수까지 되돌려 다시 사람 차례가 되게 한다(보통 2수).
  사람끼리 모드는 1수.
- **판 방향**: AI 대국에서 사람이 흑이면 판을 뒤집어 사람 쪽이 아래로 오게 한다.
- **입력**: 아마존 탭 → 이동 칸(초록 점) → 화살 칸(빨간 점). 화살 단계에서는 아마존을 도착 칸에
  미리 그려 보여주고, 옮긴 아마존을 다시 탭하면 이동 단계로 돌아간다. 다른 칸 탭/Esc/선택 취소로 해제.
- **반응형**: 세로 화면은 판 위/패널 아래, 가로(가로세로비 ≥ 5/4, 폭 ≥ 700px)는 좌우 배치.
  칸은 `<button>`이라 터치/키보드 모두 동작, 버튼 최소 높이 44px, `touch-action: manipulation`.

## 성능 기록

### 수 생성 (2026-10-12, 1단계)

환경: 클라우드 컨테이너, Intel Xeon @ 2.10GHz, 단일 스레드, `--release` (lto, codegen-units=1), 네이티브.

| 항목 | 결과 |
|---|---|
| `generate_moves` 초기 위치 (2176수, Vec 재사용) | 4.09 µs/호출, 531 M수/s |
| `generate_moves` 무작위 대국 3565개 위치 | 2.07 µs/호출, 225 M수/s |
| perft(2) 비트보드 (bulk count) | 4,307,152 / 0.003s ≈ 1,341 M리프/s |
| perft(2) 단순 배열 | 4,307,152 / 0.034s ≈ 128 M리프/s |
| perft(3) 비트보드 (bulk count) | 8,350,439,170 / 6.73s ≈ 1,240 M리프/s |

검증된 perft 값 (비트보드와 단순 배열 구현 일치): perft(1)=2,176, perft(2)=4,307,152, perft(3)=8,350,439,170.

WASM 쪽 속도는 아직 측정 안 함 (4단계 MCTS 전에 측정할 것).

### 대국 승률

(4단계부터 기록)

## 배포

- main 푸시 → `.github/workflows/pages.yml`이 테스트, WASM 빌드, `web/`을 Pages에 배포.
- 레포 설정에서 Pages 소스가 "GitHub Actions"여야 한다 (`POST /repos/{owner}/{repo}/pages`, `build_type: workflow`).

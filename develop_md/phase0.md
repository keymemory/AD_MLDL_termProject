# Phase 0 — 마스터 로드맵 (VLM Token Compression 논문 실험)

> **연구**: LLaVA-1.5-7B 시각 토큰 2단계 압축 (Stage1 VisPruner 선택 → Stage2 Spherical K-Means 병합).
> 기존 VisPruner(pruning-only)를 select-then-merge로 확장.
> **목표 학회**: AAAI급 (top-tier). 마감 여유 가정, 탄탄함 우선.
> **담당**: 담당자 A(이론·수식·논문) = 본인 / 담당자 B(실험 메인) = 팀원(조희정). 본 로드맵은 A 트랙 + 공용 실험.

---

## 0. 논문의 구조 — 무엇이 정당성이고 무엇이 노벨티인가

이 구분이 전체 작업의 나침반이다. 헷갈리면 여기로 돌아온다.

| 칸 | 내용 | 선점돼도 되나 | 채우는 Phase |
|---|---|---|---|
| **정당성** (기본기) | Stage1 selection 수식화 (왜 이렇게 자르나) | OK (인용+차별화) | Phase 1~2 |
| **노벨티** (기여) | ① select-then-merge 구조 ② task별 압축 민감도 분석 | 안 됨 | Phase 3~5 |

- **수식화는 노벨티가 아니라 정당성**이다. energy/통계 수식이 선점됐어도(E-AdaPrune 등) 상관없다 — 우리 용도는 "method를 단단하게"이지 "새 수식 발명"이 아니다.
- **노벨티는 Stage2 병합 구조 + task별 분석**에서 나온다. Table 4(POPE·GQA 유니모달 / OCR 단조 / SQA 평탄)가 핵심 자산.
- 교수님 피드백 (1)(2)(3) = 전부 **정당성 보강** 요구. 이걸 Phase 1~3에서 처리한다.

---

## 1. 진행 원칙 — 개발/확인/수정/진행 사이클

각 Phase는 **반드시 아래 4단계를 한 사이클로** 돈다. 한 Phase가 끝나야 다음으로 간다.
여러 Phase를 동시에 벌이지 않는다(코드 충돌·회귀 추적 불가 방지).

```
┌─ ① 개발 (Develop) ──────────────────────────────┐
│  Claude Code가 해당 Phase md대로 코드 수정/추가      │
│  새 기능은 항상 플래그로 격리, default=기존 동작        │
└──────────────────────────────────────────────────┘
                    ↓
┌─ ② 확인 (Verify) ───────────────────────────────┐
│  (a) 회귀 검증: 새 기능 OFF일 때 기존 결과 비트 동일      │
│  (b) 동작 검증: 새 기능 ON일 때 의도대로 작동(소규모 job) │
│  → 통과 못 하면 ③으로, 통과하면 본 실험 실행            │
└──────────────────────────────────────────────────┘
                    ↓
┌─ ③ 수정 (Fix) ──────────────────────────────────┐
│  버그·이상치·예상과 다른 결과 분석 후 수정              │
│  수정 후 ②로 되돌아가 재검증 (회귀 깨졌는지 항상 확인)   │
└──────────────────────────────────────────────────┘
                    ↓
┌─ ④ 진행 (Proceed) ──────────────────────────────┐
│  결과를 results.tsv + 분석 노트에 기록                 │
│  → 그 결과를 반영해 다음 Phase md를 작성 (미리 안 만듦)  │
│  → 본인은 병렬로 해당 부분 논문 텍스트 작성             │
└──────────────────────────────────────────────────┘
```

> **핵심 규칙**: 다음 Phase md는 **이전 Phase 결과를 본 뒤** 작성한다.
> 결과가 설계를 바꾸기 때문(예: 수식이 cap에 계속 걸리면 floor/cap부터 수정).
> 미리 다 만들면 십중팔구 다시 고친다.

---

## 2. 전체 Phase 개요

| Phase | 이름 | 핵심 산출 | 교수님 피드백 | 노벨티/정당성 | 상태 |
|---|---|---|---|---|---|
| **P0** | 마스터 로드맵 | 본 문서 | — | — | ✅ |
| **P1** | Selection 수식 구현 | (가)energy/(나)통계 + E2 검증 | (1) 수식 백업 | 정당성 | 📄 md 완성, 실행 대기 |
| **P2** | Selection 정당화 | 엣지 반증(E1) + selection ablation(E3) | (1)(2) | 정당성 | ⏳ P1 결과 후 |
| **P3** | 구조·비율 정당화 | M1:M2 비율(E4) + 복잡도(E5) | (2)(3) | 정당성+노벨티 | ⏳ |
| **P4** | 노벨티 강화 | task별 압축 민감도 분석 (scaling law) | 실험보강 | **노벨티** | ⏳ |
| **P5** | 확장·비교군 | 데이터셋(E6) + baseline 8~10개 | 실험보강 | 정당성 | ⏳ |
| **P6** | 논문 통합 | Intro/Related/Method/Exp 작성 | 전체 | — | ⏳ |

---

## 3. Phase별 상세 — 무엇을, 어떻게, done 조건

### P1 — Selection 수식 구현 (📄 `Phase1_Selection_Formula_Implementation.md`)
- **개발**: `--selection_method {topk, energy, statistical}` 추가.
  - (가) energy: 누적 [CLS] attention 질량 ≥ τ 인 최소 토큰 = important (적분 관점)
  - (나) statistical: μ+kσ 또는 median+k·MAD 이상치 = important (전역 통계 관점)
  - M1·r을 이미지별 자동 결정 (floor=M2, cap=384)
- **확인**: topk default = 기존 VisPruner 비트 동일 (회귀). energy/statistical 동작.
- **실험 E2**: τ 스윕(0.5~0.9) / k 스윕(1.5~2.5) / robust, POPE+GQA. τ\*/k\* 역산 → "고정 M1 = 특수해" 증명.
- **done**: 회귀 통과 + E2 완주 + 고정이 특수해로 설명됨.
- **팀원 차별화**: 팀원=변화율(1차 차분/gain elbow), 우리=적분(energy)·전역통계(statistical). 관점이 다름.

### P2 — Selection 정당화 (⏳ P1 결과 후 작성)
- **E1 엣지 반증** ("엣지만 뽑냐" — 교수님 최강조):
  - E1-a: 선택 토큰 vs Sobel/Canny 엣지맵 overlap(IoU)
  - E1-b: 선택 토큰의 객체 영역 집중도 (bbox 가용 시)
  - E1-c: ★ selection 기준 성능 비교 — CLS attn / Random / L2-norm / Sobel강도 → "엣지강도로 직접 뽑으면 더 나아야 하는데 CLS가 이김" = 직접 반증
- **E3 selection ablation** (구조 정당화):
  - All-tokens(576) / Random+kmeans / **No-selection+kmeans(결정적)** / CLS-only(VisPruner) / CLS+kmeans(Ours)
  - No-selection+kmeans가 "2단계 구조 왜 필요한가" 증명 (centroid 예산 논리)
- **done**: 엣지 반증 데이터 + Stage1 존재 이유 증명.

### P3 — 구조·비율 정당화 (⏳)
- **E4 M1:M2 비율**: M1=64~576 × M2=64. 대부분 Table 4에 있음 → 빈 구간만. %로 재라벨(10/20/40%).
- **E5 복잡도**: selection O(N·d) / k-means O(M1·K·d·I) / LLM O(M2²·D)×32레이어. FLOPs·Mem·Latency 실측. "처리 vs 임베딩 투입" 비교. zero overhead 실증.
- **done**: 비율 근거 + 복잡도 우위 정량화.

### P4 — 노벨티 강화: task별 압축 민감도 (⏳) ★ 페이퍼의 진짜 기여
- Table 4 패턴(유니모달/단조/평탄)이 **더 많은 벤치·task에서 유지되는지** 검증.
- "왜 task마다 다른가" 메커니즘 규명 (semantic vs OCR vs mixed).
- 선점 논문들(CDPruner, E-AdaPrune 등)을 **분석 대상**으로 여러 task에서 돌려 법칙 보편성 확인.
- **done**: "최적 압축이 task 의미구조에 따라 질적으로 다르다"를 정량 입증.

### P5 — 확장·비교군 (⏳)
- 데이터셋: MMStar/MMBench/HallusionBench 중 1~2개.
- baseline 8~10개: FastV, ToMe, SparseVLM, PACT, VisPruner + DivPrune, FasterVLM, VisionZip 등.
- graph/uncertainty 계열 1~2개 비교 분석(Related Work).
- **done**: 일반성·비교 우위 확보.

### P6 — 논문 통합 (⏳)
- Intro / Related Work / Method / Experiments 작성.
- 정당성(P1~3·5) + 노벨티(P3~4) 서술 통합.
- **done**: 투고 가능 초안.

---

## 4. 의존성 — 무엇이 무엇을 막는가

```
P1 (수식 구현) ─── 모든 실험의 전제 (코드 토대)
   │
   ├─→ P2 (정당화) ── P1 수식 위에서 ablation
   │      │
   │      └─→ P3 (구조·복잡도)
   │             │
   └─────────────┴─→ P4 (노벨티 분석) ── P1~3 결과 종합 필요
                          │
                          └─→ P5 (확장) ─→ P6 (논문)
```

- **P1이 안 끝나면 아무것도 못 한다** (코드 토대). 최우선.
- P2·P3는 P1 위에서. P4는 P1~3 결과를 종합해야 의미 있음.
- 논문 텍스트(P6 일부)는 **각 Phase와 병렬로** 미리 써둔다 (본인 트랙).

---

## 5. 역할 분담 (2트랙)

| | 담당자 A (본인) | 담당자 B (팀원) |
|---|---|---|
| 트랙 | 이론·수식·정당화·논문 | 실험 메인·gain elbow 방법 |
| Phase | P1(수식)·P2·P4·P6 주도 | 실험 실행·B-series ablation |
| 수식 | energy/statistical (적분·통계) | attention gain/cosine gain (변화율) |
| 합류점 | 주말 미팅: "방법(B) + 그게 왜 되는지(A)"로 통합 |

> 두 트랙의 수식이 **다른 수학 관점**이라 충돌이 아니라 비교 자산이 됨
> ("변화율 vs 적분 vs 전역통계, 어느 selection 기준이 타당한가" = 논문 한 섹션).

---

## 6. 공통 안전 수칙 (모든 Phase 적용)

1. **회귀 먼저**: 새 기능 OFF = 기존 동작 비트 동일. 검증 전 본 실험 금지.
2. **플래그 격리**: 새 기능은 항상 CLI 플래그 + getter `getattr(self,"x",default)`, default=기존.
3. **반환 shape 규약**: Stage2 ON `(B,M2,D)+None` / OFF `(B,576,D)+mask`.
4. **transformers 4.37.2 고정**: LLM 디코더 미개조, 시각토큰 단(CLIP~projector)에서만.
5. **경로 패치 선행**: 매 실행 환경에서 §0(Phase1 md) 깨진 경로 3종 확인.
6. **결과 기록**: results.tsv 스키마 준수 + Phase별 분석 노트 작성.
7. **다음 Phase는 이전 결과 후 작성**: 미리 만들지 않는다.

---

## 7. 현재 위치 & 다음 액션

- ✅ P0 마스터 로드맵 (본 문서)
- 📄 P1 md 작성 완료 → **다음: Claude Code에 P1 넘겨 ①개발 ②확인**
- ⏳ P1 실행 중 본인은 병렬로 **Method 텍스트(종류1 정당화 + A-2 수식 서술)** 작성
- ⏳ P1 결과 확인 후 → P2 md 작성

> **다음 즉시 액션**: `Phase1_Selection_Formula_Implementation.md`를 VSCode Claude Code에 전달.
> 단, 전달 시 강조: (1) `model_vqa_loader`의 batch_size 먼저 확인 (가변 길이 처리 분기),
> (2) §0 경로 패치 먼저, (3) §3 회귀 검증 통과 전 실험 금지.

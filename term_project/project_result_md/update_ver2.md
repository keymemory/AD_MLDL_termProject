## 목표
원본 VisPruner와 제안 구조의 데이터셋 커버리지를 완전히 일치시킨다.
1. VQAv2 → 원본 VisPruner 코드(VisPruner_run)에서 추가 실행
2. SQA-IMG → 제안 구조 코드(term_project)에서 추가 실행
실험 완료 후 기존 결과 문서들에 결과를 통합한다.

## 현재 상태

| 데이터셋 | 원본 (VisPruner_run) | 제안 (term_project) |
|:---|:---:|:---:|
| POPE | ✅ | ✅ |
| GQA | ✅ | ✅ |
| TextVQA | ✅ | ✅ |
| VQAv2 | ❌ ← 이번에 실행 | ✅ |
| SQA-IMG | ✅ | ❌ ← 이번에 실행 |

## 환경
- 원본 코드: experiments/VisPruner_run/ (POPE/GQA/TextVQA/SQA-IMG 재현 완료된 환경)
- 제안 코드: experiments/term_project/ (구현 완료, 검증 완료)
- 데이터셋: experiments/dataset/vqav2/, experiments/dataset/scienceqa/ (둘 다 준비됨)
- GPU: RTX A6000, fp16, greedy, CUDA_LAUNCH_BLOCKING=1
- builder.py dtype 패치, resume 모두 적용됨

---

## 실험 A: VQAv2 → 원본 VisPruner (VisPruner_run)

제안 구조 실험에서 사용한 것과 **동일한 VQAv2 val 균형 subset 6000**을 사용한다.
(yes-no/number/other 각 2000, 제안 구조의 A시리즈와 직접 비교 가능하도록)

제안 구조 실험에서 VQAv2 val subset을 어떻게 구성했는지 확인하고 동일하게 적용.

### 실행 목록

| ID | visual_token_num | important_ratio | 비고 |
|----|--:|--:|---|
| V-576 | 576 | 0.5 | baseline (프루닝 없음) |
| V-128 | 128 | 0.5 | ↓77.8% |
| V-64 | 64 | 0.5 | ↓88.9% |
| V-32 | 32 | 0.5 | ↓94.4% |

총 4회. 기존 VisPruner_run 환경에서 eval 스크립트 방식 그대로 실행.
VQAv2 eval 방식은 제안 구조에서 사용한 로컬 채점과 동일하게 맞춘다.

### 결과 검증
- V-128, V-64, V-32 결과가 제안 구조의 A시리즈(72.18, 68.88, 63.47)와
  유사해야 함 (동일 VisPruner 알고리즘이므로)
- 차이가 크면(±2점 이상) 원인 분석

---

## 실험 B: SQA-IMG → 제안 구조 (term_project)

기존 실험과 동일한 세팅으로 SQA-IMG를 실행한다.
r = 0.5, 평가 스크립트는 기존 SQA-IMG eval 방식 그대로.

### 실행 목록

| ID | Method | enable_clustering | stage1(M1) | final(M2) | merge_method |
|----|--------|:-:|--:|--:|---|
| A-128 | VisPruner only | OFF | - | 128 | - |
| A-64 | VisPruner only | OFF | - | 64 | - |
| A-32 | VisPruner only | OFF | - | 32 | - |
| B-128 | Ours simple | ON | 192 | 128 | simple_avg |
| B-64 | Ours simple | ON | 128 | 64 | simple_avg |
| B-32 | Ours simple | ON | 64 | 32 | simple_avg |
| C-128 | Ours weighted | ON | 192 | 128 | weighted_avg |
| C-64 | Ours weighted | ON | 128 | 64 | weighted_avg |
| C-32 | Ours weighted | ON | 64 | 32 | weighted_avg |

총 9회. SQA-IMG는 2017문항으로 가벼움.

### 결과 검증
- A시리즈 결과가 원본 VisPruner 재현값(68.86, 68.57, 68.32)과
  유사해야 함 (동일 VisPruner 알고리즘이므로)
- 차이가 크면(±2점 이상) 원인 분석

---

## 데이터 경로
- VQAv2: experiments/dataset/vqav2/ (제안 구조에서 사용한 val subset 위치 확인)
- SQA-IMG: experiments/dataset/scienceqa/ (원본 재현에서 사용한 구조 참고)
- 경로가 안 잡히면 심볼릭 링크 확인/수정

## 에러 대응
- 에러 발생 시 원인 분석 → 스스로 수정 → 재실행
- 수정 내역 기록

---

## 산출물 — 기존 문서에 통합

### 1. vispruner_md/06_full_reproduction_results.md 업데이트
원본 VisPruner 재현 결과에 VQAv2 행 추가:

| Benchmark | Metric | 논문576 | 재현576 | 논문128 | 재현128 | 논문64 | 재현64 | 논문32 | 재현32 |
|-----------|--------|--------|--------|--------|--------|-------|-------|-------|-------|
| VQAv2 | Acc | 78.5 | **?** | 75.8 | **?** | 72.7 | **?** | 67.7 | **?** |

(기존 POPE/GQA/TextVQA/SQA-IMG 결과와 나란히)

### 2. test_result_md/02_main_results.md 업데이트
제안 구조 실험 1 결과표에 SQA-IMG 열 추가:

| M2 | Method | POPE | GQA | TextVQA | VQAv2 | **SQA-IMG** |
|----|--------|------|-----|---------|-------|-------------|
| 128 | A | 84.47 | 58.28 | 56.76 | 72.18 | **?** |
| 128 | B simple | 85.37 | 58.26 | 54.77 | 72.08 | **?** |
| ... | ... | ... | ... | ... | ... | **?** |

개선폭(B−A) 분석에도 SQA-IMG 추가.

### 3. test_result_md/02_main_results.md Baseline 종합 비교 업데이트
SQA-IMG 비교표 추가 (VisPruner 논문 Table 1 인용값 포함):

| Method | 128 | 64 | 32 |
|--------|-----|----|----|
| FastV (인용) | 60.2 | 51.1 | 42.6 |
| VisPruner (논문) | 69.1 | 69.1 | 69.2 |
| A VisPruner-only | ? | ? | ? |
| B Ours simple | ? | ? | ? |
| B Ours weighted | ? | ? | ? |

### 4. test_result_md/04_question_type_analysis.md 업데이트
벤치마크별 clustering 효과 종합에 SQA-IMG 추가:
- POPE: 환각 → 이득 큼
- GQA: 공간관계 → 저토큰 이득
- VQAv2: question type별 차이
- TextVQA: OCR → 소폭 하락
- **SQA-IMG: 상식 추론 → ? (실험 결과로 채움)**

### 5. 최종 커버리지 표 (02_main_results.md 상단에 추가)

| 데이터셋 | 원본 VisPruner | 제안 구조 |
|:---|:---:|:---:|
| POPE | ✅ | ✅ |
| GQA | ✅ | ✅ |
| TextVQA | ✅ | ✅ |
| VQAv2 | ✅ | ✅ |
| SQA-IMG | ✅ | ✅ |

### 6. test_result_md/06_experiment_log.md 업데이트
VQAv2(원본) + SQA-IMG(제안) 실행 명령어, 소요 시간, 에러 로그 추가.
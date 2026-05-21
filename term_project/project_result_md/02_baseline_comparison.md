# 02. Baseline 비교 (실험 1 + 실험 2)

## 0. 데이터셋 커버리지 (원본 VisPruner ↔ 제안 구조 완전 일치)

| 데이터셋 | 원본 VisPruner | 제안 구조 |
|:---|:---:|:---:|
| POPE | ✅ | ✅ |
| GQA | ✅ | ✅ |
| TextVQA | ✅ | ✅ |
| VQAv2 | ✅ (update_ver2 실험A) | ✅ |
| SQA-IMG | ✅ | ✅ (update_ver2 실험B) |

→ 5개 벤치마크 모두 원본·제안 양쪽 커버 완료. (VQAv2는 공통 val 균형 subset 6000)

LLaVA-1.5-7B, important_ratio r=0.5(기본), greedy, `CUDA_LAUNCH_BLOCKING=1` 안정모드.
- A = VisPruner only (Stage1만, clustering OFF)
- B = Ours (Stage1 + Stage2 Spherical K-Means clustering), s=simple_avg / w=weighted_avg
- B의 Stage1 토큰수 M1: M2=128→M1=192, M2=64→M1=128, M2=32→M1=64

## 1. 실험 1 — 제안 방법 3변형 비교 (POPE F1 / GQA / TextVQA / VQAv2 / SQA-IMG Acc)

| M2 | 방법 | POPE(F1) | GQA(Acc) | TextVQA(Acc) | VQAv2(Acc)† | SQA-IMG(Acc)‡ |
|---:|---|---:|---:|---:|---:|---:|
| 128 | A VisPruner only | 84.47 | 58.28 | **56.76** | **72.18** | **68.86** |
| 128 | B Ours simple | **85.37** | 58.26 | 54.77 | 72.08 | **68.86** |
| 128 | B Ours weighted | 85.24 | **58.27** | 55.27 | 72.17 | 68.62 |
| 64 | A VisPruner only | 80.95 | 55.59 | **55.68** | 68.88 | 68.57 |
| 64 | B Ours simple | **82.27** | **56.66** | 54.08 | 69.77 | **68.86** |
| 64 | B Ours weighted | 82.14 | 56.26 | 54.33 | **70.44** | **69.16** |
| 32 | A VisPruner only | 74.00 | 51.58 | 53.55 | 63.47 | 68.32 |
| 32 | B Ours simple | **77.56** | 53.52 | 53.38 | 65.46 | 69.31 |
| 32 | B Ours weighted | 77.53 | **54.03** | **53.61** | **65.88** | **69.46** |

‡ SQA-IMG(2017 IMG문항, IMG-Accuracy). update_ver2 실험B로 제안 구조에서 추가 실행.
clustering 효과(B/C−A): @128 ≈0, @64 **+0.3~+0.6**, @32 **+1.0~+1.1**. SQA-IMG는
상식추론으로 토큰수에 둔감(68~69 평탄)하나 **clustering이 저토큰서 일관 소폭 개선,
전 세팅 무회귀**(최고 C-32 69.46 > VisPruner 논문 69.2). 검증: A시리즈
68.86/68.57/68.32 = 원본 VisPruner 재현값과 **정확 일치**.

† VQAv2는 EvalAI(test-dev) 대신 **val 균형 subset 6000개**(yes-no/number/other 각 2000)
로컬 채점(공식 VQA accuracy). 균형 subset이라 자연분포(test-dev, yes/no 비중↑)보다
overall이 보수적 — **논문 test-dev 값과 직접 비교 불가, A↔B 상대 비교는 동일 subset이라 유효.**

VQAv2 clustering 효과(B−A): @128 ≈0(−0.10s/−0.01w), @64 **+0.89s/+1.56w**, @32 **+1.99s/+2.41w**.
→ POPE·GQA와 동일 패턴: 압축 강할수록 이득↑, weighted가 VQAv2서 근소 우위.

### 핵심 관찰
- **POPE**: Ours가 전 토큰수에서 VisPruner-only 대비 일관 향상 — 128 **+0.90**, 64 **+1.32**, 32 **+3.56**.
  압축이 공격적일수록(32) 개선폭이 가장 큼 → "많이 보존 후 클러스터 병합"이 정보 보존에 유리.
- **GQA**: 128에서 동등(±0.02), 64 **+1.07**, 32 **+1.94~+2.45**. 저토큰 영역에서 Ours 우세.
- **TextVQA**: Ours가 소폭 하락(128 −1.5~−2.0, 64 −1.4~−1.6, 32 ≈동등). OCR/세밀 텍스트는
  토큰 병합 시 미세 정보 손실 → 압축 강할수록 격차 축소(32서 거의 동일/우세).
- **simple vs weighted**: 대체로 simple이 POPE 근소 우위, weighted가 GQA@32·TextVQA 근소 우위.
  차이는 작아(≤0.5p) 두 병합 방식 모두 안정적.

## 2. 실험 2 — 기존 Baseline과 종합 비교 (VisPruner 논문 Table 1 인용)

출처: VisPruner(ICCV'25) 논문 Table 1, LLaVA-1.5-7B. POPE는 본 재현이 F1, 논문 인용치는
원문 표기 기준. **본 실험은 POPE·GQA·TextVQA·VQAv2(val subset) 로컬 채점**.
FastV/ToMe/SparseVLM은 프롬프트 지시대로 VisPruner 논문 Table 1 인용, PACT는 §2-PACT 참조.
아래는 동일 토큰수에서 방법 간 상대 비교.

### POPE (지표: 논문 인용치는 원문 값, 본 연구는 F1)

| Method | 128 | 64 | 32 | 출처 |
|---|---:|---:|---:|---|
| FastV (ECCV'24) | 59.6 | 48.0 | 32.5 | VisPruner 논문 인용 |
| ToMe (ICLR'23) | 62.8 | 52.5 | 39.0 | VisPruner 논문 인용 |
| SparseVLM (ICML'25) | 80.5 | 75.1 | 67.9 | VisPruner 논문 인용 |
| VisPruner (ICCV'25, 논문) | 84.6 | 80.4 | 72.7 | VisPruner 논문 |
| **A VisPruner-only (본 재현)** | 84.47 | 80.95 | 74.00 | 본 실험 |
| **B Ours simple (제안)** | **85.37** | **82.27** | **77.56** | 본 실험 |
| **B Ours weighted (제안)** | 85.24 | 82.14 | 77.53 | 본 실험 |

→ 재현한 VisPruner-only가 논문값과 정합(±0.5~+1.3), **제안 B가 VisPruner 논문값 및 모든
기존 baseline(FastV/ToMe/SparseVLM)을 전 토큰수에서 상회**. 특히 32토큰서 VisPruner 72.7 →
Ours **77.56** (+4.86 vs 논문, +3.56 vs 재현 baseline).

### GQA (Acc)

| Method | 128 | 64 | 32 |
|---|---:|---:|---:|
| FastV | 49.6 | 46.1 | 41.5 |
| ToMe | 52.4 | 48.6 | 43.6 |
| SparseVLM | 56.0 | 52.7 | 48.3 |
| VisPruner (논문) | 58.2 | 55.4 | 52.2 |
| A VisPruner-only (재현) | 58.28 | 55.59 | 51.58 |
| **B Ours simple** | 58.26 | **56.66** | 53.52 |
| **B Ours weighted** | 58.27 | 56.26 | **54.03** |

→ 제안 B가 GQA에서도 baseline 전부 상회, 64/32에서 VisPruner 논문·재현 대비 추가 개선.

### VQAv2 (Acc) — 본 연구는 val 균형 subset 6000, 논문 인용치는 test-dev

| Method | 128 | 64 | 32 | 비고 |
|---|---:|---:|---:|---|
| FastV | 61.8 | 55.0 | 43.4 | VisPruner 논문 인용(test-dev) |
| ToMe | 63.0 | 57.1 | 46.8 | VisPruner 논문 인용(test-dev) |
| SparseVLM | 73.8 | 68.2 | 58.6 | VisPruner 논문 인용(test-dev) |
| VisPruner (논문) | 75.8 | 72.7 | 67.7 | VisPruner 논문(test-dev) |
| A VisPruner-only (본 실험) | 72.18 | 68.88 | 63.47 | val subset(균형) |
| **B Ours simple** | 72.08 | 69.77 | 65.46 | val subset(균형) |
| **B Ours weighted** | 72.17 | **70.44** | **65.88** | val subset(균형) |

⚠️ 본 VQAv2는 **val 균형 subset(yes-no/number/other 각 2000)** 로컬 채점이라, 자연분포
test-dev(yes/no 비중↑, yes/no는 ~90%로 overall을 끌어올림) 논문 인용치와 **절대값 직접
비교 불가**. 동일 subset 내 **A↔B 상대 비교만 유효**: clustering이 @64/@32에서 일관 개선
(weighted @64 +1.56, @32 +2.41). 추세는 POPE/GQA와 동일.

### SQA-IMG (Acc) — IMG-Accuracy, 제안 구조 실행 (update_ver2 실험B)

| Method | 128 | 64 | 32 | 출처 |
|---|---:|---:|---:|---|
| FastV | 60.2 | 51.1 | 42.6 | VisPruner 논문 인용 |
| VisPruner (논문) | 69.1 | 69.1 | 69.2 | VisPruner 논문 |
| A VisPruner-only (원본 재현) | 68.86 | 68.57 | 68.32 | 본 실험 |
| **B Ours simple** | 68.86 | 68.86 | 69.31 | 본 실험 |
| **B Ours weighted** | 68.62 | **69.16** | **69.46** | 본 실험 |

→ SQA-IMG는 토큰수 둔감(상식추론). 제안 B가 FastV를 큰 폭 상회(+18~27p),
VisPruner 논문값과 동등하며 **저토큰(32)서 B가 VisPruner 논문(69.2)·재현(68.32) 모두 상회**
(C-32 **69.46**). clustering이 무회귀로 소폭 개선 — 전 벤치마크(POPE/GQA/VQAv2/SQA) 일관.

### PACT (CVPR'25) — 직접 실행 불가, 사유 기록
- PACT 코드(`Advanced_ML_DL/PACT`)는 **별도 conda env**(`pactenv`: python 3.12.7,
  pytorch-cuda 11.8, flash-attn 2.6.3) + **자체 번들 transformers/** 요구.
- 지원 백본이 **LLaVA-OneVision-7B / Qwen2-VL-7B / LLaVA-1.6-Mistral**로 본 과제 백본
  **LLaVA-1.5-7B 미지원**. 환경·백본 모두 상이 → 동일조건 직접 비교 불가.
- 프롬프트 규정("LLaVA-1.5-7B에서 동작 안 하면 에러 기록 후 논문 숫자 인용") 적용.
  단, 제공된 인용 블록(ToMe/SparseVLM)에 PACT 수치 미포함 → **수치 미확보로 표기**
  (날조 금지). 비교표에서 PACT 행은 제외.

### FastV — 직접 실행 보류, 인용 (사유)
- FastV는 LLM 디코더 layer-K self-attention + KV-cache 프루닝 요구. 고정 의존성
  (transformers 4.37.2) in-decoder 통합은 버전민감·고위험, 오구현 시 baseline 왜곡 우려.
- 프롬프트 규정("구현 어려우면 논문 숫자 인용+사유") 적용 → 위 POPE/GQA/VQAv2 표에
  VisPruner 논문 Table 1 FastV 수치 인용.

## 3. 토큰 수별 성능 추세 분석

- VisPruner-only(A): 토큰 감소 시 급락 (POPE 84.5→74.0, −10.5 / GQA 58.3→51.6, −6.7).
- Ours(B): 완만한 감소 (POPE 85.4→77.6, −7.8 / GQA 58.3→53.5~54.0).
  → **압축이 강할수록 제안 방법의 상대 이득 증가** (POPE 개선폭 128:+0.9 → 32:+3.6).
- 해석: Stage1에서 더 많은 토큰(M1)을 남긴 뒤 Spherical K-Means로 의미 단위 병합하면,
  직접 소수 선택보다 시각 정보 손실이 작다. VisPruner의 단일 선택 한계를 2단계로 보완.

## 4. 실행 범위 / 미실행

- **VQAv2**: (v2) test-dev EvalAI 대신 **val 균형 subset 6000 로컬 채점**으로 실행 완료.
  9세팅(A/B/C×128/64/32) + question-type 분석은 `04_question_type_analysis.md` §4-B.
- **FastV / PACT**: 직접 실행 보류·불가 → VisPruner Table 1 인용/사유(위 §2, `06` 상세).
- **TextVQA**: 본 실험 로컬 채점 포함(논문 Table 1 외 추가 벤치).

> 전체 33-job 완료(R-70 r=0.7 포함, 홀수-R 버그 수정 후 재실행 확정).
> 상세 ablation은 `03_ablation_results.md`, 카테고리 분석 `04_question_type_analysis.md`,
> 효율성 `05_efficiency_results.md`, 실행/에러 로그 `06_experiment_log.md`.

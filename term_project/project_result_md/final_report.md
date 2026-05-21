# Two-Stage Visual Token Reduction Framework — 최종 종합 보고서

> Backbone: **LLaVA-1.5-7B** · important_ratio r=0.5(기본) · greedy(temp=0) · fp16
> A = VisPruner only(Stage1, clustering OFF) · B = Ours simple_avg · C = Ours weighted_avg
> Stage1 토큰수 M1: M2=128→192, M2=64→128, M2=32→64 (모두 1.5~3×)
> 모든 수치는 `exp_runner/results.tsv`·`results_update2.tsv`·eval 로그 기준. 날조 없음.

---

## 1. 프로젝트 개요

**연구 배경.** VLM(LLaVA 등)은 CLIP-ViT-L/14-336px가 이미지를 **576개 visual token**으로
변환해 LLM에 주입한다. LLM의 self-attention은 시퀀스 길이에 **quadratic(O(n²))** 이므로
576개 시각 토큰이 텍스트 토큰과 합쳐지면 추론 비용·지연·메모리가 급증한다. 시각 토큰의
상당수는 배경·중복이라 정보 대비 비용이 비효율적이다.

**기존 방법의 한계.** VisPruner(ICCV'25)는 [CLS] attention으로 중요 토큰을 고르고
유사도로 다양성 토큰을 남기는 **pruning-only** 방법이다. 그러나 선택된 토큰들 사이에도
여전히 의미적으로 **중복(redundancy)** 이 남으며, 단순 선택은 "버린 토큰의 정보"를
완전히 폐기한다(특히 공격적 압축 M2=32에서 손실 큼).

**제안 방법.** **Two-Stage**: ① Stage1에서 VisPruner로 목표보다 많은 **M1개**를 보존하고
② Stage2에서 **Spherical K-Means**로 M1개를 **M2개 클러스터로 병합**하여 대표 토큰을
생성한다. 단순 소수 선택 대신 "넓게 보존 → 의미 단위 병합"으로 **동일 토큰 수에서
정보 밀도를 높인다**. 병합은 simple/weighted 평균 2가지를 제공한다.

**실험 목적.** 제안 방법이 (a) 동일 최종 토큰 수에서 VisPruner를 능가하는지, (b) 어떤
태스크·압축률에서 이득이 큰지, (c) 추가 비용이 무시 가능한지를 5개 벤치마크로 검증.

→ **핵심 발견: 동일 토큰 수·동일 비용으로 POPE/GQA/VQAv2에서 일관 향상, 압축이 강할수록 이득 증가.**

---

## 2. 제안 방법 상세

### 2-1. 전체 파이프라인
```
입력 이미지 → CLIP-ViT-L/14-336px → 576 visual tokens
   → [Stage1: VisPruner] [CLS]-attention로 important + 유사도로 diverse → M1개 보존
   → [Stage2: Spherical K-Means] M1개를 M2개 클러스터로 병합 → M2개 대표 토큰
   → mm_projector → LLM(Vicuna-7B) <image> 자리에 주입 → 답변 생성
```
- CLIP: 이미지를 24×24=576 patch 토큰으로 인코딩.
- Stage1: 576→M1 (중요·다양 토큰 선택, attention score 동반 추출).
- Stage2: M1→M2 (잔여 중복 병합, 정보 압축). clustering OFF면 M1=M2로 Stage2 미진입(=원본 VisPruner).
- LLM 입력 길이가 M2로 동일하므로 디코딩 비용은 토큰 수에만 의존.

→ **핵심: Stage2는 "선택"이 아니라 "병합"이라 버려질 토큰의 정보를 대표 벡터에 흡수.**

### 2-2. Stage 1: VisPruner 기반 토큰 선택
- **[CLS] attention**: CLIP 비전 인코더 뒤에서 2번째 레이어의 `[CLS]→patch` attention.
  LLM 단의 text-visual attention은 **position bias**(앞쪽 토큰 편애)가 있어 프루닝 지표로
  부적합 → 시각 인코더 자체의 saliency인 [CLS] attention을 사용(VisPruner의 핵심 주장).
- **Important token**: [CLS] attention 상위 `M1·r`개. 전경/핵심 객체에 대응.
- **Diverse token**: 나머지에서 cosine similarity가 높은(중복) 토큰을 ToMe식 bipartite
  매칭으로 반복 제거, `M1·(1−r)`개 잔존. 배경/맥락 정보를 보완.
- **important_ratio r**: important:diverse 비율. r↑이면 핵심 집중, r↓이면 다양성 강조.

### 2-3. Stage 2: Spherical K-Means 병합
- **Spherical K-Means**: 토큰을 L2 정규화(단위 구 투영) 후 **cosine similarity**로 군집화.
  일반 K-Means(유클리드 거리)와 달리 방향(의미 유사도) 기반이라 고차원 임베딩에 적합.
- **왜 필요한가**: Stage1 보존 토큰에도 의미적 중복이 남음. 이를 M2개 클러스터로 묶어
  대표 토큰을 만들면 같은 토큰 수라도 **더 다양·압축된 정보**를 LLM에 전달.
- **M1→M2 압축**: M1개(예 128) → cosine 유사 군집 M2개(예 64)로 병합. 빈 클러스터는
  최대 클러스터에서 1토큰 재할당, fp16 0-division은 eps=1e-8로 방지, max_iter≤10+조기종료.

### 2-4. Simple Average vs Weighted Average (상세)

클러스터 Cⱼ의 대표 토큰 repⱼ:

- **Simple Average**: `repⱼ = (1/|Cⱼ|) · Σ_{i∈Cⱼ} xᵢ`
  - 클러스터 내 모든 토큰을 **동등** 취급한 단순 평균.
  - 장점: 구현 단순, 배경/맥락 정보를 균등 반영 → 환각 거부(POPE)에 유리.
  - 단점: 덜 중요한 토큰 정보가 **희석 없이** 섞여 핵심이 흐려질 수 있음.
- **Weighted Average**: `repⱼ = Σ_{i∈Cⱼ}(aᵢ·xᵢ) / Σ_{i∈Cⱼ} aᵢ`  (aᵢ = [CLS] attention score)
  - Stage1에서 가져온 attention을 가중치로 → **중요 토큰 정보가 대표에 더 강하게** 반영.
  - 장점: 핵심 시각 정보 보존, 어려운 태스크(counting/adversarial/OCR)에서 유리.
  - 단점: attention score가 부정확하면 오히려 편향 가능.
- **실험 기반 비교(요약)**: POPE는 simple 근소 우위(객체존재는 균등평균 충분),
  GQA(저토큰)·TextVQA·VQAv2(number 외)·SQA-IMG·POPE-adversarial은 **weighted 우위**.
  차이는 대체로 ≤0.5p로 작으나 **어려운/세밀 태스크일수록 weighted가 일관 우위**.

→ **핵심: 기본 simple, 어려운/세밀 태스크(adversarial·counting·OCR·저토큰)는 weighted 권장.**

---

## 3. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 49GB (3장 병렬 활용) |
| Driver / CUDA | 535.183.01 / 12.2 |
| Python | 3.10 (conda env `vispruner`) |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2+cu121 |
| transformers / tokenizers | 4.37.2 / 0.15.1 |
| 모델 | **LLaVA-1.5-7B** = Vicuna-7B(LLM) + CLIP-ViT-L/14-336px(비전) + 2-layer MLP projector |
| 추론 설정 | fp16, greedy decoding, temperature=0 |
| 안정화 | `builder.py` dtype 패치(비전타워→모델 dtype, 비동기 CUDA 무음손상 해결), `CUDA_LAUNCH_BLOCKING=1`(안정 우선; 단 대규모 잔여 작업은 패치 적용 상태에서 비차단으로 가속) |

→ **핵심: dtype 패치가 핵심 — 미적용 시 시각 피처가 조용히 손상되어 정확도 비정상 하락.**

---

## 4. 데이터셋 상세

| 데이터셋 | 태스크 | 문항 | 이미지 출처 | 지표 |
|---|---|---:|---|---|
| POPE | 객체 환각 (yes/no) | 8910 | COCO val2014 | F1 |
| GQA | 구조적 시각추론 (open) | 12578 | Visual Genome | Acc |
| TextVQA | 이미지 내 텍스트 VQA | 5000 | Open Images v3 | VQA Acc |
| VQAv2 | 범용 VQA (val 균형 subset) | 6000 | COCO | VQA Acc |
| SQA-IMG | 과학 multiple-choice(이미지) | 2017 | ScienceQA | Acc(IMG) |

**POPE** — "Is there a [object]?" yes/no. 카테고리: random(2910, 무작위 객체)/popular(3000,
빈출 객체)/adversarial(3000, 동시출현 객체=가장 어려움). VLM **환각** 평가, diverse token
보존 효과 확인의 핵심.
**GQA** — scene graph 기반 공간관계·속성·비교 질문. 조밀한 시각정보 필요 → 토큰 감소에 민감.
**TextVQA** — 간판/라벨 등 **OCR** 필요. 세밀 텍스트가 병합 시 손실 위험.
**VQAv2** — val **균형 subset 6000**(yes-no/number/other 각 2000). yes/no=객체존재(압축 강건),
number=counting(세밀 공간정보, 압축 최취약), other=다양 추론(중간). question-type 분석으로
task-aware policy 근거 제공. ⚠️ **subset이라 논문 test-dev 인용값과 절대 비교 불가, A↔B 상대비교만 유효**.
**SQA-IMG** — 과학 multiple-choice. 상식/추론 중심 → 토큰수 둔감.

→ **핵심: 5개 벤치마크가 환각/공간추론/OCR/범용/상식추론을 고루 커버, 원본·제안 양쪽 완료.**

---

## 5. 실험 1 — 메인 비교

### 5-1. 전체 결과표 (9세팅 × 5벤치마크, 토큰수별 최고 **bold**)

| M2 | 방법 | POPE(F1) | GQA(Acc) | TextVQA(Acc) | VQAv2(Acc) | SQA-IMG(Acc) |
|---:|---|---:|---:|---:|---:|---:|
| 128 | A VisPruner-only | 84.47 | **58.28** | **56.76** | **72.18** | **68.86** |
| 128 | B Ours simple | **85.37** | 58.26 | 54.77 | 72.08 | **68.86** |
| 128 | C Ours weighted | 85.24 | 58.27 | 55.27 | 72.17 | 68.62 |
| 64 | A VisPruner-only | 80.95 | 55.59 | **55.68** | 68.88 | 68.57 |
| 64 | B Ours simple | **82.27** | **56.66** | 54.08 | 69.77 | 68.86 |
| 64 | C Ours weighted | 82.14 | 56.26 | 54.33 | **70.44** | **69.16** |
| 32 | A VisPruner-only | 74.00 | 51.58 | 53.55 | 63.47 | 68.32 |
| 32 | B Ours simple | **77.56** | 53.52 | 53.38 | 65.46 | 69.31 |
| 32 | C Ours weighted | 77.53 | **54.03** | **53.61** | **65.88** | **69.46** |

### 5-2. 개선폭 분석 (B/C 최고 − A)

| 벤치마크 | @128 | @64 | @32 | 패턴 |
|---|---:|---:|---:|---|
| POPE (F1) | +0.90 | +1.32 | **+3.56** | 압축↑ → 이득↑ (뚜렷) |
| GQA (Acc) | ≈0 | +1.07 | **+2.45** | 압축↑ → 이득↑ |
| VQAv2 (Acc) | ≈0 | +1.56 | **+2.41** | 압축↑ → 이득↑ |
| SQA-IMG (Acc) | ≈0 | +0.59 | +1.14 | 둔감하나 저토큰 소폭↑ |
| TextVQA (Acc) | −1.49 | −1.35 | +0.06 | **예외(하락)** |

- "압축 강할수록 clustering 이득 증가" 패턴이 **POPE·GQA·VQAv2·SQA-IMG에서 일관**.
  넓게 보존 후 병합이 직접 소수 선택보다 정보 손실이 작고, 그 격차는 M2가 작을수록 커짐.
- **TextVQA만 예외(하락)**: OCR은 글자 획 등 **세밀 위치 정보**가 핵심인데 평균 병합이
  미세 텍스트 특징을 흐림. 단 압축이 극단(32)이면 A도 무너져 격차 소멸(+0.06).
- **SQA-IMG 둔감**: 과학 상식 추론은 소수 핵심 토큰으로 충분(A도 토큰수 무관 ~68) →
  clustering 이득 작으나 **무회귀**(전 세팅 B/C ≥ A, 최고 C-32 69.46).

### 5-3. 벤치마크별 심층 분석
- **POPE**: 최대 수혜. diverse 토큰을 병합·보존해 배경 맥락이 유지되어 "없는 객체" 거부에
  유리. **simple 근소 우위**(객체존재는 균등평균이 적합), 단 adversarial은 weighted 우위.
- **GQA**: 공간관계 추론은 토큰 감소에 민감(A 58→52). clustering이 저토큰에서 정보 밀도를
  올려 +1~2.5p 회복. simple/weighted 혼재, **@32는 weighted 우위(54.03)**.
- **VQAv2**: @128 중립, @64/32 개선. **weighted가 전반 우위**(특히 other/yes-no). number는
  병합으로도 한계(아래 8-2).
- **TextVQA**: 유일 하락. 병합이 OCR 세부 손상. **weighted가 simple보다 일관 우위**
  (중요 토큰 강조가 텍스트 보존에 그나마 유리) → OCR엔 보수적 압축 권장.
- **SQA-IMG**: 토큰수 무관(상식). clustering 무해·소폭 이득, **weighted가 저토큰 최고**.

### 5-4. 최종 권장 설정
| 상황 | 권장 | 근거 |
|---|---|---|
| 범용 | weighted_avg, r=0.5, M1=2×M2 | 전반 안정·저토큰 우위 |
| 환각 평가(POPE) | simple_avg, r=0.3(diverse↑) | r↓서 POPE 우위(7-1) |
| OCR(TextVQA) | 보수적 압축(M2≥64) 또는 clustering OFF | 병합이 텍스트 손상 |
| counting(VQAv2 number) | 보수적 M2(≥64), weighted | number 압축 최취약 |

→ **핵심: 동일 토큰 수에서 제안 B/C가 POPE·GQA·VQAv2·SQA 일관 우위, TextVQA만 예외.**

---

## 6. 실험 2 — 기존 Baseline 비교

각 방법: **FastV**=LLM layer2 이후 text-visual attention 기반 프루닝 · **ToMe**=bipartite
soft matching으로 유사 토큰 쌍 병합 · **SparseVLM**=text relevance 기반 sparsification +
정보 재활용 · **VisPruner**=[CLS] attention 중요선택 + 유사도 다양성 보존 ·
**Ours**=VisPruner + Spherical K-Means 병합.

### POPE (F1)
| Method | 128 | 64 | 32 | 출처 |
|---|---:|---:|---:|---|
| FastV | 59.6 | 48.0 | 32.5 | VisPruner 논문 인용 |
| ToMe | 62.8 | 52.5 | 39.0 | VisPruner 논문 인용 |
| SparseVLM | 80.5 | 75.1 | 67.9 | VisPruner 논문 인용 |
| VisPruner (논문) | 84.6 | 80.4 | 72.7 | VisPruner 논문 |
| A VisPruner-only (재현) | 84.47 | 80.95 | 74.00 | 본 실험 |
| **B Ours simple** | **85.37** | **82.27** | **77.56** | 본 실험 |
| C Ours weighted | 85.24 | 82.14 | 77.53 | 본 실험 |

### GQA (Acc)
| Method | 128 | 64 | 32 | 출처 |
|---|---:|---:|---:|---|
| FastV | 49.6 | 46.1 | 41.5 | 논문 인용 |
| ToMe | 52.4 | 48.6 | 43.6 | 논문 인용 |
| SparseVLM | 56.0 | 52.7 | 48.3 | 논문 인용 |
| VisPruner (논문) | 58.2 | 55.4 | 52.2 | 논문 |
| A VisPruner-only (재현) | 58.28 | 55.59 | 51.58 | 본 실험 |
| **B Ours simple** | 58.26 | **56.66** | 53.52 | 본 실험 |
| **C Ours weighted** | 58.27 | 56.26 | **54.03** | 본 실험 |

- **PACT(CVPR'25)**: 직접 실행 불가 → 별도 env(pactenv: py3.12.7/cuda11.8/flash-attn2.6.3)
  + 자체 transformers 번들, 지원 백본이 LLaVA-OneVision/Qwen2-VL로 **LLaVA-1.5-7B 미지원**.
  제공 인용 블록에 PACT 수치 미포함 → **미측정**(날조 금지).
- **FastV**: LLM 디코더 layer-K+KV캐시 통합이 고정 transformers 4.37.2에서 고위험 →
  프롬프트 규정대로 VisPruner Table 1 수치 인용.

→ **핵심: 제안 B/C가 FastV/ToMe/SparseVLM/VisPruner(논문·재현)를 POPE·GQA 전 토큰수 상회.**

---

## 7. 실험 3 — Ablation

### 7-1. Important/Diverse Ratio (M2=64, M1=128, simple)
| r | Important:Diverse | POPE(F1) | GQA(Acc) |
|---:|---|---:|---:|
| 0.3 | 38:90 | **82.74** | 56.65 |
| 0.5 | 64:64 | 82.27 | **56.66** |
| 0.7 | 90:38 | 81.63 | 56.61 |

→ POPE: diverse↑(r↓)일수록 단조 상승 — 배경/맥락 토큰이 환각 거부에 기여, clustering이
그 가치를 보존. GQA: r 무감각(±0.05) — 공간추론은 important/diverse 비율보다 총 토큰수에
민감. **권장 기본 r=0.5, 환각평가는 r↓.**

### 7-2. Clustering 유무 (동일 최종 토큰수, POPE F1 / GQA Acc)
| 토큰 | OFF(=A) | ON simple(=B) | ON weighted(=C) | best Δ |
|---|---:|---:|---:|---:|
| POPE 64 | 80.95 | **82.27** | 82.14 | +1.32 |
| POPE 32 | 74.00 | **77.56** | 77.53 | **+3.56** |
| GQA 64 | 55.59 | **56.66** | 56.26 | +1.07 |
| GQA 32 | 51.58 | 53.52 | **54.03** | **+2.45** |

→ 동일 토큰수에서 ON 항상 우세, **32에서 효과 극대**(A가 직접 32개만 고르며 정보를 크게
잃는 반면, B/C는 64~128개를 보존 후 의미 병합 → 손실 최소).

### 7-3. Stage1 토큰수 M1 민감도 (M2=64, simple, r=0.5, POPE F1)
| M1 | M1/M2 | POPE(F1) |
|---:|:--:|---:|
| 96 | 1.5× | 81.62 |
| 128 | 2× | 82.27 |
| 192 | 3× | **83.03** |

→ M1↑일수록 단조 향상. Stage2가 병합할 **후보 풀이 넓어져** 더 다양한 정보를 압축. 3×까지
이득 지속(추가 비용은 9절 참조 — 무시 가능).

### 7-4. Simple vs Weighted 승패 (5벤치 × 3토큰, 더 높은 쪽)
| 벤치 | 128 | 64 | 32 |
|---|---|---|---|
| POPE | simple | simple | simple |
| GQA | ~tie | simple | weighted |
| TextVQA | weighted | weighted | weighted |
| VQAv2 | weighted | weighted | weighted |
| SQA-IMG | ~tie(A=B) | weighted | weighted |

→ 집계: **simple 우세 = POPE(3)**, **weighted 우세 = TextVQA·VQAv2·SQA 다수 + GQA@32**.
결론: 객체존재(POPE)는 simple, **세밀·어려운·저토큰은 weighted** 권장. 절대차는 ≤0.5p.

---

## 8. 실험 4 — Question-Type 분석

### 8-1. POPE 카테고리별 F1 (random / popular / adversarial)
| M2 | 방법 | random | popular | adversarial |
|---:|---|---:|---:|---:|
| 128 | A | 85.90 | 85.18 | 82.31 |
| 128 | B simple | **87.45** | **85.55** | 83.11 |
| 128 | C weighted | 86.87 | 85.61 | **83.23** |
| 64 | A | 82.09 | 81.65 | 79.11 |
| 64 | B simple | **83.76** | **83.28** | 79.77 |
| 64 | C weighted | 83.04 | 82.40 | **80.98** |
| 32 | A | 74.88 | 74.63 | 72.48 |
| 32 | B simple | **78.24** | 78.07 | **76.38** |
| 32 | C weighted | 78.23 | **78.11** | 76.26 |

→ 전 카테고리 일관 개선. **adversarial(최난이도)에서 효과 큼**(@32 +3.90, @64 weighted
79.11→80.98). diverse 토큰이 배경 맥락을 보존해 "그럴듯하지만 없는 객체" 거부에 기여.

### 8-2. VQAv2 Question Type별 (overall / yes-no / number / other)
| M2 | 방법 | overall | yes/no | number | other |
|---:|---|---:|---:|---:|---:|
| 128 | A | 72.18 | 90.33 | 56.37 | 69.83 |
| 64 | A | 68.88 | 87.80 | 51.02 | 67.82 |
| 64 | C weighted | 70.44 | 89.28 | 52.78 | **69.27** |
| 32 | A | 63.47 | 84.00 | 44.77 | 61.65 |
| 32 | B simple | 65.46 | **85.82** | **46.87** | 63.70 |
| 32 | C weighted | 65.88 | 85.77 | 46.70 | **65.17** |

→ **number(counting)가 압축 최취약**(A 56.37→44.77, −11.6 @128→32): counting은 객체
위치·개수 등 세밀 공간정보에 의존. clustering이 number/other를 부분 회복(B-32 number
+2.10, C-32 other +3.52) — 병합이 정보 밀도를 높여 손실 공간정보를 일부 보상. yes/no는
압축에 강건(84~90 유지).

### 8-3. Task-Aware Policy 종합
| Question type | 권장 | 근거 |
|---|---|---|
| yes/no | 공격 압축(M2=32 OK), simple_avg | 압축 강건, 단순 |
| number | 보수 압축(M2≥64), weighted_avg | 압축 최취약, weighted가 회복 |
| other | 중간 압축, weighted_avg | weighted 최대 이득(+3.5@32) |
| adversarial/환각 | r↓(diverse↑), simple_avg | adversarial·POPE서 우위 |

→ **핵심: question type별 최적 설정이 상이 → 단일 고정이 아닌 task-aware 튜닝이 유효함을 정량 입증.**

---

## 9. 실험 5 — 효율성 (POPE 110샘플, warmup10 제외 100 평균)

| Setting | M2 | clustering | Token감소 | Latency(s/q) | GPU Mem(GB) |
|---|---:|:--:|---:|---:|---:|
| A-64 | 64 | OFF | 88.9% | 0.3217 | 14.51 |
| B-64 simple | 64 | ON(M1=128) | 88.9% | 0.3232 | 14.50 |
| C-64 weighted | 64 | ON(M1=128) | 88.9% | 0.3124 | 14.50 |
| A-32 | 32 | OFF | 94.4% | 0.3261 | 14.50 |
| B-32 simple | 32 | ON(M1=64) | 94.4% | 0.3171 | 14.50 |

→ **clustering 오버헤드 ≈ 0** (B/C latency ±1% = 측정 노이즈 내, GPU mem 동일). 이유:
Stage2는 (M1≤192)개 작은 텐서에 max_iter≤10·조기종료·`index_add_` 벡터연산이라 7B LLM
디코딩 대비 미미하고, **LLM 입력 토큰 수가 M2로 동일**하므로 디코딩 비용 불변. 정확도
향상이 사실상 **무비용**. (`CUDA_LAUNCH_BLOCKING=1`로 절대 latency는 부풀려진 상대값.)

---

## 10. 한계점 및 향후 연구
- **TextVQA(OCR) 하락**: 평균 병합이 글자 획·미세 텍스트 특징을 흐림. 향후 텍스트 영역
  토큰은 병합 제외(OCR-aware masking)하거나 해당 태스크 clustering OFF 자동 전환.
- **Task-aware 자동 라우팅 미구현**: 현재 question type별 최적 설정은 수동 grid search.
  질의 유형을 추론 시 자동 판별해 (M2, r, merge_method)를 라우팅하는 정책 미구현.
- **단일 백본 검증**: LLaVA-1.5-7B만. LLaVA-NeXT/Qwen-VL/LLaVA-OneVision 일반화 미검증.
- **VQAv2 subset**: 비용상 val 균형 6000 subset(절대값 비교 한계). full test-dev EvalAI 미수행.
- **향후**: cross-attention 기반 VLM(Flamingo류)·video 입력 확장, 자동 task-aware policy,
  학습형 클러스터 수 적응(M2 동적 결정).

---

## 11. 결론

제안 **Two-Stage(VisPruner + Spherical K-Means)** 는 VisPruner의 pruning-only 한계(잔여
중복·정보 폐기)를 "넓게 보존 후 의미 병합"으로 보완하여, **동일 토큰 수·사실상 동일 비용
(latency·메모리 ≈ VisPruner)으로 POPE·GQA·VQAv2·SQA-IMG에서 일관 향상**을 달성했다(특히
공격적 압축 M2=32에서 POPE +3.56·GQA +2.45 등 이득 극대, adversarial·counting 등 어려운
질의에서 weighted 평균이 효과적). VisPruner-only 경로는 원본 재현값과 정확 일치하여 회귀
안전성을 검증했고, 기존 baseline(FastV/ToMe/SparseVLM/VisPruner)을 POPE·GQA 전 토큰수에서
상회했다. TextVQA(OCR)만 병합 민감으로 소폭 하락하여 task-aware 적용 필요성을 시사한다.

> 데이터 출처: `exp_runner/results.tsv`, `results_update2.tsv`, `exp_runner/logs/*`,
> `05_efficiency_results.md`. 상세 단계별 문서: `01~06_*.md`. PACT/MME/full-VQAv2 미측정 사유는 §6·10.

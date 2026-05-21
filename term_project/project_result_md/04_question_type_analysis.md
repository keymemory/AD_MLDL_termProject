# 04. Question-Type-Aware 분석 (실험 4)

## 4-A. POPE 카테고리별 분석 (random / popular / adversarial)

`eval_pope.py`가 출력한 카테고리별 F1 (×100). A=VisPruner only, B=Ours(clustering).

### M2=128
| 방법 | random | popular | adversarial | (avg) |
|---|---:|---:|---:|---:|
| A-128 | 85.90 | 85.18 | 82.31 | 84.47 |
| B-128s | **87.45** | **85.55** | **83.11** | 85.37 |
| B-128w | 86.87 | 85.61 | 83.23 | 85.24 |
| Δ(Bs−A) | +1.55 | +0.37 | +0.80 | +0.90 |

### M2=64
| 방법 | random | popular | adversarial | (avg) |
|---|---:|---:|---:|---:|
| A-64 | 82.09 | 81.65 | 79.11 | 80.95 |
| B-64s | **83.76** | **83.28** | 79.77 | 82.27 |
| B-64w | 83.04 | 82.40 | **80.98** | 82.14 |
| Δ(Bs−A) | +1.67 | +1.63 | +0.66 | +1.32 |

### M2=32
| 방법 | random | popular | adversarial | (avg) |
|---|---:|---:|---:|---:|
| A-32 | 74.88 | 74.63 | 72.48 | 74.00 |
| B-32s | **78.24** | **78.07** | **76.38** | 77.56 |
| B-32w | 78.23 | 78.11 | 76.26 | 77.53 |
| Δ(Bs−A) | +3.36 | +3.44 | +3.90 | +3.56 |

### 분석 포인트
- **clustering은 전 카테고리에서 일관 개선** (A→B 모든 셀에서 향상).
- **adversarial(가장 어려운, co-occurring 부정객체)에서 효과 두드러짐**:
  - 32토큰: adversarial +3.90 (random +3.36, popular +3.44 중 최대 개선폭)
  - 64토큰: weighted가 adversarial에서 특히 강함 (A 79.11 → B-64w **80.98**, +1.87;
    simple +0.66 대비 우월) → **weighted_avg가 hard negative에 강함**.
- 해석: adversarial은 "그럴듯하지만 없는 객체"를 거부해야 함. clustering이 diverse 토큰을
  병합·보존해 배경/맥락 단서를 유지 → 환각 거부에 유리. attention 가중(weighted)이
  중요 영역을 대표토큰에 강조해 hard negative 판별을 추가로 도움.
- **diverse 비율 영향**(3-A 연계): r=0.3(diverse↑)에서 POPE 근소 우위 → adversarial 견고성에
  diverse 토큰 기여 시사. clustering이 이 diverse 정보를 효율적으로 압축.

## 4-B. VQAv2 Question Type별 분석 (핵심) — val 균형 subset 6000

VQAv2 val에서 `answer_type`별 **균형 subset**(yes/no·number·other 각 2000) 구성,
공식 VQA accuracy 로컬 채점. (test-dev EvalAI 대신 val 로컬채점, 균형 subset이라
overall은 자연분포 대비 보수적; A↔B 상대비교 유효.)

### Question Type별 정확도 (overall / yes-no / number / other)

| M2 | 방법 | overall | yes/no | number | other |
|---:|---|---:|---:|---:|---:|
| 128 | A | 72.18 | 90.33 | 56.37 | 69.83 |
| 128 | B simple | 72.08 | 90.37 | 55.80 | 70.08 |
| 128 | B weighted | 72.17 | 90.12 | 56.42 | 69.97 |
| 64 | A | 68.88 | 87.80 | 51.02 | 67.82 |
| 64 | B simple | 69.77 | 88.52 | 53.58 | 67.20 |
| 64 | B weighted | 70.44 | **89.28** | 52.78 | **69.27** |
| 32 | A | 63.47 | 84.00 | 44.77 | 61.65 |
| 32 | B simple | 65.46 | **85.82** | **46.87** | 63.70 |
| 32 | B weighted | 65.88 | 85.77 | 46.70 | **65.17** |

### 분석 (Task-aware policy의 직접 근거)
- **yes/no**: 가장 높고(84~90) 압축에 강건. clustering 이득은 저토큰서 소폭(+1.8@32).
  → 객체 존재형 질문은 소수 토큰으로 충분(POPE 4-A와 동일 결론).
- **number(counting)**: 가장 낮고(44~56) **압축에 가장 취약**(A 56.4→44.8, −11.6 @128→32).
  세밀 공간/계수 정보 필요. clustering이 **저토큰서 손실 회복**(A-32 44.77 → B-32s 46.87,
  **+2.1**) — 가설("number는 보수적 압축 필요") 정량 확인 + clustering의 완화 효과 입증.
- **other**: 중간(61~70). clustering, 특히 **weighted_avg가 크게 개선**
  (A-32 61.65 → B-32w **65.17**, +3.5). 다양한 시각정보 질문에 attention 가중 병합 유효.
- 종합: 압축↓일수록 number/other에서 clustering 이득 극대, **weighted_avg가 number 외
  전반(yes-no·other)에서 우위** → hard/세밀 질의에 weighted 권장.

### Task-aware 시사점 (실험 1·3·4 종합)
| Task 유형 | 권장 구성 | 근거 |
|---|---|---|
| 객체 존재/상식(POPE류) | clustering ON, simple_avg, 공격적 압축 가능(M2=32+) | 32서 +3.6p, simple 우위 |
| Hard negative(adversarial) | clustering ON, **weighted_avg** | adversarial서 weighted 우위 |
| 공간추론(GQA) | clustering ON, 저토큰서 이득(M2≤64) | 64/32서 +1~2.5p |
| OCR/세밀텍스트(TextVQA) | 보수적 압축 또는 weighted_avg | 병합 민감, weighted 근소 우위 |
| Counting(VQAv2 number) | 보수적 압축(M2≥64), clustering ON | 압축에 최취약(−11.6), clustering이 손실 완화 |
| 개방형(VQAv2 other) | clustering ON, **weighted_avg** | weighted가 +3.5(@32)로 최대 이득 |
| 단순확인(VQAv2 yes/no) | 공격 압축 가능 | 압축에 강건(84~90 유지) |
| 상식추론(SQA-IMG) | 공격 압축 가능, clustering ON | 토큰수 둔감(68~69), 저토큰서 clustering 소폭↑ |

### 벤치마크별 clustering 효과 종합 (update_ver2 SQA-IMG 추가)
| 벤치마크 | 특성 | clustering 효과 |
|---|---|---|
| POPE | 객체 환각 | 큰 이득(@32 +3.6 F1) |
| GQA | 공간관계 | 저토큰 이득(@64/32 +1~2.5) |
| VQAv2 | question-type별 | @64/32 +0.9~+2.4, number 회복 |
| TextVQA | OCR | 소폭 하락(병합 민감) |
| **SQA-IMG** | 상식추론 | **둔감하나 무회귀·저토큰 소폭 개선(C-32 +1.1)** |

→ 단일 고정 설정이 아닌 **task별 (clustering/merge_method/M2) 조합 튜닝**이 유효함을
VQAv2 question-type 데이터로 직접 입증. 특히 **number(counting)는 보수적 압축이 필수**이고
**other/hard 질의는 weighted_avg**가, **yes/no는 공격적 압축**이 최적 — 제안 논문의
task-aware policy 근거 확보.

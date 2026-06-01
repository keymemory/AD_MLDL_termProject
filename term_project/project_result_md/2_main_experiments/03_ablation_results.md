# 03. Ablation Study (실험 3)

LLaVA-1.5-7B, greedy, 안정모드. 기준: clustering ON, simple_avg, M1/M2 명시.

## 3-A. Important/Diverse Ratio (r) — M2=64, M1=128, simple, POPE+GQA

| ID | r | Important:Diverse | POPE(F1) | GQA(Acc) |
|---|---:|---|---:|---:|
| R-30 | 0.3 | 38 : 90 | **82.74** | 56.65 |
| R-50 (=B-64s) | 0.5 | 64 : 64 | 82.27 | **56.66** |
| R-70 | 0.7 | 90 : 38 | 81.63 | 56.61 |

분석(확정): **POPE는 diverse 비중↑(r↓)일수록 단조 향상** — r=0.3(82.74) > 0.5(82.27) >
0.7(81.63). diverse 토큰(배경/맥락)이 환각 거부에 기여, clustering이 그 가치를 잘 보존.
**GQA는 r에 거의 무감각**(56.61~56.66, ±0.05) — 공간추론은 important/diverse 비율보다
총 토큰수에 더 민감. 결론: **기본 r=0.5 무난**, POPE류 환각평가는 r↓(diverse↑)가 유리.

> ※ 버그: r=0.7에서 VisPruner 원본 diverse-선택 루프가 홀수 R일 때 `R//2-r` vs `ceil(R/2)-r`
> 인덱싱 불일치로 IndexError. `llava_arch.py`에서 arange 확장 길이를 `distinct_indices`
> 실제 길이에 맞추도록 수정(짝수 R 동작 불변). 검증: 재실행 시 재시도/에러 0.

## 3-B. Clustering 유무 직접 비교 (동일 최종 토큰수, POPE+GQA)

실험 1 결과 재사용 (C-off=A, C-on=B).

### 64토큰
| ID | 구성 | POPE(F1) | GQA(Acc) |
|---|---|---:|---:|
| C-off64 (=A-64) | M1=64→M2=64, clustering OFF | 80.95 | 55.59 |
| C-on64s (=B-64s) | M1=128→M2=64, simple | **82.27 (+1.32)** | **56.66 (+1.07)** |
| C-on64w (=B-64w) | M1=128→M2=64, weighted | 82.14 (+1.19) | 56.26 (+0.67) |

### 32토큰
| ID | 구성 | POPE(F1) | GQA(Acc) |
|---|---|---:|---:|
| C-off32 (=A-32) | M1=32→M2=32, OFF | 74.00 | 51.58 |
| C-on32s (=B-32s) | M1=64→M2=32, simple | **77.56 (+3.56)** | 53.52 (+1.94) |
| C-on32w (=B-32w) | M1=64→M2=32, weighted | 77.53 (+3.53) | **54.03 (+2.45)** |

→ **동일 최종 토큰수에서 clustering ON이 항상 우세**. 효과는 32토큰에서 가장 큼
(POPE +3.5p, GQA +2~2.5p). "더 많이 보존 후 의미 병합 > 직접 소수 선택" 입증.

## 3-C. Stage1 토큰수 M1 민감도 — M2=64, simple, r=0.5, POPE

| ID | M1 | M1/M2 | POPE(F1) |
|---|---:|:---:|---:|
| M-96 | 96 | 1.5× | 81.62 |
| M-128 (=B-64s) | 128 | 2× | 82.27 |
| M-192 | 192 | 3× | **83.03** |

→ **M1이 클수록 단조 향상** (81.62 → 82.27 → 83.03). Stage1에서 더 많은 후보를 남길수록
Stage2 클러스터링이 정보를 더 잘 압축. 단, M1↑는 Stage2 연산량↑ (효율성은 `05` 참조).
M1/M2 = 3× 까지 이득 지속 확인.

## 3-D. Simple Average vs Weighted Average (실험 1 B-시리즈 재분석)

| 벤치 | M2 | simple | weighted | 우세 |
|---|---:|---:|---:|---|
| POPE | 128 | **85.37** | 85.24 | simple +0.13 |
| POPE | 64 | **82.27** | 82.14 | simple +0.13 |
| POPE | 32 | **77.56** | 77.53 | ≈ (simple +0.03) |
| GQA | 128 | 58.26 | **58.27** | ≈ |
| GQA | 64 | **56.66** | 56.26 | simple +0.40 |
| GQA | 32 | 53.52 | **54.03** | weighted +0.51 |
| TextVQA | 128 | 54.77 | **55.27** | weighted +0.50 |
| TextVQA | 64 | 54.08 | **54.33** | weighted +0.25 |
| TextVQA | 32 | 53.38 | **53.61** | weighted +0.23 |

분석: 차이는 전반적으로 작음(≤0.5p). **POPE는 simple, TextVQA·GQA(저토큰)는 weighted**가
근소 우위. weighted_avg는 [CLS] attention 가중으로 중요 토큰 정보를 대표토큰에 더 반영 →
세밀 정보 필요한 TextVQA에서 유리. POPE(객체 존재)는 단순 평균으로 충분.
실용 권장: **기본 simple_avg**, TextVQA류 task는 weighted_avg 고려.

## 요약
- clustering은 동일 토큰수에서 항상 이득, 압축 강할수록(32) 효과 극대 (3-B).
- M1↑ → 성능↑ 단조, 3× 까지 이득 (3-C).
- POPE는 diverse↑(r↓) 단조 유리, GQA는 r 무감각 (3-A).
- simple/weighted 차이 작음, task별 미세 선택 (3-D).

# AGG Experiment Plan and Results

## 목적

AGG(Attention Gain + Greedy diversity Gain)는 Stage-1에서 보존할 important token과 diverse token의 개수를 별도 하이퍼파라미터 없이 결정한다. 사용자가 지정하는 유일한 토큰 수는 Stage-2 최종 개수 `M2`이다.

## 방법

### Important token

1. CLS-to-patch attention을 내림차순으로 정렬한다.
2. 인접한 attention의 1차 차분을 marginal attention gain으로 정의한다.
3. gain 곡선과 양 끝점을 잇는 직선 사이 거리가 최대인 elbow까지 important token으로 선택한다.

### Diverse token

1. important token을 초기 선택 집합으로 둔다.
2. 남은 token 중 현재 집합과의 최대 cosine similarity가 가장 낮은 token을 greedy하게 추가한다.
3. `1 - max cosine similarity`를 marginal diversity gain으로 기록한다.
4. diversity gain 곡선의 elbow까지 diverse token으로 선택한다.

`M1 = n_important + n_diverse`이며, `M1 < M2`인 경우 최종 병합을 위해 top-attention token으로만 부족분을 채운다. 이후 spherical k-means로 `M2`개까지 병합한다.

## 실험 설정

- Model: LLaVA-1.5-7B
- Datasets: POPE, GQA, TextVQA, ScienceQA-IMG
- `M2`: 32, 64, 128
- Merge: simple average, attention-weighted average
- Baseline: `M1 = 2 * M2`, important:diverse = 50:50
- 실행 환경: `vispruner`, GPU 1, nohup

## AGG 결과

| Dataset | M2 | Simple | Weighted | Best |
|---|---:|---:|---:|---:|
| POPE AvgF1 | 32 | 0.8047 | **0.8115** | 0.8115 |
| POPE AvgF1 | 64 | 0.8467 | **0.8485** | 0.8485 |
| POPE AvgF1 | 128 | 0.8576 | **0.8583** | 0.8583 |
| GQA Acc | 32 | **54.56** | 54.40 | 54.56 |
| GQA Acc | 64 | 56.73 | **57.13** | 57.13 |
| GQA Acc | 128 | **58.79** | 58.71 | 58.79 |
| TextVQA Acc | 32 | 50.41 | **51.80** | 51.80 |
| TextVQA Acc | 64 | 52.81 | **53.44** | 53.44 |
| TextVQA Acc | 128 | **56.20** | 55.98 | 56.20 |
| SQA-IMG Acc | 32 | 69.32 | **69.63** | 69.63 |
| SQA-IMG Acc | 64 | **69.77** | 69.75 | 69.77 |
| SQA-IMG Acc | 128 | 69.65 | **69.75** | 69.75 |

## 평균 선택 구성

| Dataset | Important avg | Diverse avg | Important % | Diverse % |
|---|---:|---:|---:|---:|
| POPE | 6.60 | 98.57 | 6.27% | 93.73% |
| GQA | 6.67 | 100.59 | 6.22% | 93.78% |
| TextVQA | 6.86 | 105.13 | 6.13% | 93.87% |
| SQA-IMG | 6.40 | 85.07 | 6.99% | 93.01% |

## Fixed 50:50과 비교

각 설정에서 merge 방식 중 높은 점수를 비교했다.

| Dataset | M2 | AGG best | Fixed best | AGG - Fixed |
|---|---:|---:|---:|---:|
| POPE | 32 | 0.8115 | 0.7724 | **+0.0391** |
| POPE | 64 | 0.8485 | 0.8222 | **+0.0263** |
| POPE | 128 | 0.8583 | 0.8593 | -0.0010 |
| GQA | 32 | 54.56 | 53.84 | **+0.72** |
| GQA | 64 | 57.13 | 56.81 | **+0.32** |
| GQA | 128 | 58.79 | 58.48 | **+0.31** |
| TextVQA | 32 | 51.80 | 53.49 | -1.69 |
| TextVQA | 64 | 53.44 | 54.38 | -0.94 |
| TextVQA | 128 | 56.20 | 55.07 | **+1.13** |
| SQA-IMG | 32 | 69.63 | 70.36 | -0.73 |
| SQA-IMG | 64 | 69.77 | 69.87 | -0.10 |
| SQA-IMG | 128 | 69.75 | 70.08 | -0.33 |

전체 행 단위 결과는 `exp2_results.tsv`에 수록한다.

## POPE Weighted Latency

AGG weighted merge 조건만 대상으로 `M2=128 -> 64 -> 32`를 nohup worker 하나에서 순차 실행했다. Latency는 각 조건의 inference process 시작 직전부터 answer JSONL 생성 완료 직후까지의 wall-clock time을 생성 문항 수로 나눈 값이다. 따라서 모델 로딩, image I/O, preprocessing, AGG selection, spherical k-means, weighted merge, generation, answer write를 포함하고 POPE evaluator 시간은 제외한다.

| M2 | AvgF1 | AvgAcc | Generated | M1 mean | M1 std | M1 min | M1 max | INFER_SEC | Latency sec/q |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.8099 | 0.8288 | 8910/8910 | 105.17 | 19.21 | 55 | 158 | 2726.45 | 0.3060 |
| 64 | 0.8470 | 0.8572 | 8910/8910 | 105.20 | 19.12 | 64 | 158 | 2711.07 | 0.3043 |
| 128 | 0.8585 | 0.8662 | 8910/8910 | 129.22 | 4.25 | 128 | 158 | 2817.60 | 0.3162 |

성능은 `M2=128`이 가장 높고, latency는 `M2=64`가 가장 낮았다. `M2=32`와 `M2=64`는 AGG가 선택한 평균 `M1`이 거의 같아서 latency 차이가 작지만, `M2=32`는 성능 하락폭이 더 크다. 정리 TSV는 `pope_agg_weighted_latency_results.tsv`에 수록한다.

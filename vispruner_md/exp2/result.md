# AGG Final Result

## 결론

AGG는 attention과 feature geometry의 marginal gain을 각각 사용해 important/diverse token 수를 자동으로 결정한다. 고정 비율이나 attention mass threshold 없이 `M2`만 지정할 수 있다는 것이 핵심 장점이다.

POPE에서는 `M2=32, 64`에서 fixed 50:50보다 각각 `+0.0391`, `+0.0263` AvgF1이 높았다. GQA에서는 `M2=32, 64, 128`에서 각각 `+0.72`, `+0.32`, `+0.31` Acc가 높았다. TextVQA에서는 `M2=128`에서 `+1.13` Acc로 가장 큰 개선을 보였다.

반면 TextVQA의 작은 M2와 ScienceQA-IMG에서는 fixed 50:50이 더 안정적이었다. 따라서 AGG는 시각적 객체 및 장면 증거가 중심인 POPE/GQA에 특히 효과적이며, OCR 또는 외부 지식 비중이 큰 과제에서는 diverse gain 정의를 추가 개선할 여지가 있다.

## 최고 성능

| Dataset | M2 | Merge | Metric |
|---|---:|---|---:|
| POPE | 32 | weighted | 0.8115 AvgF1 |
| POPE | 64 | weighted | 0.8485 AvgF1 |
| POPE | 128 | weighted | 0.8583 AvgF1 |
| GQA | 32 | simple | 54.56 Acc |
| GQA | 64 | weighted | 57.13 Acc |
| GQA | 128 | simple | 58.79 Acc |
| TextVQA | 32 | weighted | 51.80 Acc |
| TextVQA | 64 | weighted | 53.44 Acc |
| TextVQA | 128 | simple | 56.20 Acc |
| SQA-IMG | 32 | weighted | 69.63 Acc |
| SQA-IMG | 64 | simple | 69.77 Acc |
| SQA-IMG | 128 | weighted | 69.75 Acc |

평균적으로 AGG가 선택한 Stage-1 token의 약 6-7%가 important, 93-94%가 diverse였다. 이는 중요한 소수 token을 attention 급락점으로 고정한 뒤, 대부분의 예산을 시각적으로 중복되지 않는 근거 보존에 사용하는 방식으로 해석할 수 있다.

## POPE Latency

POPE weighted merge 조건에서 `M2=128, 64, 32`를 같은 GPU 1번에서 순차 실행했다. Latency는 모델 로딩부터 answer JSONL 생성 완료까지 포함한 end-to-end inference 평균이며 evaluator 시간은 제외했다.

| M2 | AvgF1 | AvgAcc | M1 mean | INFER_SEC | Latency sec/q |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.8099 | 0.8288 | 105.17 | 2726.45 | 0.3060 |
| 64 | 0.8470 | 0.8572 | 105.20 | 2711.07 | 0.3043 |
| 128 | 0.8585 | 0.8662 | 129.22 | 2817.60 | 0.3162 |

비교 기준으로는 `M2=64`가 가장 균형적이다. `M2=128`은 성능이 가장 높지만 latency가 가장 길고, `M2=32`는 latency 이득 없이 성능 손실이 크다.

# POPE AGG Weighted Latency Experiment

## 요청 조건

| 항목 | 설정 |
|---|---|
| Model | LLaVA-1.5-7B + AGG (`attngain` + `greedygain`) |
| Dataset | POPE full test, 8,910 questions |
| Metrics | POPE Average F1, Accuracy |
| Cluster merge | `weighted_avg` |
| M2 | 128 -> 64 -> 32 순차 실행 |
| Environment | conda `vispruner` |
| GPU | physical GPU 1, NVIDIA GeForce RTX 4090 24GB |
| Inference | batch size 1, fp16, temperature 0, `vicuna_v1` |
| Stability | `CUDA_LAUNCH_BLOCKING=1` |
| Execution | one nohup worker, three independent model processes |

## Latency 정의

각 M2 조건에서 모델 inference process를 시작하기 직전부터 answer JSONL 생성이 끝난 직후까지의 wall-clock 시간을 `INFER_SEC`로 기록한다.

```text
LATENCY_SEC_PER_Q = INFER_SEC / generated_question_count
```

포함 범위:

- 조건마다 1회 수행되는 model/vision tower loading
- image file I/O와 preprocessing
- token selection, spherical k-means, weighted merge
- autoregressive answer generation과 answer JSONL write

제외 범위:

- POPE evaluator 실행 시간
- job 사이 shell 대기 시간

따라서 이 값은 kernel-only latency가 아니라 **full-dataset end-to-end inference latency의 문항당 평균**이다. 이후 비교 모델도 동일한 checkpoint, POPE question/image, batch size 1, fp16, temperature 0, 단독 GPU, 전체 8,910문항, process-per-condition 방식으로 측정해야 한다.

Accuracy는 POPE의 `random`, `popular`, `adversarial` category별 Accuracy를 산술 평균한 `AvgAcc`로 기록한다. F1 역시 evaluator가 출력하는 세 category F1의 산술 평균 `AvgF1`이다.

모델 로딩 시간은 8,910문항에 걸쳐 amortize되지만 포함된다. 실행 재개 시 전체 조건 latency가 아니므로 worker는 `LATENCY_SEC_PER_Q=RESUMED`로 기록한다.

## 실행 파일

- Jobs: `term_project/exp_runner/jobs/exp2_agg_weighted_pope_latency_jobs.tsv`
- Worker: `term_project/exp_runner/workers/worker_exp2.sh`
- Runtime output: `reproduced_results/pope_agg_weighted_latency/`
- Answers: `term_project/exp_runner/exp2_answers/pope/`
- Evaluator logs: `term_project/exp_runner/exp2_logs/`
- Selection stats: `reproduced_results/pope_agg_weighted_latency/attn_stats/`

## 실행 상태

- 시작: 2026-07-08 22:22 KST
- GPU: physical GPU 1
- compute PID: `140156`
- 시작 순서: `AGG-LAT-128w` -> `AGG-LAT-64w` -> `AGG-LAT-32w`
- 현재 상태: 완료
- nohup log: `reproduced_results/pope_agg_weighted_latency/nohup.log`
- worker log: `/tmp/worker_exp2_g1.log`

## 결과

모든 조건에서 8,910/8,910개 answer가 생성되었다.

| M2 | ID | Merge | AvgF1 | AvgAcc | Generated | M1 mean | M1 std | M1 min | M1 max | INFER_SEC | Latency sec/q |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | AGG-LAT-32w | weighted_avg | 0.8099 | 0.8288 | 8910/8910 | 105.17 | 19.21 | 55 | 158 | 2726.45 | 0.3060 |
| 64 | AGG-LAT-64w | weighted_avg | 0.8470 | 0.8572 | 8910/8910 | 105.20 | 19.12 | 64 | 158 | 2711.07 | 0.3043 |
| 128 | AGG-LAT-128w | weighted_avg | 0.8585 | 0.8662 | 8910/8910 | 129.22 | 4.25 | 128 | 158 | 2817.60 | 0.3162 |

## 해석

- 성능은 `M2=128`이 가장 높았다: AvgF1 0.8585, AvgAcc 0.8662.
- `M2=64`는 `M2=128` 대비 AvgF1은 -0.0116 낮지만 latency는 0.3043 sec/q로 가장 낮았다.
- `M2=32`는 latency가 `M2=64`와 거의 동일하지만 AvgF1이 0.8099로 더 크게 하락했다.
- AGG의 자동 `M1`은 `M2=32`와 `M2=64`에서 거의 같은 평균 약 105개를 선택했다. 그래서 두 조건의 latency 차이가 작게 나온 것으로 보인다.
- 이번 latency는 모델 로딩을 포함한 full-dataset end-to-end 평균이다. 비교 모델과 맞출 때도 동일하게 전체 8,910문항, batch size 1, fp16, temperature 0, 조건별 독립 process 기준으로 재야 한다.

정리 TSV는 `vispruner_md/exp2/pope_agg_weighted_latency_results.tsv`에 저장되어 있다. 실행 직후의 local runtime TSV는 `reproduced_results/pope_agg_weighted_latency/exp2_results.tsv`에 있다.

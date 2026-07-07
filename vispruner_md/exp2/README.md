# Exp2: Adaptive Visual Token Selection

이 디렉터리는 Stage-1의 important/diverse token 수를 정하는 방법론 비교와 최종 결과를 담는다.

## 파일 구성

- `result.md`: 핵심 결과와 최종 결론
- `exp2_plan.md`: 실험 설계, 진행 기록, 전체 결과표
- `exp2_results.tsv`: 실험별 집계 결과
- `../../term_project/exp_runner/jobs/exp2_attngain_greedygain_*.tsv`: AGG 재현용 실험 조합
- `../../term_project/exp_runner/workers/worker_exp2.sh`: 추론 및 평가 worker

샘플별 답변, 로그, attention 통계 및 생성 이미지는 용량이 크고 재생성 가능하므로 Git에서 제외한다.

## 실행

`vispruner` 환경에서 GPU 번호와 job 파일을 지정한다. 모델 및 데이터 경로는 worker 상단에 정의된 환경변수로 덮어쓸 수 있다.

```bash
cd term_project
nohup conda run -n vispruner bash exp_runner/workers/worker_exp2.sh 1 \
  exp_runner/jobs/exp2_attngain_greedygain_pope_gqa_jobs.tsv \
  > exp_runner/exp2_worker.nohup.log 2>&1 &
```

다른 환경에서는 `MODEL`, `POPE_QF`, `POPE_IMG`, `POPE_ANN`, `GQA_QF`, `GQA_IMG`, `GQA_EVAL_DIR`, `GQA_Q_PATH`, `TEXTVQA_QF`, `TEXTVQA_IMG`, `TEXTVQA_ANN`, `SQA_QF`, `SQA_IMG`, `SQA_BASE`를 지정한다. AGG는 `--select_mode attngain --diverse_mode greedygain` 조합으로 활성화된다.

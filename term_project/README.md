# term_project — Two-Stage Visual Token Reduction (구현 루트)

본 디렉토리는 과제 제안 방법 **Two-Stage(VisPruner + Spherical K-Means)** 의 실제 구현·실험 코드 루트입니다.
LLaVA-1.5-7B 백본 위에 Stage1(VisPruner pruning)과 Stage2(Spherical K-Means 병합)를 통합했습니다.

> 프로젝트 개요·결과 요약은 상위 [`../README.md`](../README.md) 참조.
> 본 코드는 [LLaVA](https://github.com/haotian-liu/LLaVA)와 [VisPruner](https://github.com/Theia-4869/VisPruner)를 기반으로 합니다.

## 디렉토리

```
term_project/
├── llava/                     모델 코드 (LLaVA + Two-Stage 통합)
│   └── model/
│       ├── spherical_kmeans.py            ★ Stage2: Spherical K-Means 병합 (신규)
│       ├── llava_arch.py                  M1/M2 분리 + Stage2 호출 ([Two-Stage] 주석)
│       ├── builder.py                     dtype 패치 (CUDA 무음손상 해결)
│       └── language_model/llava_llama.py  clustering 설정 getter
├── exp_runner/                실험 실행 인프라
│   ├── workers/               worker_*.sh    GPU별 추론·채점 워커
│   ├── launchers/             launch_*.sh    다중 GPU 런처
│   ├── jobs/                  exp_jobs_*.tsv  실험 작업 정의
│   ├── results/              results*.tsv   실험 결과
│   ├── efficiency.py          실험5 효율성(지연·메모리) 측정
│   ├── vqa_eval.py            VQAv2 로컬 채점 (공식 정규화)
│   └── setup_*.py             ScienceQA/TextVQA 데이터 준비
├── scripts/
│   ├── convert_gqa_for_eval.py    GQA 채점 변환 (워커가 사용)
│   └── legacy_llava/              LLaVA 원본 학습·평가 스크립트 (본 과제 미사용·보관)
├── project_result_md/         결과 문서 → 카테고리별 정리 (final_report.md 우선)
├── sanity_check.sh            구현 sanity 3종 (POPE 300 subset)
├── EVAL.md                    벤치마크 데이터 준비 안내 (LLaVA 원본)
└── pyproject.toml             패키지 정의
```

## 핵심 구현

| 파일 | 역할 |
|---|---|
| `llava/model/spherical_kmeans.py` | Stage2 — 단위 구 투영 후 cosine 기반 k-means, simple/weighted 평균 병합 |
| `llava/model/llava_arch.py` | `encode_images`에서 Stage1(VisPruner) 선택 → Stage2 병합. `enable_clustering=False`면 원본 VisPruner와 비트동일 |
| `llava/model/language_model/llava_llama.py` | clustering 하이퍼파라미터(M1, M2, method, max_iter) getter |
| `llava/model/builder.py` | `device_map='auto'` 시 vision tower dtype 강제 캐스팅 (무음 손상 방지) |

핵심 수정부는 코드에 `[VisPruner]` / `[Two-Stage]` / `[FIX]` / `[RESUME]` 주석으로 표시.

## 실행 (모델·데이터 복원 후)

```bash
pip install -e .

# 1) 구현 sanity (VisPruner-only / two-stage simple / weighted)
bash sanity_check.sh

# 2) M1 Scaling Law 실험 (다중 GPU 병렬, 결과는 exp_runner/results/)
bash exp_runner/launchers/launch_b.sh     # POPE + GQA
bash exp_runner/launchers/launch_c.sh     # TextVQA
bash exp_runner/launchers/launch_d.sh     # ScienceQA

# 3) 효율성(지연·GPU 메모리) 측정
python exp_runner/efficiency.py
```

> 실행 스크립트의 경로(모델·데이터·conda)는 실험 서버 기준 절대경로로 작성되어 있습니다.
> 데이터 준비는 [EVAL.md](EVAL.md), 재현 절차는 `project_result_md/2_main_experiments/06_experiment_log.md` 참조.

## 결과 문서

| 폴더 | 내용 |
|---|---|
| `project_result_md/final_report.md` | 최종 종합 보고서 (먼저 읽기) |
| `project_result_md/1_implementation/` | 구현 보고서·검증·명세준수 체크리스트 |
| `project_result_md/2_main_experiments/` | 실험1~5 (baseline·ablation·질의유형·효율·실험로그) |
| `project_result_md/3_scaling_experiments/` | M1 Scaling Law (B/C/D + 교차 벤치마크 분석) |
| `../vispruner_md/` | VisPruner 원본 재현 문서 (회귀 비교용) |

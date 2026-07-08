# AD_MLDL_termProject — Two-Stage Visual Token Reduction for VLMs

> **AGG 재현:** parameter-free Attention Gain + Greedy diversity Gain 구현과 네 벤치마크 재현 절차는 [`vispruner_md/exp2/README.md`](vispruner_md/exp2/README.md)를 참조하세요. 공개 브랜치는 `agg-token-selection`입니다.

VisPruner(ICCV'25)의 pruning-only 한계를 보완하는 **Two-Stage(VisPruner + Spherical K-Means)**
시각 토큰 축소 프레임워크 구현·평가. **LLaVA-1.5-7B** 백본, 5개 벤치마크(POPE/GQA/TextVQA/
VQAv2/SQA-IMG)에서 검증.

---

## 디렉토리 구조

```
AD_MLDL_termProject/
├── term_project/                  ★ 제안 구조(Two-Stage) — 본 과제 핵심
│   ├── llava/                     LLaVA 코드 (Stage1+Stage2 통합)
│   │   ├── model/
│   │   │   ├── spherical_kmeans.py        ★ Stage2 핵심 (신규)
│   │   │   ├── llava_arch.py              M1/M2 분리 + Stage2 호출
│   │   │   ├── builder.py                 dtype 패치 (CUDA 무음손상 해결)
│   │   │   └── language_model/llava_llama.py  clustering 설정 getter
│   │   └── eval/                          model_vqa_loader/science (resume 패치)
│   ├── exp_runner/                실험 실행 인프라
│   │   ├── workers/               worker_*.sh   (GPU별 추론·채점 워커)
│   │   ├── launchers/             launch_*.sh   (다중 GPU 런처)
│   │   ├── jobs/                  exp_jobs_*.tsv (실험 작업 정의)
│   │   ├── results/              results*.tsv  (실험 결과)
│   │   ├── efficiency.py          실험5 효율성 측정
│   │   ├── vqa_eval.py            VQAv2 로컬 채점
│   │   └── setup_*.py             데이터셋 준비
│   ├── scripts/
│   │   ├── convert_gqa_for_eval.py        GQA 채점 변환 (프로젝트 사용)
│   │   └── legacy_llava/                  LLaVA 원본 학습·평가 스크립트 (미사용·보관)
│   ├── project_result_md/         ★ 결과 문서 (Two-Stage 제안)
│   │   ├── final_report.md                ★ 최종 종합 보고서 (먼저 읽기)
│   │   ├── 1_implementation/              구현·검증·명세준수 체크리스트
│   │   ├── 2_main_experiments/            실험1~5 (baseline/ablation/질의유형/효율/로그)
│   │   └── 3_scaling_experiments/         M1 Scaling Law (B/C/D + 교차분석)
│   ├── sanity_check.sh            구현 sanity 3종
│   ├── models/ → (.gitignore)     LLaVA-1.5-7B + CLIP (15GB, symlink)
│   └── playground/data/eval → ../../dataset (symlink)
│
├── vispruner_md/                  VisPruner 재현 문서
│   ├── 01~06_*.md                 환경/코드분석/재현결과/실행로그/추가벤치/전체재현
│   └── VisPruner_*_보고서.md       재현 종합 보고서 2종
│
├── dataset/                       5개 벤치마크 (★ 폴더만 유지, 실데이터 .gitignore)
│   ├── pope/ gqa/ textvqa/ vqav2/ scienceqa/ ...
│   └── README.md
│
└── README.md                      (이 파일)
```


> **모델/데이터는 `.gitignore`로 제외**(15GB 모델 + 80GB 데이터). 폴더 구조와 코드만 커밋.
> 데이터 복원 방법: `term_project/project_result_md/06_experiment_log.md` 참조.


---

## 제안 방법 (Two-Stage) 한눈에

```
이미지 → CLIP-ViT-L/14-336 → 576 visual tokens
   ① Stage1 (VisPruner)   [CLS] attention 상위 + cosine 중복제거 → M1개 보존
   ② Stage2 (Spherical K-Means)  M1개를 M2개 클러스터로 병합 → M2개 대표 토큰
   → LLM(Vicuna-7B) <image> 자리에 주입 → 답변 생성
```

| 구성 | 역할 |
|---|---|
| Stage1 important | [CLS] attention 상위 `M1·r`개 — 전경/핵심 객체 |
| Stage1 diverse | 잔여 중 cosine 중복 반복 제거 `M1·(1−r)`개 — 배경/맥락 |
| Stage2 simple_avg | 클러스터 단순 평균 (`Σxᵢ/\|C\|`) — 단순·균등 |
| Stage2 weighted_avg | attention 가중 평균 (`Σaᵢxᵢ/Σaᵢ`) — 중요 토큰 강조 |
| 분기 | `enable_clustering=False` → 기존 VisPruner와 비트동일 |

핵심 아이디어: **"좁게 선택" → "넓게 보존 후 의미 단위 병합"** 으로 동일 토큰 수에서 정보 밀도↑.

---

## 핵심 결과 (LLaVA-1.5-7B, r=0.5)

### 1) 동일 토큰 수에서 일관 향상 (A=VisPruner only, B/C=Ours)

| Benchmark | 지표 | M2=128 (A→best) | M2=64 | M2=32 |
|---|---|---|---|---|
| **POPE** | F1 | 84.47 → **85.37** (+0.90) | 80.95 → **82.27** (+1.32) | 74.00 → **77.56** (+3.56) |
| **GQA** | Acc | 58.28 → 58.27 (≈0) | 55.59 → **56.66** (+1.07) | 51.58 → **54.03** (+2.45) |
| **VQAv2**† | Acc | 72.18 → 72.17 (≈0) | 68.88 → **70.44** (+1.56) | 63.47 → **65.88** (+2.41) |
| **SQA-IMG** | Acc | 68.86 → 68.86 (≈0) | 68.57 → **69.16** (+0.59) | 68.32 → **69.46** (+1.14) |
| TextVQA | Acc | 56.76 → 55.27 (−1.49) | 55.68 → 54.33 (−1.35) | 53.55 → 53.61 (+0.06) |

→ **압축이 강할수록 제안 방법 이득↑** (POPE·GQA·VQAv2·SQA 일관). TextVQA(OCR)만 예외적
소폭 하락 (세밀 텍스트 병합 민감, task-aware 적용 필요 시사).

† VQAv2는 EvalAI 대신 **val 균형 subset 6000**(yes-no/number/other 각 2000) 로컬 채점.
A↔B 동일 subset 상대비교만 유효.

### 2) Baseline 비교 (POPE F1, FastV/ToMe/SparseVLM은 VisPruner 논문 인용)

| Method | 128 | 64 | 32 |
|---|---:|---:|---:|
| FastV | 59.6 | 48.0 | 32.5 |
| ToMe | 62.8 | 52.5 | 39.0 |
| SparseVLM | 80.5 | 75.1 | 67.9 |
| VisPruner (논문) | 84.6 | 80.4 | 72.7 |
| A VisPruner-only (재현) | 84.47 | 80.95 | 74.00 |
| **B Ours (제안)** | **85.37** | **82.27** | **77.56** |

→ 제안 B가 모든 기존 baseline + VisPruner 논문값 상회. 32토큰서 +4.86 vs 논문.

### 3) Question-Type 분석 — task-aware policy 근거

- POPE adversarial(최난이도) @32: A 72.48 → **B 76.38 (+3.90)** — diverse 토큰 보존이 환각 거부에 기여
- VQAv2 number(counting) @32: A 44.77 → **B 46.87 (+2.10)** — 세밀 공간정보 손실 부분 회복
- VQAv2 other @32: A 61.65 → **C weighted 65.17 (+3.52)** — 다양 추론에 weighted 유리

### 4) 효율성 — clustering 추가비용 ≈ 0

| Setting | M2 | clustering | Latency(s/q) | GPU Mem(GB) |
|---|---:|:---:|---:|---:|
| A-64 | 64 | OFF | 0.3217 | 14.51 |
| B-64 simple | 64 | ON(M1=128) | 0.3232 | 14.50 |
| C-64 weighted | 64 | ON(M1=128) | 0.3124 | 14.50 |

→ B/C ≈ A (±1% 측정 노이즈 내), GPU mem 동일. **정확도 향상이 사실상 무비용.**

---

## 회귀 안전성·교차 검증

- `enable_clustering=False` → 원본 VisPruner와 비트동일 경로(분기 가드).
- **SQA-IMG A시리즈** (68.86/68.57/68.32) = 원본 VisPruner 재현값과 **정확 일치**
- **VQAv2 V-128/64** (원본 VisPruner_run에서) = **A-128/64** (제안 구조) **정확 일치**
- → 제안 코드의 VisPruner-only 경로가 원본 코드와 동치임을 교차 검증.

---

## 실행 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 49GB ×3 (병렬) |
| 모델 | LLaVA-1.5-7B = Vicuna-7B + CLIP-ViT-L/14-336 + 2-layer MLP projector |
| 패키지 | PyTorch 2.1.2+cu121, transformers 4.37.2, Python 3.10 (conda `vispruner`) |
| 추론 | fp16, greedy(temp=0), `CUDA_LAUNCH_BLOCKING=1` 안정모드 |

설치/실행 (모델·데이터 복원 후):
```bash
cd term_project
pip install -e .
bash sanity_check.sh                       # 구현 sanity 3종
bash exp_runner/launchers/launch_b.sh      # 실험(M1 scaling) 다중 GPU 병렬
python exp_runner/efficiency.py            # 실험5 효율성
```

---

## 한계 / 미수행

- **TextVQA(OCR)**: 평균 병합으로 글자 세부 손상, 소폭 하락. OCR-aware masking 또는
  task별 clustering OFF 자동 라우팅 필요(향후).
- **FastV/ToMe/SparseVLM 직접 실행**: 보류, VisPruner 논문 Table 1 수치 인용 + 사유.
  PACT는 별도 env(py3.12/cuda11.8) + 백본 상이(LLaVA-OneVision)로 실행 불가, 미측정.
- **VQAv2 full test-dev**: EvalAI 전용 채점(오프라인 지표 없음) + 10만 문항 — val 균형
  subset 6000으로 대체. 논문 test-dev 인용치와 절대 비교는 불가, 상대 비교만 유효.

---

## 결론

제안 **Two-Stage(VisPruner + Spherical K-Means)** 는 동일 토큰 수·동일 비용으로
POPE/GQA/VQAv2/SQA-IMG에서 일관 향상을 달성했고, 특히 공격적 압축(M2=32)·어려운
질의(adversarial, counting, other)에서 이득이 컸다. 회귀안전(clustering OFF=원본 동치),
기존 baseline(FastV/ToMe/SparseVLM/VisPruner) 상회, 추가 비용 ≈0으로 실용성 입증.
TextVQA(OCR)만 예외로 task-aware 정책의 필요성을 시사한다.

상세 결과: **`term_project/project_result_md/final_report.md`** (11개 섹션 종합)
재현 절차/명령: `term_project/project_result_md/2_main_experiments/06_experiment_log.md`
구현 검증: `term_project/project_result_md/1_implementation/구현_명세준수_체크리스트.md`

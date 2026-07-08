# 새 대화창 인수인계 프롬프트 (복사해서 새 채팅 첫 메시지로 붙여넣기)

> 아래 ✂️ 사이를 통째로 복사해서 새 채팅 첫 메시지로 보내세요.
> 새 Claude가 즉시 현재까지의 연구/실험 상태를 파악하고 이어서 작업합니다.

---

✂️ ─────────────────────────────────────────────────────────────────────────

## 인수인계 — VLM 시각 토큰 축소 연구 (VisPruner + Two-Stage)

### 0. 작업 레포 (반드시 여기 안에서 작업)
- **`/home/jhlee/CLUST_KETI/AD_MLDL_termProject/`** (Git 연동됨)
- 새 작업 시작 전 `README.md`와 `term_project/project_result_md/final_report.md`를 먼저 읽어줘.

### 1. 지금까지 한 일 (요약)

연구 주제: **VLM(LLaVA-1.5-7B)의 시각 토큰(576) 축소 — VisPruner(ICCV'25) 개선**

VisPruner의 한계(pruning-only — 선택 후 버린 토큰 정보 폐기, 잔여 중복 미처리)를 보완하기 위해
**Two-Stage 프레임워크**를 제안·구현·평가 완료:
```
이미지 → CLIP-ViT-L/14-336 → 576 visual tokens
  ① Stage1 (VisPruner): [CLS] attention top + cosine 중복제거 → M1개 보존
  ② Stage2 (Spherical K-Means): M1 → M2(<M1) 클러스터로 의미 단위 병합
     대표토큰: simple_avg | weighted_avg(attention 가중)
  → LLM(Vicuna-7B) <image>에 M2개 주입
```
- `enable_clustering=False` 면 기존 VisPruner와 비트동일 (회귀안전 가드).
- 핵심 신규 코드: `term_project/llava/model/spherical_kmeans.py`
- 통합 지점: `term_project/llava/model/llava_arch.py::encode_images`
- CLI 인자 추가: `--enable_clustering --stage1_tokens --merge_method --kmeans_max_iter`

### 2. 실험 결과 (핵심)

LLaVA-1.5-7B / r=0.5 / greedy(temp=0) / fp16. A=VisPruner only, B=Ours simple, C=Ours weighted.

| 벤치 | 지표 | M2=128 (A→best) | M2=64 | M2=32 |
|---|---|---|---|---|
| POPE | F1 | 84.47 → **85.37** (+0.90) | 80.95 → **82.27** (+1.32) | 74.00 → **77.56** (+3.56) |
| GQA | Acc | 58.28 ≈ 58.27 | 55.59 → **56.66** (+1.07) | 51.58 → **54.03** (+2.45) |
| VQAv2† | Acc | 72.18 ≈ 72.17 | 68.88 → **70.44** (+1.56) | 63.47 → **65.88** (+2.41) |
| SQA-IMG | Acc | 68.86 ≈ 68.86 | 68.57 → **69.16** (+0.59) | 68.32 → **69.46** (+1.14) |
| TextVQA | Acc | 56.76 → 55.27 (−1.49) | 55.68 → 54.33 (−1.35) | 53.55 → 53.61 (≈0) |

→ **압축이 강할수록 제안 방법 이득↑** (POPE/GQA/VQAv2/SQA 일관). TextVQA(OCR)만 예외 하락.
→ 효율성: clustering 추가 비용 **≈0** (latency·GPU mem ≈ VisPruner-only).

† VQAv2는 EvalAI(test-dev) 대신 **val 균형 subset 6000**(yes-no/number/other 각 2000) 로컬 채점.

### 3. Baseline 처리 (반드시 알고 있어야 함)

- **POPE/GQA**: A(우리 VisPruner-only 재현)가 논문값과 정합 → FastV/ToMe/SparseVLM/VisPruner를
  논문 Table 1에서 인용해 같은 표에 비교 가능(표준 테스트셋 공유).
- **VQAv2**: ⚠️ 우리는 val subset 6000 / 논문 인용치는 test-dev 107k → **시험지가 달라 절대 비교 불가**.
  같은 subset 안에서 **A↔B 상대 비교만 유효**. (이 점은 final_report.md §6에 명시)
- **FastV/ToMe/SparseVLM**: 직접 실행 안 함 → VisPruner 논문 Table 1 인용. FastV는 LLM
  디코더 layer-K + KV cache 통합이 transformers 4.37.2에서 고위험.
- **PACT**: 실행 불가(별도 env: pactenv py3.12/cuda11.8/flash-attn, 백본 LLaVA-OneVision/Qwen2-VL로
  본 과제 LLaVA-1.5-7B 미지원). 수치 미확보(날조 금지).

### 4. 기술적 핵심 / 함정 (반드시 인지)

1. **dtype 패치 (필수)**: `term_project/llava/model/builder.py` 와 `VisPruner_run/llava/model/builder.py`
   에서 `device_map='auto'`일 때 비전 타워를 모델 dtype(fp16)으로 캐스팅. 미적용 시 비전타워(fp32)와
   모델(fp16) 혼용으로 **비동기 CUDA "illegal memory access"** 발생 → 시각 피처 무음 손상 →
   정확도 비정상 하락(POPE 0.72). 이 패치 없으면 모든 결과가 잘못 나옴.
2. **CUDA_LAUNCH_BLOCKING=1**: 안정 우선이면 켜기(2~3배 느림). dtype 패치가 있으면 끄고 빠르게 가도 됨
   (V-32 가속 사례 — 1.06 it/s → 7.90 it/s).
3. **resume**: `model_vqa_loader/science/.py`에 `answered_ids` 패치 적용됨. 크래시·중단 시 이어쓰기.
4. **VisPruner 원본 diverse 루프 버그**: 홀수 R에서 `ceil(R/2) ≠ R//2` 불일치로 IndexError. r=0.5는
   짝수 R이라 미발생, r=0.7서 노출. `term_project/llava/model/llava_arch.py`의 arange 길이를
   `distinct_indices.shape[1]`로 맞춰 수정 완료(짝수 R 동작 불변).
5. **모델/데이터 경로**: 둘 다 `.gitignore`. 모델 = `VisPruner_run/models/llava-v1.5-7b` +
   `clip-vit-large-patch14-336`. config.json의 `mm_vision_tower`는 **절대경로**로 패치되어 있음
   (디렉토리 이동 시 갱신 필요). 데이터 = `dataset/<bench>/` (폴더 구조와 `.gitkeep`만 commit).

### 5. 디렉토리 (현재 상태)
```
AD_MLDL_termProject/
├── term_project/                    ★ 제안 코드 (Two-Stage)
│   ├── llava/                       핵심 (spherical_kmeans.py 신규)
│   ├── exp_runner/                  results.tsv·results_update2.tsv·logs/·vqa_eval.py
│   ├── project_result_md/           ★ 결과 문서 (final_report.md 등 14개)
│   ├── playground/data/eval → ../dataset (symlink)
│   └── models → ../VisPruner_run/models (symlink)
├── VisPruner_run/                   원본 VisPruner 재현 환경 (회귀비교용, 동일 패치)
│   └── models/                      LLaVA-1.5-7B + CLIP (.gitignore, 15GB)
├── dataset/                         5개 벤치마크 + README.md (실데이터 .gitignore, 80GB)
├── vispruner_md/                    VisPruner 1차 재현 문서
├── reproduce.sh, run_exp*.sh        실행 스크립트
└── .gitignore                       dataset/모델/csv/이미지/answers/logs 제외
```

### 6. 결과 문서 (참고 우선순위)
- **`term_project/project_result_md/final_report.md`** ★★★ — 11개 섹션 종합 보고서(메인)
- `term_project/project_result_md/Two_Stage_실험결과_종합보고서.md` — 단독 완결 클로드웹 검토용
- `01_implementation_report.md` 구현 / `02_baseline_comparison.md` 비교 / `03_ablation_results.md` ablation
- `04_question_type_analysis.md` POPE/VQAv2 question-type / `05_efficiency_results.md` 효율성 / `06_experiment_log.md` 실행·에러 로그
- `Two_Stage_구현검증_보고서.md`, `구현_명세준수_체크리스트.md` 구현 검증
- 원시 결과: `term_project/exp_runner/results.tsv`, `results_update2.tsv`

### 7. 환경 / 재현 방법

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 49GB ×3 |
| conda env | `vispruner` (Python 3.10) |
| 핵심 패키지 | torch 2.1.2+cu121 / transformers 4.37.2 / tokenizers 0.15.1 |

```bash
cd term_project
pip install -e .                       # llava 패키지 등록
bash sanity_check.sh                   # 구현 sanity 3종 (POPE 300 subset)
bash exp_runner/launch.sh              # 실험 본 실행 (3-GPU 락기반 병렬)
python exp_runner/efficiency.py        # 효율성 측정 (POPE 110샘플)
```
모델/데이터는 `.gitignore`되어 있으니 신규 환경이면 06_experiment_log.md 참고해 다운로드.

### 8. 한계 / 미해결 (앞으로 할 만한 것)

- **TextVQA(OCR) 하락 −1.5**: 평균 병합이 글자 세부 흐림. OCR-aware masking, task별 clustering OFF
  자동 라우팅 미구현.
- **Task-aware 자동 라우팅**: question type별 최적 (M2, r, merge_method) 조합은 grid search로 파악했으나
  추론 시 자동 선택 정책 미구현.
- **단일 백본**: LLaVA-1.5-7B만. LLaVA-NeXT/Qwen-VL/LLaVA-OneVision 일반화 미검증.
- **FastV/SparseVLM/PACT 직접 실행**: 보류·인용. VQAv2 baseline을 공정 비교하려면 동일 subset 6000으로
  직접 실행 필요(현재 미수행).
- **VQAv2 full test-dev**: EvalAI 제출 미수행. val subset 6000으로 대체.
- **clustering 변형**: 현재 spherical k-means + simple/weighted avg만. soft assignment, learnable centroid,
  hierarchical merging 등 변형 미실험.

### 9. 다음 단계로 가능한 연구 방향 (선택지, 새 채팅에서 결정)
- (a) **Task-aware 자동 라우팅**: 질문 임베딩 → (M2, merge_method) 분류기 학습.
- (b) **OCR-friendly merge**: 텍스트 영역 토큰을 병합에서 제외/보존하는 마스킹.
- (c) **다른 백본 일반화**: LLaVA-NeXT 또는 Qwen2-VL에 동일 구조 적용.
- (d) **FastV/SparseVLM 직접 구현**: 공정한 baseline 비교 완성 (특히 VQAv2 subset에서).
- (e) **VQAv2 full test-dev**: EvalAI 계정 만들어 제출 (논문 인용치와 직접 비교 가능).
- (f) **Stage2 변형 ablation**: soft assignment / learnable centroid / hierarchical merging.
- (g) **PACT 환경 구축**: 별도 pactenv로 LLaVA-OneVision 백본 비교.

### 10. 처음 받은 새 Claude가 할 일
1. `README.md` + `term_project/project_result_md/final_report.md` 읽기 (필수).
2. `term_project/exp_runner/results.tsv` + `results_update2.tsv`로 실제 수치 확인 (날조 검증용).
3. 위 §4 함정(특히 dtype 패치)을 코드에서 확인 (`grep "next(model.parameters()).dtype" term_project/llava/model/builder.py`).
4. 사용자가 어떤 방향(§9의 a~g 또는 새 방향)을 원하는지 한 번 확인하고 계획 수립.
5. 새 실험은 기존 `term_project/exp_runner/` 패턴(락 기반 워커 + resume + retry) 재사용 권장.

✂️ ─────────────────────────────────────────────────────────────────────────

---

## 사용 안내

위 ✂️ 사이 블록을 통째로 복사해서 **새 Claude 채팅의 첫 메시지로 보내세요**.
새 Claude가 즉시 (a) 레포 위치, (b) 지금까지 한 일·결과, (c) 기술적 함정,
(d) 결과 문서 위치, (e) 다음 연구 후보를 파악하고 이어서 작업할 수 있습니다.

이 파일 자체는 `.gitignore`되지 않으니 commit 가능 (참고용).

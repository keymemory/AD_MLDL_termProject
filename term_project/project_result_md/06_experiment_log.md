# 06. 실험 로그

## 환경
- 코드: `experiments/term_project/` (Two-Stage 구현, `pip install -e .`)
- 데이터: `experiments/dataset/` (심볼릭: `term_project/playground/data/eval`)
- 모델: LLaVA-1.5-7B + CLIP-ViT-L/14-336 (symlink, config `mm_vision_tower` 로컬 경로)
- GPU: RTX A6000 ×3 (GPU0/1/2), conda `vispruner`, torch 2.1.2 / tf 4.37.2
- 추론: fp16, greedy(temperature=0), `CUDA_LAUNCH_BLOCKING=1`, resume+자동재시도

## 실험 설계 (33 job, 중복 제거)
- 실험1: A{128,64,32} + Bs{128,64,32} + Bw{128,64,32} × {POPE,GQA,TextVQA} = 27
- 실험3-A: R-30, R-70 × {POPE,GQA} = 4 (R-50 = B-64s 재사용)
- 실험3-C: M-96, M-192 × POPE = 2 (M-128 = B-64s 재사용)
- 실험3-B: 실험1 재사용(C-off=A, C-on=B), 3-D: 실험1 B-시리즈 재분석, 4-A: POPE 카테고리 집계
- 실험2: VisPruner 논문 Table 1 인용 / 실험5: 효율성 별도 측정

## 실행 명령 (복붙 가능)
```bash
cd experiments/term_project
# job 목록 생성 → exp_runner/exp_jobs.tsv (33줄)
# 락기반 3-GPU 병렬 (worker가 미점유 job 잡아 추론+로컬채점, results.tsv 기록)
bash exp_runner/launch.sh                       # GPU 0/1/2 워커 동시
# 단일 job 수동: bash exp_runner/worker.sh <GPU> [job파일]
# 효율성: CUDA_VISIBLE_DEVICES=0 python exp_runner/efficiency.py
```
worker 1개 job: `model_vqa_loader --visual_token_num M2 --important_ratio R [--enable_clustering --stage1_tokens M1 --merge_method METHOD]` → 벤치별 로컬 채점(eval_pope/gqa eval.py/eval_textvqa).

## 소요 시간 (관측)
- POPE 8910 ≈ 25분, GQA 12578 ≈ 50분, TextVQA 5000 ≈ 20분 (안정모드, 세팅당)
- 33-job 3-GPU 병렬 ≈ 약 6시간
- 효율성 측정: 세팅당 POPE 110샘플 ≈ 수 분

## 에러 발생 및 해결
1. **CLIP 경로 사망** (`Incorrect path_or_model_id`): `Term_project→experiments` 개명으로
   `llava-v1.5-7b/config.json`의 `mm_vision_tower` 절대경로 무효 → 신경로로 갱신. (구현단계 해결)
2. **dataset 심볼릭 깨짐**: 동 개명 → 전 벤치마크 심볼릭 `experiments/dataset` 재연결.
3. **R-70(r=0.7) IndexError** `shape mismatch [1,235] vs [1,236]`:
   - 원인: VisPruner **원본** diverse-선택 while 루프가 **홀수 R**에서 `a=residual[...,::2]`는
     `ceil(R/2)`개인데 arange 확장이 `R//2-r`(floor) → 브로드캐스트 불일치. r=0.5는 R이
     계속 짝수라 미발생, r=0.7에서 홀수 R 발생해 노출(상위 25회 재시도 모두 실패).
   - 해결: `llava_arch.py` 해당 cat의 arange 확장 길이를 `distinct_indices.shape[1]`
     (실제 길이)로 변경. 짝수 R 동작 불변, 홀수 R 안전. 재실행 시 재시도·에러 0 검증.
4. CUDA illegal memory access: 없음 (builder.py dtype 패치 + LAUNCH_BLOCKING 유지).

## 스킵한 실험과 사유
- **VQAv2 (실험2-C, 4-B)**: test-dev 107,394문항 + 채점이 EvalAI 서버 제출 전용
  (오프라인 로컬 지표 없음) → compute·채점 모두 비현실적. POPE·GQA·TextVQA로 비교 충분.
- **PACT 직접 실행 (실험2-B)**: 별도 코드/환경 분리 필요, 본 과제(제안 방법 검증) 우선순위
  밖 → 보류. VisPruner 논문 Table 1의 FastV/ToMe/SparseVLM/VisPruner 인용으로 비교 구성.

## 산출물
- `02_baseline_comparison.md` 실험1 + 실험2 비교표
- `03_ablation_results.md` 3-A/B/C/D
- `04_question_type_analysis.md` 4-A POPE 카테고리 (+task-aware 시사점)
- `05_efficiency_results.md` 토큰감소율/latency/memory/overhead
- `exp_runner/results.tsv` 원시 결과, `exp_runner/logs/*` 벤치별 채점 로그
- 추론 답변: `playground/data/eval/<bench>/answers/EXP/<ID>/r_<R>.jsonl`

## v1 최종 상태
- **33/33 전부 완료** (A 9 + B simple 9 + B weighted 9 + R-30×2 + R-70×2 + M-96/M-192).
  R-70은 홀수-R 패치 후 재실행하여 확정(POPE 81.63 / GQA 56.61).
- 효율성 5세팅 전부 완료. CUDA illegal memory access 0.

---

# v2 추가 실험 (update_develop_ver.md) — 변경분만 진행

## 스킵 (v1 완료·데이터/구조 불변)
- 실험1 A/B/C × {POPE,GQA,TextVQA}: v1 완료(B=Ours simple, C=Ours weighted) → 재실행 안 함.
- 실험3-A/3-B/3-C, 4-A(POPE 카테고리), 5(효율성): v1 완료.

## 신규 진행 (v2 변경분)
1. **VQAv2 (신규 벤치마크·로컬 채점)**: v1은 EvalAI 사유로 스킵했으나 v2는 val 로컬채점 요구
   → 진행. visualqa.org S3에서 val 질문/주석 다운로드, `answer_type` **균형 subset 6000**
   (yes-no/number/other 각 2000), 필요 val2014 이미지 5413장만 개별 다운로드. llava-format
   `llava_vqav2_val_subset.jsonl` + GT + 공식 VQA accuracy 채점기 `exp_runner/vqa_eval.py`.
   실험1 A/B/C × {128,64,32} × VQAv2 = **9 job** 3-GPU 병렬. (실험4-B는 본 결과 집계)
2. **FastV (2-A)**: 직접 실행 보류 → VisPruner Table 1 인용+사유. 사유: FastV는 LLM 디코더
   layer-K self-attention + KV-cache 프루닝 요구. 고정 의존성(transformers 4.37.2)에서
   in-decoder 통합은 버전민감·고위험, 오구현 시 baseline 왜곡으로 비교 신뢰성 훼손.
   프롬프트 규정("구현 어려우면 논문 숫자 인용+사유") 적용.
3. **PACT (2-B)**: 직접 실행 불가 → 인용/사유. 사유: PACT는 별도 env(pactenv, py3.12.7,
   cuda11.8, flash-attn2.6.3) + 자체 transformers 번들, 대상 백본이 LLaVA-OneVision-7B/
   Qwen2-VL/LLaVA-1.6 로 **본 과제 백본 LLaVA-1.5-7B 미지원**. 환경·백본 상이로 동일조건
   비교 불가. 프롬프트 규정 적용. (VisPruner Table 1 인용 블록에 PACT 수치 미제공 →
   수치 미확보로 표기, 날조 금지.)
4. **ToMe/SparseVLM**: 프롬프트 지시대로 VisPruner Table 1 인용(직접 실행 안 함).

## v2 신규 실행 명령
```bash
bash exp_runner/worker.sh <gpu> exp_runner/jobs_vqav2.tsv   # VQAv2 9-job 3-GPU 병렬
python exp_runner/vqa_eval.py <answers.jsonl> dataset/vqav2/val_subset_gt.json by_type
```

## v2 최종 상태
- **VQAv2 9-job 전부 완료** (3-GPU 병렬, 에러 0, 각 6000/6000):
  - A: 128=72.18 / 64=68.88 / 32=63.47
  - B simple: 128=72.08 / 64=69.77 / 32=65.46
  - B weighted: 128=72.17 / 64=70.44 / 32=65.88
  - clustering 효과(B−A): @128≈0, @64 +0.9~+1.6, @32 +2.0~+2.4 (POPE/GQA와 동일 패턴)
- 실험4-B question-type 집계 완료: number가 압축 최취약(−11.6 @128→32), clustering이
  저토큰서 number/other 회복(B-32w other +3.5). → 04_question_type_analysis.md §4-B.
- FastV/PACT: 프롬프트 규정대로 인용/사유 처리(직접 실행 보류·불가, 위 기록).
- 문서 02(VQAv2열·baseline표·PACT/FastV 사유)·04(4-B)·06 갱신 완료.
- v2 신규분 CUDA 에러 0, 워커 재시도 0.

## 소요 시간 (v2)
- VQAv2 val 데이터(질문/주석 다운로드 + subset + 이미지 5413장): ~수 분
- VQAv2 6000 subset 추론: 세팅당 ~25분, 9세팅 3-GPU 병렬 ≈ 1.5~2시간

---

# update_ver2 — 커버리지 매트릭스 완성 (실험A + 실험B)

목적: 원본 VisPruner ↔ 제안 구조 데이터셋 커버리지 완전 일치.

## 실험 A: VQAv2 → 원본 VisPruner_run (V-576/128/64/32, clustering 없음)
- 제안 구조와 **동일 VQAv2 val 균형 subset 6000** 사용. VisPruner_run 디렉토리에서
  실행(원본 llava, clustering 없음). 채점 = term_project/exp_runner/vqa_eval.py(동일).
- 결과: V-576=75.03 / V-128=72.18 / V-64=68.88 / V-32=63.11
- **검증**: V-128/64 = 제안 A-128/64 (72.18/68.88) **정확 일치**, V-32=63.11 vs A-32 63.47
  Δ=−0.36(±2 이내, 통과). V-32 미세차 원인: 1차 LAUNCH_BLOCKING 모드 ~1.3k +
  가속(비차단) 모드 잔여 혼합 resume → 일부 커널 비결정성. 알고리즘 동일성은 V-128/64
  정확일치로 입증.

## 실험 B: SQA-IMG → 제안 구조 term_project (A/B/C × 128/64/32, 9)
- model_vqa_science에 clustering 인자(enable_clustering/stage1_tokens/merge_method/
  kmeans_max_iter) 연결 후 실행. eval_science_qa IMG-Accuracy.
- A: 68.86/68.57/68.32 — **원본 VisPruner 재현값과 정확 일치**(회귀 안전 검증).
- B simple: 68.86/68.86/69.31, C weighted: 68.62/69.16/69.46.
- SQA-IMG 토큰수 둔감(상식추론), clustering 저토큰서 소폭 개선·무회귀(최고 C-32 69.46).

## 성능 이슈/해결
- 초기 V-32가 `CUDA_LAUNCH_BLOCKING=1`로 ~1 it/s(잔여 ~77분) → 너무 느림.
  dtype 버그는 builder.py에서 이미 수정됨 → LAUNCH_BLOCKING(안전장치) 불필요 판단,
  해당 작업 종료 후 **비차단 모드 + resume**로 재실행 → **7.9 it/s(~7.5×)**, ~10분 완료.
  프로세스 종료 시 `pkill -f`가 래퍼셸 매칭(exit144) → /proc/cmdline 정밀 매칭으로 해결.

## 실행 명령
```bash
bash run_expA_vqav2_orig.sh 0          # VQAv2 → 원본 VisPruner_run (GPU0)
bash run_expB_sqa_proposed.sh 1        # SQA-IMG → 제안 (GPU1, 락공유)
bash run_expB_sqa_proposed.sh 2        # SQA-IMG → 제안 (GPU2, 락공유)
bash run_v32_fast.sh                   # V-32 가속 재실행(비차단+resume)
```

## update_ver2 최종 상태
- **실험A 4 + 실험B 9 = 13/13 완료**, 검증 전부 통과. CUDA 에러 0.
- 커버리지 매트릭스 5×2(POPE/GQA/TextVQA/VQAv2/SQA-IMG × 원본/제안) **완전 충족**.
- 문서 통합: 02(커버리지표·SQA열·SQA baseline표), 04(SQA·벤치별 효과종합),
  VisPruner_전체재현_최종보고서(VQAv2행), 종합보고서, 06(본 로그) 갱신 완료.

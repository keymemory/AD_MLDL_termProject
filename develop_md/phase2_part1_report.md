# Phase 2 Part 1 — 데이터셋 확장 + 건강 τ/k 확인 (구현·검증 보고)

> **작업 A**(MME·MMBench 추가) + **작업 B**(topk 작동 검증) + **작업 C**(M2별 건강 τ/k) 수행 보고.
> 규칙대로 확인 질문 없이 진행, 오류는 자동 수정 후 기록(§4).

---

## 0. 결과 요약

| 작업 | 상태 | 결과 |
|---|---|---|
| **C — M2별 건강 τ/k** | ✅ | `phase2_healthy_range.md` (probe 42조합) |
| **A — MME 통합** | ✅ | 이미지 추출 + `mme_eval.py`(parquet GT) + worker.sh `mme` 케이스 |
| **A — MMBench 통합** | ✅ | `model_vqa_mmbench.py` selection 인자 + `mmbench_eval.py` + worker.sh `mmbench` 케이스 |
| **B — topk 검증** | ✅ | 소규모(MMBench 68.00/MME unmatched=0) + worker.sh full(MME 1752.57 / MMBench 72.13, results 기록) |

---

## 1. 작업 C — M2별 건강 τ/k (probe, 핵심)

`probe_dist.py`(LLM 디코딩 없이 vision tower+selection만)로 M2=32/64/128 ×
energy τ(0.3~0.9) + statistical k(0.2~0.8) = 42조합 분포 측정. 상세 표: `phase2_healthy_range.md`.

### 종합 — M2별 건강 구간 (adapt% ≥ 90)

| M2 | energy 건강 τ | statistical 건강 k |
|---|---|---|
| 32 | 0.6~0.8 | 0.2~0.6 |
| 64 | 0.7~0.8 | 0.2~0.3 |
| **128** | **0.8만** | **없음(전구간 붕괴)** |

**핵심 발견 (Part 2 본실험 전제):**
- floor=M2이므로 **M2가 커질수록 건강구간이 급격히 좁아진다.** M2=128에선 energy는 τ=0.8 단일점만,
  statistical은 전 구간 floor 붕괴(이상치가 M1=128을 채울 만큼 안 나옴).
- → **단일 τ/k 고정은 위험.** Part 2는 각 M2의 건강 τ/k를 써야 adaptive가 topk와 구별된다.
  특히 M2=128 statistical은 본실험에서 제외하거나 r_floor 재설계 검토 필요.

---

## 2. 작업 A — MME·MMBench 통합

### 2-1. MME
- **데이터**: `MME/llava_mme.jsonl`(2374질문) + `hf_mme/*.parquet`(이미지 bytes). 단 `MME_Benchmark_release_version/`
  이미지 폴더·`eval_tool/` 채점도구는 **없음** → 직접 처리.
- **이미지 추출**: `exp_runner/extract_mme_images.py` — parquet bytes를 **llava_mme image 경로**로 저장(2-A 오류 참조).
- **추론**: `model_vqa_loader`(selection 인자 이미 있음) 그대로 사용. `--image-folder MME_Benchmark_release_version`.
- **채점**: `exp_runner/mme_eval.py` — 공식 eval_tool 우회, **parquet GT 직접**. MME score(각 category
  (acc+acc_plus)×100, perception 10 + cognition 4 합). 매칭은 `(category, basename, 정규화질문)`.

### 2-2. MMBench
- **데이터**: `mmbench/mmbench_dev_20230712.tsv`(4377질문, 정답 answer 포함, 이미지 base64). 완비.
- **추론**: `model_vqa_mmbench.py`에 **selection 인자 추가**(enable_clustering/stage1_tokens/merge_method/
  kmeans_max_iter/selection_method/energy_tau/stat_k/stat_robust). `--single-pred-prompt`로 객관식 직답.
- **채점**: `exp_runner/mmbench_eval.py` — 답변에서 A/B/C/D 추출 → tsv answer 비교 accuracy.

### 2-3. worker.sh 통합 (11컬럼 호환)
- `total_q`: mme=2374, mmbench=4377 추가.
- `case BENCH`: mme(llava_mme.jsonl, MME_Benchmark_release_version), mmbench(tsv, 이미지 무).
- **추론 엔트리 분기**: mmbench → `model_vqa_mmbench`(+single-pred-prompt), 그 외 → `model_vqa_loader`.
- **채점 분기**: mme → `mme_eval.py`(METRIC=MME), mmbench → `mmbench_eval.py`(METRIC=Acc).
- selection 인자(SEL_ARGS)·clustering(CL_ARGS) 그대로 전달 → adaptive selection 호환.

---

## 3. 작업 B — topk 작동 검증

| 벤치 | 규모 | 결과 |
|---|---|---|
| MMBench | dev 200 subset, topk M2=128 | **Acc 68.00 (136/200)** — LLaVA-1.5-7b 정상 범위 ✅ |
| MME | 200 subset(code_reasoning 40 + artwork 160), topk M2=128 | unmatched=0, perception 109.38 / cognition 40.00 ✅ |

→ 두 벤치 모두 topk 추론·채점 정상.

### worker.sh full 1 job 기록 검증 (완료, results_phase2.tsv)
| job | 결과 | 완주 |
|---|---|---|
| P2chk-mme (mme, topk M2=128) | **MME total 1752.57** | 2374/2374 |
| P2chk-mmb (mmbench, topk M2=128) | **Acc 72.13** | 4377/4377 |

→ worker.sh 11컬럼 체계로 MME/MMBench 정상 추론·채점·기록 확인. **작업 B 완전 통과.**
(LLaVA-1.5-7b M2=128 topk 기준 정상 범위: MME~1750, MMBench~72.)

> 참고: MMBench는 `model_vqa_mmbench`가 `_adaptive_log` 집계를 안 해 AVG_M1/floor/cap이 "-"로 기록됨
> (topk라 무관). Part 2에서 MMBench로 adaptive selection을 쓰려면 `model_vqa_mmbench.py`에도
> .meta 집계(model_vqa_loader 방식) 추가가 필요하다.

---

## 4. ★ 오류 자동 수정 기록 (지시서 규칙)

| # | 오류 | 원인 | 수정 |
|---|---|---|---|
| 1 | MME parquet 읽기 실패 | vispruner env에 pyarrow 없음 | `pip install pyarrow` |
| 2 | MME 추론 FileNotFound(artwork) | parquet question_id(`artwork/14777.jpg`)와 llava_mme image(`artwork/images/14777.jpg`) 경로 체계 상이(카테고리별 images/ 유무) | extract를 **llava_mme image 경로 기준**으로 재작성, parquet은 (category,basename)로 매칭 |
| 3 | MME 채점 unmatched=160(artwork) | answers question_id와 parquet question_id의 images/ 차이 | mme_eval 매칭 키를 `qkey=(category, basename)`로 정규화 |
| 4 | analyze_healthy 출력 실패 | `develop_md/`가 term_project 부모에 있음(cd 후 상대경로 깨짐) | 절대경로 출력 |
| 5 | E2 phase1_e2_result.md 빈 파일 | `run_e2_chain.sh`가 cd term_project 후 `develop_md/`에 써서 경로 어긋남 | 올바른 경로로 `generate_e2_report.py` 재실행(92줄 생성) |

---

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `develop_md/phase2_healthy_range.md` | M2×τ/k 42조합 분포 + 건강구간 (작업 C) |
| `exp_runner/probe_dist.py` | M2 파일명 포함하도록 수정(probe_{method}_M{M2}_{p}) |
| `exp_runner/analyze_healthy.py` | 건강구간 집계 |
| `exp_runner/extract_mme_images.py` | MME parquet→이미지 추출 |
| `exp_runner/mme_eval.py` | MME 채점(parquet GT) |
| `exp_runner/mmbench_eval.py` | MMBench 채점 |
| `llava/eval/model_vqa_mmbench.py` | selection 인자 추가 |
| `exp_runner/worker.sh` | mme/mmbench 케이스 통합 |

---

## 6. Part 2 본실험 권고 (이번 미수행)

1. **건강구간만 사용**: M2별 건강 τ/k(§1 표). M2=128 statistical은 붕괴라 제외/재설계.
2. **전 구간 성능 측정**(붕괴 구간 포함)으로 "왜 건강구간이 최적인지" 정당화.
3. 전 데이터셋(POPE/GQA/TextVQA/VQAv2/SQA/MME/MMBench) × M2(32/64/128) × {topk, energy, statistical} × {simple, weighted}.
4. job tsv → `launch.sh`(또는 launch_phase1 패턴) 병렬 → results 집계.

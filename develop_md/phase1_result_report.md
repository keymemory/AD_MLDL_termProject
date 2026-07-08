# Phase 1 — 구현·회귀검증 완료 보고서 (→ 클로드 웹 공유용)

> **작성**: VSCode Claude Code. **단계 0~6 구현 완료 + 단계 7 회귀 검증(POPE 300 subset) 비트동일 통과.**
> **상태**: `--selection_method {topk,energy,statistical}` 추가 완료, topk default = 기존 VisPruner **비트동일** 입증.
> **다음**: full POPE 확정 / energy·statistical 동작 검증 / E2 본실험 — 결정 대기(아래 §6).

---

## 1. 회귀 검증 결과 — 통과 ✅ (핵심)

동일 POPE 300 subset에서 **구코드 reference** ↔ **신코드 topk(default)** 를 대조:

| 검증 항목 | reference (구코드) | 신코드 topk | 결과 |
|---|---|---|---|
| 줄수 | 300/300 | 300/300 | ✅ |
| **답변 텍스트 300개** | — | **text_diff = 0** | ✅ 전부 일치 |
| **POPE F1** | 0.8519353659428237 | 0.8519353659428237 | ✅ 소수점 16자리 동일 |
| `.meta` 집계 | — | `avg_M1=128.0, avg_r=0.5, n=300` | ✅ 정상 |

- **비트동일 증명**: 300개 답변이 한 글자도 다르지 않고 F1이 끝자리까지 동일 →
  `selection_method="topk"` default가 기존 VisPruner 경로를 **0 영향**으로 보존.
- **방법**: reference는 구코드(코드 수정 전)로 POPE 300 subset(M2=128, clustering off)을 먼저 떠서
  스냅샷 확보 → 코드 수정(단계 1~6) → 신코드를 `selection_method` 미지정(=topk)으로 동일 추론 → 대조.
  answer_id는 매 실행 랜덤이라 question_id별 **답변 텍스트**로 비교(정확한 비트동일 비교).
- ※ subset F1(0.8519)은 full A-128(0.8446)과 **문항 수가 달라** 값 자체는 다름. 회귀의 본질인
  "구코드↔신코드 동일 출력"은 이 subset에서 완전히 입증됨. (full 8910 확정은 §6의 옵션)

---

## 2. 구현 완료 내역 (단계 0~6)

| 단계 | 파일 | 변경 내용 |
|---|---|---|
| **0** | launch.sh·worker.sh·sanity_check.sh·models·playground/data/eval·config.json | **옛 SKKU 경로 5종 패치**(아래 §3) |
| **1·2** | `llava/eval/model_vqa_loader.py`, `model_vqa_science.py` | `--selection_method/--energy_tau/--stat_k/--stat_robust` 4인자 + load_pretrained_model kwargs |
| **3** | `llava/model/language_model/llava_llama.py` | `__init__` 파라미터 4개 + getter 4개(`getattr+default`) + `_adaptive_log` 버퍼 |
| **4** | `llava/model/adaptive_selection.py`(**신규**) + `llava_arch.py` | `compute_adaptive_counts` 헬퍼 + `encode_images` topk/energy/statistical 분기 |
| **5** | `llava/eval/model_vqa_loader.py` | 추론 후 `_adaptive_log` 평균 → `<answers>.meta` 출력(보완 A) |
| **6** | `exp_runner/worker.sh` | job 11컬럼 파싱+topk 폴백 + `$SEL_ARGS` + `.meta` 읽어 results.tsv에 SELMETHOD/AVG_M1/AVG_R 컬럼 추가(보완 B) |

**변경 규모**: tracked 9파일 (+109 −17), 신규 1파일(`adaptive_selection.py`).
**원칙 준수**: `forward`는 dead path라 **건드리지 않음**(보고서 C). 모든 신규 getter default=기존 동작.

### 2-1. 핵심 — encode_images 분기 (회귀 안전 설계)
```python
# [공통] [CLS] attention saliency 점수 (head 평균)
image_attentions = image_attentions.mean(dim=1)
selection_method = self.get_selection_method()
if selection_method == "topk":
    # 기존 VisPruner 경로 (고정 M1·r) — 결과 비트동일
    M1 = self.get_stage1_tokens() if enable_clustering else M2
    important_token_num = int(M1 * self.get_important_ratio())
    diverse_token_num = M1 - important_token_num
else:
    # (가)energy / (나)statistical — 이미지별 가변 n_imp·M1 (batch_size=1)
    n_imp_list, M1_list, _info = compute_adaptive_counts(
        image_attentions, selection_method, M2,
        energy_tau=self.get_energy_tau(), stat_k=self.get_stat_k(),
        stat_robust=self.get_stat_robust())
    important_token_num = n_imp_list[0]; M1 = M1_list[0]
    diverse_token_num = M1 - important_token_num
# [진단] (n_imp, M1) 누적 — 출력 무영향
if hasattr(self, "_adaptive_log"):
    self._adaptive_log.append((int(important_token_num), int(M1)))
# 이하 기존 diverse 루프 + Stage2 그대로
```

### 2-2. compute_adaptive_counts (adaptive_selection.py 신규)
- **energy(가)**: 내림차순 [CLS]attn 누적질량 ≥ τ 최소 토큰 = n_imp (적분 관점)
- **statistical(나)**: `μ+kσ`(또는 `--stat_robust`면 `median+k·MAD`) 이상치 개수 = n_imp (전역통계)
- **공통 M1 유도**: `M1 = clamp(round(n_imp/0.5), floor=M2, cap=384)`, `n_imp ≤ M1`, `r = n_imp/M1`(자동)
- 단위 동작 확인(random 점수): energy→n_imp=167/M1=334, statistical→floor 도달(n_imp=1/M1=64). floor/cap 정상.

---

## 3. 작업 중 발견·처리한 이슈

### 3-1. ★ 옛 경로 5종 깨짐 (Phase1 §0은 3종이었으나 실측 5종)
디렉토리가 `SKKU_Works/...` → `AD_MLDL_termProject/`로 이동되며 깨진 경로 — **전부 복구 완료**:
1. `worker.sh`·`launch.sh`·`sanity_check.sh`의 `TP=`/`cd` 경로
2. `term_project/models` 심볼릭링크 → `../VisPruner_run/models` 재생성
3. `config.json` `mm_vision_tower` 절대경로
4. **(신규 발견) `term_project/playground/data/eval` 데이터 심볼릭링크** → `AD_MLDL_termProject/dataset` 재생성
5. **(신규 발견) `sanity_check.sh`의 `cd` 경로** — TP와 함께 패치

### 3-2. CUDA illegal memory access — `CUDA_LAUNCH_BLOCKING=1` 필요
- reference 1차 시도(blocking 미설정)가 7문항 만에 `illegal memory access`로 사망(llava_arch.py:172).
- dtype 패치는 정상(builder.py L45 model fp16 / L161 vision tower fp16 캐스팅 확인). 원인은 비동기 실행.
- **`CUDA_LAUNCH_BLOCKING=1`로 재시도 → 정상 완주**(worker.sh도 항상 =1). 모든 추론은 이 설정 권장.

### 3-3. GPU 처리
- GPU 0·1을 점유한 vllm(`google/gemma-4-26B`)은 **타 사용자(`sihwang`) 프로세스** → 종료 안 함
  (권한 없음 + 남의 실행 작업 파괴 불가). 회귀 추론은 **GPU 2번에서 정상 완주**.

---

## 4. 결과 출력 스키마 (확장됨)

- **job tsv (11컬럼)**: `ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST`
  - 기존 7컬럼 job도 그대로 동작(worker.sh가 뒤 4컬럼 없으면 `topk 0.5 2.0 0` 폴백).
- **results.tsv (13컬럼)**: 기존 10컬럼 + `SELMETHOD AVG_M1 AVG_R`
  - AVG_M1/AVG_R은 `.meta`에서 읽은 **실제 평균**(energy/statistical은 이미지별 가변 → 평균, topk는 고정값).
- **`.meta`**: `{"selection_method","count","avg_n_imp","avg_M1","avg_r"}` 한 줄 JSON.

---

## 5. Phase 1 done 조건 대비 진행

| § | done 조건 | 상태 |
|---|---|---|
| 6-1 | §0 경로 패치, 추론 정상 | ✅ (5종 패치) |
| 6-2 | 4인자 4 touch-point 추가 | ✅ |
| 6-3 | **회귀 통과(topk=기존 비트동일)** | ✅ **통과** |
| 6-4 | energy/statistical 이미지별 M1·r 자동결정, floor/cap | ⏳ 단위테스트 OK, **실추론 동작검증 미수행** |
| 6-5 | results.tsv에 SELMETHOD/AVG_M1/AVG_R | ✅ (구현 완료, E2서 실데이터) |
| 6-6 | E2 매트릭스 완주 + τ*/k* 역산 | ⏳ **미수행(별도 지시 대기)** |

---

## 6. ★ 다음 단계 — 결정 요청

회귀가 subset에서 통과했습니다. 다음 중 선택(권장 순서: **2 → 1 → 3**):

1. **full POPE(8910) 비트동일 최종 확정** — 2단계 계획의 2단계. 구코드 reference 8910 + 신코드 topk 8910
   대조. blocking=1로 각 ~2–3시간. *기존 A-128 answers 파일이 보존돼 있으면 그것과 직접 대조해 시간 단축 가능*(확인 필요).
2. **energy/statistical 동작 검증** (done 조건 6-4) — POPE 300 subset에 energy·statistical 각 1회 추론하여
   (a) 안 죽고 (b) `.meta`에 가변 AVG_M1·AVG_R이 찍히는지 (c) floor/cap 비율 확인. ~10분. **가장 가벼우며 구현 신뢰도 확보.**
3. **E2 본실험** — τ 스윕(0.5~0.9)/k 스윕(1.5~2.5)/robust, POPE+GQA + τ*/k* 역산. (Phase1 §5, 별도 지시 대기)

> 권장: **2번(동작 검증)으로 energy/statistical이 실제 작동함을 먼저 확인** → 1번(full 확정) → 3번(E2).
> 클로드 웹에서 진행 방향을 정해 회신해 주시면 그대로 수행하겠습니다.

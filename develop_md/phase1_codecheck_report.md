# Phase 1 — Claude Code 사전검증 보고서 (→ 클로드 웹 회신용)

> **작성**: VSCode Claude Code (현재 repo를 직접 읽어 대조). **코드 수정 전, 확인·계획만.**
> **대상**: `develop_md/phase1.md`(Selection 수식 구현 지시서) 실행 가능성 검증.
> **결론 한 줄**: md 좌표는 **전부 정확**, batch_size=1이라 **가변길이 문제 해소**, 경로 3종 **실제로 깨짐 확인**.
> md에 **빠진 추가 수정 지점 2개 + 함정 1개**를 발견 → 아래 §4. 진행 전 **결정 2가지** → §6.

---

## 1. 좌표 검증 — phase1.md vs 실제 코드 (✅ 전부 일치)

| phase1.md 좌표 | 실제 위치 | 결과 |
|---|---|---|
| `llava_arch.py` L158~162 (important 선택) | L158~162 | ✅ 정확 |
| diverse 선택 L164~191 | L164~191 | ✅ |
| Stage2 병합 호출 L196~213 | L196~213 | ✅ |
| `llava_llama.py` `__init__` L44~65 | param L44~46 / 저장 L54~62 / post_init L65 | ✅ |
| `llava_llama.py` getter L71~89 | L71~89 | ✅ |
| `model_vqa_loader.py` argparse L168~175 | L168~175 (clustering args L171~175) | ✅ |
| `model_vqa_loader.py` 모델생성 L86~92 | 호출 L85, kwargs L87~92, 닫음 L93 | ✅ (1줄 오프셋, 무시 가능) |

→ **md의 4 touch-point 체인 좌표를 그대로 신뢰하고 작업 가능.**

---

## 2. ★ batch_size 확인 (phase1.md §2-4 핵심 분기점)

`term_project/llava/eval/model_vqa_loader.py` **L72~73**:
```python
def create_data_loader(..., batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
```
- **평가 추론은 무조건 `batch_size=1`** — 기본값 + `assert`로 강제. 호출부(L119)도 인자 없이 = 1. generate도 input_ids B=1.
- **결론**: phase1.md §2-4가 우려한 "이미지별 가변 M1 → 배치 텐서 처리 불가" 문제는 **실질적으로 없음.**
  `encode_images`의 `for b in range(B)` 루프는 항상 **1회**만 돈다.
- **구현 단순화**: 기존 per-image 루프(Stage2가 이미 `for b in range(B)`)를 그대로 쓰면 B=1에서 자연 동작.
  별도 가변길이 패딩/분기 불필요. §2-4의 "B>1 per-image 루프"는 **작성은 하되 실제론 안 타는 안전망**으로만.

---

## 3. 경로 3종 깨짐 — 전부 실측 확인 (phase1.md §0 정확, 그대로 적용 가능)

| 대상 | 현재 가리키는 곳(옛 SKKU 경로) | 상태 |
|---|---|---|
| `exp_runner/worker.sh`·`launch.sh` `TP=` | `.../SKKU_Works/.../term_project` | 🔴 깨짐 |
| `term_project/models` 심볼릭링크 | `.../SKKU_Works/.../VisPruner_run/models` | 🔴 깨짐(타겟 없음) |
| `config.json` `mm_vision_tower` | `.../SKKU_Works/.../clip-vit-large-patch14-336` | 🔴 깨짐 |

- 실제 모델은 `../VisPruner_run/models`에 **존재 확인**. phase1.md §0의 `sed`/`ln` 패치 3종 그대로 적용 OK.
- 추가 확인값: `config.json`의 `mm_vision_select_layer: -2`, `image_aspect_ratio: "pad"`,
  `mm_patch_merge_type: flat`(단일 이미지) — Stage2 clustering 단일이미지 경로 전제와 일치.

---

## 4. ★ phase1.md에 빠진 것 / 함정 (보완 필요)

확인 중 발견한, **md §2에 명시되지 않은 추가 수정 지점 2개 + 함정 1개**:

### A. §4 결과 집계(AVG_M1/AVG_R)는 `model_vqa_loader.py` 수정이 **추가로** 필요
- md §2의 touch-point엔 이 파일의 **집계 코드가 없음**.
- 실제 흐름: `generate`가 반환하는 `visual_token_num`([llava_llama.py L178])은
  `image_features[0].shape[0]`([llava_arch.py L432]) = **최종 M2(병합 후)**일 뿐 **M1/n_imp가 아님.**
- energy/statistical의 실제 M1·r을 기록하려면(md §4-3 방식 채택):
  1. 모델에 누적 버퍼 attribute `self._adaptive_log` 추가 → `encode_images` 내부에서 (n_imp, M1) push
  2. `model_vqa_loader.py`가 추론 종료 후 평균 계산 → `<answers>.meta` 파일로 출력
  3. `worker.sh`가 `.meta`를 읽어 results.tsv의 `AVG_M1`/`AVG_R` 컬럼에 기록
- **generate 반환 시그니처는 안 건드리는 게 안전**(건드리면 L159·L130 연쇄 수정 발생).

### B. `worker.sh`의 job 파싱 라인도 확장 필요
- md §4-2는 job tsv에 4컬럼(`SELMETHOD ETAU SK SROBUST`) 추가를 지시.
- 그러나 `worker.sh`의 `while read -r ID BENCH M2 CLUST M1 METHOD R`(**7컬럼**)을
  **11컬럼으로 확장** + **뒤 4컬럼이 없으면 topk로 폴백**(하위호환)하도록 짜야
  §3 회귀검증 job(7컬럼)도 안 깨짐. (md엔 read 라인 수정이 명시 안 됨)

### C. 함정 — `forward`는 dead path, **건드리지 말 것**
- `llava_llama.py` `forward`(L107~115)는 `prepare_inputs`를 **6개로 unpack**하지만,
  실제 함수는 **7개를 반환**([llava_arch.py L432]). → `forward` 호출 시 ValueError.
- 단 **평가는 `generate`(7-unpack, 정상)만 사용** → `forward`는 안 탐. Phase1(평가 전용)에 영향 없음.
- ⚠️ 이걸 모르고 `forward`를 "수정"하려 들면 혼란 유발. **Phase1에서 `forward`는 손대지 않는다.**

---

## 5. 구현 계획 (B=1 전제로 단순화)

| 단계 | 작업 | 파일 |
|---|---|---|
| **0** | 경로 3종 패치(sed/ln) + `pip install -e .` 확인 | worker.sh, launch.sh, config.json, models 링크 |
| **1** | CLI 인자 4개 추가(`--selection_method/--energy_tau/--stat_k/--stat_robust`) | model_vqa_loader.py (+science.py) |
| **2** | 모델 생성 kwargs 전달 | model_vqa_loader.py L85~93 |
| **3** | `__init__` 파라미터 + 4 getter(`getattr+default`) | llava_llama.py |
| **4** | `compute_adaptive_counts` 헬퍼 신규 + L158~162 분기(topk/energy/statistical) | **신규** `adaptive_selection.py` + llava_arch.py |
| **5** | (보완 A) `_adaptive_log` 버퍼 push + model_vqa_loader 집계 → `.meta` | llava_arch.py, model_vqa_loader.py |
| **6** | (보완 B) worker.sh 11컬럼 파싱 + SEL_ARGS + results.tsv 컬럼 확장 | worker.sh |
| **7** | **회귀 검증**(topk=기존 비트동일) → 통과 후에만 E2(τ/k 스윕, POPE+GQA) | exp_runner |

- **신규 헬퍼는 `llava/model/adaptive_selection.py`로 분리** 권장(llava_arch.py 오염↓, 회귀 격리).
- 회귀 안전 원칙: 모든 신규 getter default = 기존 동작. `selection_method="topk"`면 L194~199 기존 경로
  **한 글자도 안 바꿈** → A-시리즈 수치 비트동일 보장.

---

## 6. ★ 진행 전 결정 필요 2가지 (클로드 웹/사용자 판단)

막히는 부분은 없음(위 A·B는 Claude Code가 보완 구현 가능). 다만 다음 2개는 방침 결정 필요:

1. **회귀 검증 규모** — phase1.md §3은 full POPE(**8910문항**)로 RT-A128 실행 → 수십 분 소요.
   - (권장) **POPE 300 subset으로 빠른 회귀+동작 확인 → 통과 시 full로 확정**(2단계).
   - 또는 처음부터 full POPE.
   → 어느 쪽?

2. **신규 헬퍼 위치** — `adaptive_selection.py` **별도 파일**(권장, 회귀 격리) vs `llava_arch.py` 상단 인라인.
   → 어느 쪽?

> 위 2가지만 정해지면 Claude Code가 §5 순서대로 **§0 경로패치 → 코드 구현 → 회귀 검증**까지 진행.
> phase1.md 본문은 **거의 수정 불필요**(좌표 정확). 단 §4 집계가 `model_vqa_loader.py`까지 닿는다는 점과
> worker.sh read 라인 11컬럼 확장(§4-B)만 phase1.md에 한 줄씩 반영해두면 지시서가 완전해짐.

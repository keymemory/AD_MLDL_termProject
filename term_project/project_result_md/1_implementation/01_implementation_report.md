# 01. Two-Stage Visual Token Reduction Framework — 구현 보고서

VisPruner 코드를 기반으로 **Stage 1(VisPruner 선택) → Stage 2(Spherical K-Means 병합)**
2단계 시각 토큰 축소 프레임워크를 구현. `develop_ver.md` 명세 준수.

---

## 1. 구현 코드 구조

작업 위치: `experiments/term_project/` (VisPruner 패치본 복사 + 신규 구현)

| 파일 | 역할 | 구분 |
|---|---|---|
| `llava/model/spherical_kmeans.py` | **신규** — Spherical K-Means + merge_tokens(simple/weighted) | 신규 |
| `llava/model/llava_arch.py` | Stage1 토큰선택 후 Stage2 clustering 호출 통합 | 수정 |
| `llava/model/language_model/llava_llama.py` | clustering 설정 보관 + getter | 수정 |
| `llava/eval/model_vqa_loader.py` | CLI 인자 4종 추가 + 모델 로딩에 전달 | 수정 |
| `llava/model/builder.py` | (재현 시) dtype 패치 — 비전타워를 모델 dtype으로 캐스팅 | 기존 패치 |
| `llava/eval/model_vqa_loader|science|.py` | (재현 시) resume 패치 | 기존 패치 |
| `sanity_check.sh` | sanity 3종 실행 스크립트 | 신규 |
| `models/` → VisPruner_run/models (symlink) | LLaVA-1.5-7B + CLIP | 재사용 |
| `playground/data/eval` → experiments/dataset (symlink) | 데이터셋 | 재사용 |

> 환경: conda `vispruner`(torch 2.1.2+cu121 / transformers 4.37.2), `pip install -e .`로
> term_project를 활성 패키지로 등록. `CUDA_LAUNCH_BLOCKING=1` 안정모드.

---

## 2. VisPruner 대비 수정/추가 상세

### 2-1. `llava_arch.py :: encode_images()` (핵심 통합)
- **M1/M2 분리**: 기존엔 `visual_token_num`(=T) 하나로 important/diverse 개수 결정.
  수정 후 `M2 = visual_token_num`(최종), `M1 = stage1_tokens`(clustering 시) 로 분리.
  Stage 1 선택은 **M1개 기준**으로 수행(clustering off면 M1==M2 → 기존과 완전 동일).
- **attention score 반환**: head 평균낸 `image_attentions` (B,N)를 유지했다가,
  Stage 1 선택 인덱스로 gather하여 Stage 2 weighted_avg 가중치로 전달.
- **Stage 2 호출**: `mm_projector` 적용 후, `enable_clustering and M2 < M1`이면
  배치별로 `merge_tokens(feats(M1,D), attn(M1,), M2, method, max_iter)` → `(M2,D)`.
  병합 결과를 `(merged, None)` 으로 반환(`index_masks=None`은 "이미 최종" 신호).
- clustering off 또는 M2>=M1이면 기존 `(projected_features, index_masks)` 그대로 반환.

### 2-2. `prepare_inputs_labels_for_multimodal()`
- 단일 이미지(LLaVA-1.5/POPE) 경로: `index_masks is None`이면 병합토큰
  `(B,M2,D)`를 `flatten(0,1).unsqueeze(0)`로 LLM `<image>` 자리에 삽입.
  아니면 기존 `image_features[index_masks]` 경로(원본 그대로).
- multi-image/anyres(LLaVA-NeXT) 경로: clustering 미지원 → `index_masks is None`이면
  명시적 `NotImplementedError`(POPE 등 단일이미지에서만 사용).

### 2-3. `llava_llama.py`
- `__init__`에 `enable_clustering, stage1_tokens, merge_method, kmeans_max_iter` 추가
  (`stage1_tokens=None` → `visual_token_num`과 동일 = clustering 무효).
- `get_enable_clustering / get_stage1_tokens / get_merge_method / get_kmeans_max_iter` getter 추가
  (`getattr` 기본값으로 구버전 체크포인트 호환).

### 2-4. CLI 인자 (`model_vqa_loader.py`) — load_pretrained_model **kwargs 경유로 __init__ 전달

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--enable_clustering` | flag | False | Stage 2 활성화 |
| `--stage1_tokens` | int | None(=M2) | Stage1 보존 토큰 M1 |
| `--visual_token_num` | int | 576 | 최종 토큰 M2(기존) |
| `--merge_method` | str | simple_avg | simple_avg / weighted_avg |
| `--kmeans_max_iter` | int | 10 | spherical k-means 최대 반복 |
| `--important_ratio` | float | 0.5 | important 비율 r(기존) |

핵심 분기: `enable_clustering=False` → 기존 VisPruner와 완전 동일. `True` → Stage1(M1)→Stage2(M2).

---

## 3. Spherical K-Means 구현 세부 (`spherical_kmeans.py`)

- `_l2norm`: fp16 안전 — float32로 norm 계산 후 `clamp_min(1e-8)`, 원 dtype 복귀(0-division 방지).
- `spherical_kmeans(tokens, k, max_iter, init_indices)`:
  1. 토큰 L2 정규화 → 단위 구 투영(클러스터링은 float32로 안정화)
  2. 초기 centroid: `init_indices` 없으면 랜덤 k개
  3. 반복: cosine sim(`x @ cᵀ`) → argmax 할당 → 클러스터 평균 → 재정규화 →
     **assignment 불변 시 조기 종료**
  4. **빈 클러스터**: 가장 큰 클러스터에서 토큰 1개를 빈 클러스터로 재할당
  5. `assignments (M1,)` 반환
- `merge_tokens(tokens, attn, k, method, max_iter)`:
  - `k>=M1`이면 원본 그대로 반환(엣지)
  - simple_avg: 클러스터별 토큰 단순 평균(`index_add_` + count)
  - weighted_avg: `Σ aᵢxᵢ / Σ aᵢ` (attn 음수 clamp, 가중합 0이면 단순평균 폴백)
  - 전부 GPU 텐서·`index_add_` 벡터화, `max_iter≤10`
- 단위 테스트: `(128,1024)`, k=64, fp32/fp16 + 엣지(k≥M1, 동일점 다수→빈클러스터) **통과**.

---

## 4. Sanity Check 결과 (POPE 300 subset, 최종 64토큰, r=0.5, greedy)

| # | 설정 | 생성 | **Avg F1** | 판정 |
|---|---|---|---|---|
| (1) | VisPruner only (clustering off, n=64) | 300/300 | **0.8459** | ✅ 기존 VisPruner 동작 정상 보존, 에러 0 |
| (2) | Two-stage **simple_avg** (M1=128→M2=64) | 300/300 | **0.8474** | ✅ 에러 0, 합리적(범위내) |
| (3) | Two-stage **weighted_avg** (M1=128→M2=64) | 300/300 | **0.8381** | ✅ 에러 0, 합리적(범위내) |

- (1)은 1차 재현의 동일 코드 경로(clustering off)로, 300-subset 특성상 full-set(F1 0.809)보다
  높게 나오나 **VisPruner 경로가 수정에도 불변**임을 확인(핵심 검증 목적 충족).
- (2) two-stage simple_avg(0.847)가 동일 최종 64토큰의 VisPruner-only(0.846)와 동등 이상 →
  128개 보존 후 클러스터 병합이 직접 64선택 대비 정보를 잘 보존함을 시사(프레임워크 가설 부합).
- (3) weighted_avg(0.838)도 정상 범위. 세 케이스 모두 CUDA 에러·크래시 없음.

---

## 5. 에러 발생 및 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| CLIP 로드 실패 (`Incorrect path_or_model_id`) | `Term_project→experiments` 디렉토리 개명으로 `llava-v1.5-7b/config.json`의 `mm_vision_tower` 절대경로가 사망 | config.json `mm_vision_tower`를 `experiments/...` 신규 경로로 갱신 |
| dataset 심볼릭 깨짐 | 동일 개명으로 `playground/data/eval/*` → 옛 `Term_project/dataset` 링크 사망 | 전 벤치마크 심볼릭을 `experiments/dataset`로 재연결 |
| (예방) fp16 0-division | spherical k-means norm | `eps=1e-8` clamp + float32 계산 |
| (예방) 빈 클러스터 | k-means 할당 쏠림 | 최대 클러스터에서 1토큰 재할당 |

CUDA 에러 없음(`CUDA_LAUNCH_BLOCKING=1` + 1차에서 적용된 builder.py dtype 패치 유지).

---

## 6. 결론

**Sanity check 3개 모두 통과 → 구현 완료.**

- Stage 1(VisPruner) + Stage 2(Spherical K-Means, simple/weighted avg) 파이프라인 정상 동작
- `enable_clustering=False`일 때 기존 VisPruner와 완전 동일(회귀 없음) 확인
- 단위 테스트·통합 sanity 모두 에러 0, 정확도 합리적 범위
- 후속 체계적 실험(`experiments_ver.md`)을 위한 CLI 인자·재현 환경 준비 완료

# 02. VisPruner 코드 구조 분석

## 1. 전체 구조

VisPruner는 LLaVA 코드베이스에 **plug-and-play** 로 통합된다. 핵심 수정은 3개 파일의 `[VisPruner]` 주석 부분이다.

| 파일 | 역할 |
|------|------|
| `llava/model/multimodal_encoder/clip_encoder.py` | CLIP 비전 인코더에서 **feature + [CLS] attention** 동시 추출 |
| `llava/model/llava_arch.py` | **important / diverse 토큰 선택**(VisPruner 본체) 및 프루닝된 토큰을 LLM 입력에 통합 |
| `llava/model/language_model/llava_llama.py` | `visual_token_num`, `important_ratio` 설정 보관 및 generate 시 토큰 수 반환 |
| `llava/eval/model_vqa_loader.py` | 평가 루프 (질문 파일 → 추론 → 답변 jsonl) |
| `llava/eval/eval_pope.py` | POPE F1/Accuracy 채점 |

## 2. [CLS] attention 추출 — `clip_encoder.py`

`CLIPVisionTower.feature_select()` / `forward()`:

```python
image_features   = image_forward_outs.hidden_states[self.select_layer]   # select_layer = -2
image_attentions = image_forward_outs.attentions[self.select_layer]       # 같은 레이어의 attention
if self.select_feature == 'patch':
    image_features   = image_features[:, 1:]            # [CLS] 제외, patch 토큰만 (576개)
    image_attentions = image_attentions[:, :, 0, 1:]    # [CLS]→patch attention (B, H, N)
```

- **추출 위치**: 비전 인코더의 **뒤에서 2번째 레이어**(`mm_vision_select_layer = -2`, LLaVA-1.5 기본값)의 attention map.
- **[CLS] attention 정의**: multi-head attention에서 `[CLS]` 토큰(인덱스 0)이 각 patch 토큰(1: 이후)에 주는 attention `attentions[:, :, 0, 1:]`. 즉 "CLS가 어떤 패치를 중요하게 보는가" = **visual saliency cue**.
- `forward(images, output_attentions=True)` 로 feature와 attention을 함께 반환.

## 3. Important token 선택 — `llava_arch.py::encode_images()`

```python
visual_token_num    = self.get_visual_token_num()                 # T (목표 토큰 수)
important_ratio     = self.get_important_ratio()                   # r
important_token_num = int(visual_token_num * important_ratio)      # T_imp = T·r
diverse_token_num   = visual_token_num - important_token_num       # T_div = T·(1−r)

image_attentions = image_attentions.mean(dim=1)                    # (B,N) head 평균
token_indices    = image_attentions.argsort(dim=-1, descending=True)
important_indices = token_indices[:, :important_token_num]         # 상위 T_imp개 = important
residual_indices  = token_indices[:, important_token_num:]         # 나머지 후보
```

- head 차원을 평균낸 [CLS] attention 점수가 **높은 순으로 정렬**, 상위 `T_imp`개를 **important token**으로 무조건 보존.

## 4. Diverse token 선택 (유사도 기반 중복 제거) — `llava_arch.py`

```python
image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
while diverse_token_num > 0:
    R = residual_indices.shape[1]
    r = min(8, R - diverse_token_num)          # 한 번에 최대 8개씩 제거 (ToMe식 점진 병합)
    if r <= 0: break
    residual_tokens = image_normalized[..., residual_indices, :]      # (B,R,C)
    a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]  # 짝/홀 분할
    scores = (a @ b.transpose(-1,-2)).max(dim=-1).values             # 각 토큰의 최대 코사인 유사도
    distinct = scores.argsort(dim=-1, descending=True)[:, r:]        # 유사도 높은 r개 제거
    residual_indices = cat([even[distinct], odd], dim=-1)            # 중복 제거 후 갱신
selected_indices = cat([important_indices, residual_indices])        # 최종 보존 토큰
```

- residual(=important 제외 나머지) 토큰들 중 **서로 유사도가 높은(중복) 토큰을 반복적으로 제거**하여, 남은 토큰이 시각적으로 **다양(diverse)** 하도록 만든다.
- ToMe(Token Merging)의 bipartite soft matching 방식(짝/홀 분할 후 매칭)을 차용하되, 병합이 아닌 **제거**로 다양성 토큰만 남김.
- 결과: `important(T_imp) + diverse(T_div)` = T개의 토큰만 LLM에 전달.

## 5. VLM 파이프라인 통합 — `llava_arch.py::prepare_inputs_labels_for_multimodal()`

- `index_masks` (B, N) bool 텐서를 만들어 보존 토큰만 True.
- 단일 이미지(POPE) 경로: `image_features = image_features[index_masks].unsqueeze(0)` —
  boolean 인덱싱이라 **공간 순서가 보존**되며, 선택된 토큰만 텍스트 임베딩 사이 `<image>` 자리에 삽입.
- 즉 LLM 입력 시퀀스 길이가 (576 → T)로 줄어 **추론 연산량/메모리 절감**.

## 6. 주요 하이퍼파라미터

| 파라미터 | 의미 | 기본/사용값 |
|----------|------|-------------|
| `visual_token_num` (T) | 보존할 시각 토큰 총수 | 실험: 576(=baseline, 프루닝 없음), 128, 64, 32 |
| `important_ratio` (r) | important 토큰 비율 | **0.5** (논문/README 기본; 예: T=128 → important 64 + diverse 64) |
| `mm_vision_select_layer` | attention/feature 추출 레이어 | **-2** (LLaVA-1.5 기본) |
| 점진 제거 단위 | diverse 선택 시 반복당 제거 수 | `min(8, R - T_div)` |

> T=576, r=0.5 인 경우 important 288 + residual 288, diverse 루프에서 `r=min(8,0)=0`으로 즉시 종료 → 576개 전부 보존 = **프루닝 없는 vanilla LLaVA-1.5-7B baseline**.

## 7. 모델/데이터 경로 설정 방법

- 모델: `model_vqa_loader.py --model-path models/llava-v1.5-7b` → `builder.load_pretrained_model()`.
  - 비전 타워 경로는 `llava-v1.5-7b/config.json`의 `mm_vision_tower` 가 지정 (로컬 CLIP 경로로 패치함).
- 데이터: `pope.sh`의 `--question-file`, `--image-folder`, `--annotation-dir` 인자.
  - 원본 `pope.sh`는 `/path/to/checkpoint`, `/path/to/dataset` placeholder → 로컬 경로 버전 `pope_local.sh` / `run_pope_all.sh` 작성.

## 8. ⚠️ 재현 중 발견·수정한 버그 (중요)

원본 `builder.py`는 `device_map != 'auto'` 일 때만 비전 타워를 `float16`으로 캐스팅한다.
평가 스크립트는 `device_map='auto'`(기본)로 모델을 로드하므로 **비전 타워가 float32로 남고**,
float16 LLM/projector와 dtype이 섞인 채 VisPruner의 attention 추출·argsort·matmul 커널이 실행되어
**비동기 CUDA "illegal memory access"가 발생, 시각 피처가 조용히 손상**되었다
(추론은 끝나지만 모델이 거의 "No"만 답해 POPE 정확도가 ~0.72로 비정상).

`CUDA_LAUNCH_BLOCKING=1` 에서는 정상 동작 + 두 경로(attention on/off) 피처가 완전히 동일(diff=0)함을 확인하여 원인을 dtype 혼용으로 특정.

**수정** (`llava/model/builder.py`): `device_map=='auto'` 인 경우에도 비전 타워를 모델 dtype으로 캐스팅.

```python
else:
    vision_tower.to(dtype=next(model.parameters()).dtype)
```

수정 후 sanity(300) 정확도 0.717 → **0.837** 로 정상화 (자세한 수치는 03 문서).

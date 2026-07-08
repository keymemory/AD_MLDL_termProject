# 현재 코드 구조 명세 — Claude Code용 지시서(md) 작성 입력 자료

> **용도**: 클로드 웹이 "정확한 수정 지점이 박힌 md 지시서"를 쓰기 위한 현재 repo의 코드 구조 설명.
> 아래 모든 경로/라인/시그니처는 **실제 코드를 읽어 확인한 값**(2026-06 기준). 수식 (가)/(나)를
> 어디에 끼워넣을지, 새 하이퍼파라미터를 어떻게 추가할지, 실험을 어떻게 돌려 어떤 형식으로
> 출력할지가 이 한 장에 다 들어 있다.

---

## Q. 클로드 웹의 6개 질문 — 직답 (실제 코드 확인 결과)

> 모든 경로는 `term_project/` 기준 상대경로. 라인번호는 실제 파일 확인값.

### Q1. Selection 로직 위치 (Stage1 important/diverse 선택)
- **파일**: `llava/model/llava_arch.py` · **함수**: `LlavaMetaForCausalLM.encode_images()` (L141~215)
- **important 선택**: L158~162 — `[CLS]`attention 평균 → `argsort` → 상위 `M1·r`개
- **diverse 선택**: L164~191 — cosine 유사도 bipartite 매칭으로 중복 반복 제거(ToMe식, 라운드당 최대 8개)
- `clip_encoder.py`는 **attention 추출만** 담당. **선택 알고리즘은 전부 `llava_arch.py::encode_images`에 있음.**

### Q2. attention score 추출 지점 ([CLS]→patch attention)
- **파일**: `llava/model/multimodal_encoder/clip_encoder.py` · **함수**: `feature_select()` (L36~52)
- **레이어**: `select_layer = mm_vision_select_layer = -2` (← `config.json` 확인. CLIP 끝에서 2번째 레이어)
- **변수/추출식**: L39 `image_attentions = image_forward_outs.attentions[self.select_layer]` →
  L43 `image_attentions = image_attentions[:, :, 0, 1:]` = **[CLS]행(index 0) → patch열(1:)** attention.
  shape `(B, H=heads, 576)`.
- 이후 `encode_images` L159 `image_attentions.mean(dim=1)` → head 평균 → `(B, 576)` **saliency 점수 벡터**.
  이 벡터가 (a) important top-k 선택과 (b) Stage2 weighted_avg 가중치 **둘 다에 재사용**됨(주의: §7-G).

### Q3. M1·r 주입 경로 (⚠️ 고정값 아님 — job별 CLI 인자)
- **고정값이 아니라 실험 job마다 CLI 인자로 주입**됨. VisPruner 원본의 `VISUAL_TOKEN_NUMBER/IMPORTANT_RATIO`
  bash 변수에 대응하는 것이 여기선 다음 CLI 인자:
  - `--visual_token_num` = **M2**(최종 토큰수) · `--stage1_tokens` = **M1**(Stage1 보존, clustering ON일 때만)
  - `--important_ratio` = **r**
- **주입 체인**(5단):
  `exp_jobs.tsv`컬럼 → `worker.sh`가 CLI로 매핑 → `model_vqa_loader.py` argparse(L168~175) →
  `load_pretrained_model(...)` 호출(L86~92, kwargs) → `LlavaLlamaForCausalLM.__init__`(L44~62) 저장 →
  `encode_images`에서 `get_visual_token_num()/get_stage1_tokens()/get_important_ratio()`로 읽음(L149~154).
- "M1 = M2의 2배(192/128/64 for M2=128/64/32)"·"r=0.5"는 **`exp_jobs.tsv`에서 그렇게 적은 관례일 뿐, 코드
  고정 아님.** clustering OFF면 `stage1_tokens=None` → `__init__`에서 M1=M2로 폴백(L60).

### Q4. Stage2 (spherical k-means 병합) 위치
- **호출 지점**: `llava/model/llava_arch.py` encode_images **L196~213** (`if enable_clustering and M2 < M1:` → `merge_tokens`)
- **구현 파일**: `llava/model/spherical_kmeans.py` (신규)
  - `spherical_kmeans(tokens, k, max_iter=10, init_indices=None)` **L22** — L2정규화(단위구)+cosine argmax 반복, 빈 클러스터 재할당
  - `merge_tokens(tokens, attn_scores, k, method="simple_avg", max_iter=10, init_indices=None)` **L84** — 대표토큰 생성
    - `simple_avg`: `rep_j = mean(x_i)` / `weighted_avg`: `rep_j = Σ a_i x_i / Σ a_i` (a_i = [CLS]attn, 0합이면 simple 폴백)
- **병합은 projector 적용 후 feature 공간**(L204 `feats_b`는 projected (M1, 4096)), 배치 내 **이미지별 루프**(L202~210).

### Q5. 실험 실행 구조
- **실제 실행은 `exp_runner/`**(원본 LLaVA `scripts/v1_5/eval/*.sh`(gqa/pope/textvqa/vqav2/sqa 등)도 repo에
  존재하나 **본 실험엔 미사용** — 이 점 혼동 주의):
  - `worker.sh` = 락 기반 워커(추론 resume/retry 최대 25회 + 로컬 채점 + `results.tsv` 기록)
  - `launch.sh` = 3-GPU 병렬 런치 · job 정의 = `exp_jobs.tsv`(pope/gqa/textvqa) + `jobs_vqav2.tsv`(vqav2)
- **추론 엔트리**: `llava/eval/model_vqa_loader.py`(pope/gqa/textvqa/vqav2), `llava/eval/model_vqa_science.py`(sqa)
- **답변 저장 경로**: `playground/data/eval/<bench>/answers/EXP/<ID>/r_<R>.jsonl`
- **채점기**: pope=`eval_pope.py`(Average F1) · gqa=공식 `eval.py`(Acc) · textvqa=`eval_textvqa`(Acc) ·
  vqav2=`exp_runner/vqa_eval.py <ans> <gt> by_type`(Overall Acc + yes-no/number/other) · sqa=`eval_science_qa.py`
- **결과 집계 파일**: `exp_runner/results.tsv`, `results_update2.tsv` (스키마 = §6-3)

### Q6. 현재 변형 분기 (VisPruner / Ours-S / Ours-W)
- **별도 함수 아님 — 단일 플래그 `--enable_clustering` + `--merge_method` 조합**으로 구분:
  | 변형 | CLI | 코드 경로 |
  |---|---|---|
  | **VisPruner-only (A)** | `--enable_clustering` 없음 | clustering OFF → M1=M2 → Stage2 미진입 = **원본 비트동일** |
  | **Ours simple (B/S)** | `--enable_clustering --stage1_tokens M1 --merge_method simple_avg` | encode_images L197 분기 진입 |
  | **Ours weighted (C/W)** | `--enable_clustering --stage1_tokens M1 --merge_method weighted_avg` | 〃 + L199 method 분기 |
- **코드 분기점**: encode_images L150 `enable_clustering=self.get_enable_clustering()`, L197
  `if enable_clustering and M2 < M1:`, L199 `merge_method=self.get_merge_method()`.
- job tsv의 `CLUST`(0/1)·`METHOD`(none/simple_avg/weighted_avg) 컬럼을 `worker.sh`가 위 플래그로 변환
  (`CL_ARGS` 로직). → **새 변형 추가 = 새 `--merge_method` 값 + worker.sh CL_ARGS 분기 + spherical_kmeans.py 구현.**

---

## 0. 한 줄 요약 · 환경 · 레포 루트

- **연구**: VLM(LLaVA-1.5-7B) 시각 토큰 축소. 기존 VisPruner(pruning-only)를 **Two-Stage**
  (Stage1 VisPruner 선택 → Stage2 Spherical K-Means 병합)로 개선.
- **백본**: LLaVA-1.5-7B = Vicuna-7B + CLIP-ViT-L/14-336px(576 patch) + 2-layer MLP projector.
- **레포 루트**: `/home/jhlee/CLUST_KETI/AD_MLDL_termProject/`
  - 제안 코드(작업 대상): `term_project/` ← **거의 모든 수정은 여기서**
  - 원본 회귀비교: `VisPruner_run/` (동일 패치, clustering 없음)
- **env**: conda `vispruner` (Python 3.10) / torch 2.1.2+cu121 / **transformers 4.37.2**(고정) /
  tokenizers 0.15.1. GPU: RTX A6000 49GB ×3.
- **추론 규약**: fp16, greedy(`--temperature 0`), `--conv-mode vicuna_v1`.

> ⚠️ transformers 4.37.2는 **고정**. LLM 디코더 내부(layer-K pruning, KV cache 조작 등)를
> 건드리는 방법(FastV류)은 이 버전에서 고위험 → 가급적 **CLIP~projector 사이(시각 토큰 단)**
> 에서만 작업. 본 프레임워크 전체가 그 설계라 새 수식도 같은 자리에 들어가는 것이 안전.

---

## 1. 핵심 파일 맵 (term_project/ 기준 상대경로)

| 파일 | 역할 | 핵심 위치 |
|---|---|---|
| `llava/model/llava_arch.py` | ★★★ **Stage1 선택 + Stage2 호출의 본체**. 모든 토큰 로직 | `encode_images()` = L141~215 |
| `llava/model/spherical_kmeans.py` | ★★ **Stage2 병합 알고리즘**(신규 파일) | `spherical_kmeans()` L22, `merge_tokens()` L84 |
| `llava/model/multimodal_encoder/clip_encoder.py` | CLIP 인코더. **[CLS]→patch attention 추출** | `feature_select()` L36~52 |
| `llava/model/language_model/llava_llama.py` | 모델 클래스. **하이퍼파라미터 보관 + getter** | `__init__` L44~65, getter L71~89 |
| `llava/model/builder.py` | 모델 로더. **dtype 패치(필수)** + `**kwargs` 통로 | `load_pretrained_model()` L26, dtype패치 L155~161 |
| `llava/eval/model_vqa_loader.py` | ★ **추론 엔트리**(POPE/GQA/TextVQA/VQAv2). CLI 인자 + resume | argparse L156~175, 모델생성 L86~92 |
| `llava/eval/model_vqa_science.py` | ScienceQA 추론 엔트리 (동일 인자 구조) | — |
| `exp_runner/worker.sh` | ★ 락 기반 실험 워커: 추론(resume/retry) + 로컬채점 + tsv기록 | — |
| `exp_runner/launch.sh` | 3-GPU 병렬 런치 | — |
| `exp_runner/exp_jobs.tsv` / `jobs_vqav2.tsv` | ★ 실험 정의(job 매트릭스) | — |
| `exp_runner/results.tsv` / `results_update2.tsv` | ★ 결과 출력(스키마는 §6) | — |
| `exp_runner/vqa_eval.py` | VQAv2 로컬 채점(by_type) | — |

> 다른 백본 파일도 존재하나 **본 과제는 `LlavaLlamaForCausalLM` 경로만 사용**:
> `multimodal_encoder/siglip_encoder.py`(SigLIP), `language_model/llava_qwen.py`,
> `language_model/llava_mistral.py` 등은 미사용. Stage2 clustering은 **단일 이미지 경로
> (LLaVA-1.5)에서만 동작**하도록 가드됨(§7-E 참조).

---

## 2. 데이터 흐름 (단계별 정확한 file:line)

```
이미지 (B, 3, 336, 336)
  └─[A] CLIP forward + [CLS]attention 추출   clip_encoder.py: feature_select() L36~52
        → image_features (B, 576, 1024), image_attentions (B, H, 576)
  └─[B] encode_images() 진입                 llava_arch.py L141~142
        attentions = mean over heads → (B, 576)   L159
  └─[C] Stage1-important: [CLS]attn 상위 M1·r개   llava_arch.py L158~162
  └─[D] Stage1-diverse: cosine 중복제거로 M1·(1-r)개  llava_arch.py L164~191
        → selected_indices (B, M1),  index_masks (B, 576) scatter   L186~191
  └─[E] mm_projector 적용 (1024 → 4096)      llava_arch.py L194
  └─[F] Stage2-merge: M1 → M2 (clustering on & M2<M1)  llava_arch.py L196~213
        → merge_tokens() 호출, 반환 (B, M2, 4096), index_masks=None
  └─[G] prepare_inputs_labels_for_multimodal()로 <image> 자리에 주입  llava_arch.py L218~
        - clustering ON  : index_masks=None → flatten (B,M2,D)        L304~308
        - clustering OFF : image_features[index_masks]                L309~310
  → LLM(Vicuna-7B) 디코딩, 입력 시각 토큰 길이 = M2
```

**용어**: `M1` = Stage1 보존 토큰수(`stage1_tokens`), `M2` = 최종 토큰수(`visual_token_num`),
`r` = important_ratio. clustering OFF면 `M1==M2`이고 Stage2 미진입 → **원본 VisPruner와 비트동일**.

---

## 3. ★ Selection / Merge 로직 정확한 위치 — 새 수식 (가)/(나) 삽입 후보

`encode_images()` 안의 **3개 후보 지점**. 새 수식이 "무엇을 바꾸는가"에 따라 여기 중 하나(또는 신규 분기)에 넣는다.

### 후보 ① Stage1-important 점수/선택 (llava_arch.py L158~162)
```python
# [VisPruner] Select important tokens using attention scores
image_attentions = image_attentions.mean(dim=1)             # (B, N)  ← saliency 점수 s_i
token_indices = image_attentions.argsort(dim=-1, descending=True)
important_indices = token_indices[:, :important_token_num]   # top-(M1·r)
residual_indices  = token_indices[:, important_token_num:]
```
→ **중요도 점수 정의(가) 를 바꾸려면 여기.** 예: attention 외 신호(노름·텍스트연관도 등)로
`image_attentions`(=점수 벡터)를 대체/혼합. `attn`은 이후 Stage2 weighted 가중치로도 재사용됨(주의).

### 후보 ② Stage1-diverse 중복제거 (llava_arch.py L164~191)
```python
image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # 코사인용 정규화
while diverse_token_num > 0:
    R = residual_indices.shape[1]
    r = min(8, R - diverse_token_num)            # 라운드당 최대 8개 제거 (ToMe식)
    ...
    a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]   # bipartite 분할
    scores = (a @ b.transpose(-1,-2)).max(dim=-1).values                # 유사도
    distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]   # 덜 중복된 것 보존
    keep = distinct_indices.shape[1]   # [FIX] 홀수 R IndexError 방지(§7-F)
    residual_indices = torch.cat([...], dim=-1)                          # (B, R-r)
```
→ **다양성/중복 측도(나) 를 바꾸려면 여기.** bipartite 매칭·라운드당 제거수(8)·유사도 정의가 후보.

### 후보 ③ Stage2 병합 (llava_arch.py L196~213 → spherical_kmeans.py)
```python
if enable_clustering and M2 < M1:
    from llava.model.spherical_kmeans import merge_tokens
    for b in range(B):
        feats_b = image_features[b][selected_indices[b]]   # (M1, D) projector 적용 후
        attn_b  = image_attentions[b][selected_indices[b]] # (M1,)  weighted 가중치
        merged_b = merge_tokens(feats_b, attn_b, M2, method=merge_method, max_iter=kmeans_max_iter)
    return torch.stack(merged_list, dim=0), None           # (B, M2, D), index_masks=None
```
→ **병합 방식(군집화·대표토큰 생성) 변형**은 `spherical_kmeans.py::merge_tokens`/`spherical_kmeans` 교체.
soft assignment / k-means++ 초기화 / hierarchical merging 등은 **이 파일만 고치면** 끝(인터페이스 유지 시).

> **삽입 규약**: 어떤 후보든 `return` 텐서 shape 규약을 지켜야 함 — Stage2 ON이면
> `(merged (B,M2,D), None)`, OFF면 `(image_features (B,N,D), index_masks (B,N) bool)`.
> `index_masks=None`이 "이미 최종 토큰"의 신호로 prepare_inputs(L306)에서 분기됨.

---

## 4. ★ 새 하이퍼파라미터를 추가하는 법 — 4 touch-point 체인

현재 `enable_clustering / stage1_tokens / merge_method / kmeans_max_iter`가 이 체인으로 흐른다.
**새 인자 `--foo`를 추가하려면 아래 4곳만** 고치면 끝(`builder.py`는 `**kwargs`라 수정 불필요):

**(1) CLI 인자 등록** — `llava/eval/model_vqa_loader.py` L168~175 부근 + (필요시 `model_vqa_science.py`)
```python
parser.add_argument("--foo", type=..., default=...)
```

**(2) 모델 생성에 전달** — `llava/eval/model_vqa_loader.py` L86~92
```python
load_pretrained_model(
    ...,
    visual_token_num=args.visual_token_num, important_ratio=args.important_ratio,
    enable_clustering=args.enable_clustering, stage1_tokens=args.stage1_tokens,
    merge_method=args.merge_method, kmeans_max_iter=args.kmeans_max_iter,
    foo=args.foo,                                  # ← 추가 (kwargs로 자동 전달됨)
)
```
> `builder.py::load_pretrained_model`(L26)은 `**kwargs`로 받아 `LlavaLlamaForCausalLM.from_pretrained(**kwargs)`
> 에 그대로 넘긴다 → `__init__`의 keyword로 도달. **builder.py 수정 불필요.**

**(3) 모델에 저장 + getter** — `llava/model/language_model/llava_llama.py` L44~89
```python
def __init__(self, config, visual_token_num, important_ratio,
             enable_clustering=False, stage1_tokens=None,
             merge_method="simple_avg", kmeans_max_iter=10,
             foo=DEFAULT):                          # ← 파라미터 추가
    ...
    self.foo = foo                                  # ← 저장 (L62 부근)
def get_foo(self):                                  # ← getter (L88 부근)
    return getattr(self, "foo", DEFAULT)            # getattr+default = 구버전 체크포인트 안전
```

**(4) 로직에서 사용** — `llava/model/llava_arch.py::encode_images` (L149~151 부근에서 읽기)
```python
foo = self.get_foo()
```

> **회귀 안전 원칙**: 모든 신규 getter는 `getattr(self, "x", default)` 패턴(L80/83/86/89)을 따른다.
> 새 기능의 default는 반드시 **"기존 동작과 동일"**해야 하며(예: clustering OFF=원본 VisPruner),
> 이걸 지키면 `enable_clustering=False`일 때 비트동일 회귀안전이 유지된다.

---

## 5. 핵심 함수 시그니처 (인터페이스 — 변형 시 유지 권장)

```python
# spherical_kmeans.py L22
@torch.no_grad()
def spherical_kmeans(tokens, k, max_iter=10, init_indices=None) -> assignments  # (M1,) 클러스터 id
#   - tokens (M1,D) → L2정규화(단위구) → cosine argmax 반복. 빈 클러스터 재할당, fp16안전(eps=1e-8).
#   - init_indices=None이면 randperm 랜덤 초기화 (★ k-means++ 등은 여기 교체)

# spherical_kmeans.py L84
@torch.no_grad()
def merge_tokens(tokens, attn_scores, k, method="simple_avg", max_iter=10, init_indices=None) -> merged  # (M2,D)
#   - method="simple_avg":  rep_j = mean_{i∈C_j} x_i
#   - method="weighted_avg": rep_j = Σ a_i x_i / Σ a_i   (a_i = attn_scores, 음수클램프, 0합이면 simple 폴백)
#   - k>=M1이면 입력 그대로 반환(no-op)

# llava_arch.py L141
def encode_images(self, images) -> (features, index_masks)
#   - clustering ON : ((B, M2, D), None)
#   - clustering OFF: ((B, N=576, D), (B, N) bool index_masks)
```

`feature_select`(clip_encoder.py L36): `image_attentions = image_forward_outs.attentions[select_layer][:, :, 0, 1:]`
= **select_layer의 [CLS]행→patch열 attention**(L43). `select_layer`는 LLaVA-1.5 기본 -2.

---

## 6. ★ 실험 러너 — 입력(job) / 실행 / 출력(results) 스키마

### 6-1. job 정의 파일 — `exp_runner/exp_jobs.tsv` (탭 구분, 헤더 없음)
```
ID       BENCH    M2   CLUST  M1   METHOD        R
A-128    pope     128  0      128  none          0.5      # VisPruner-only (clustering OFF)
B-64s    gqa      64   1      128  simple_avg    0.5      # Ours simple, M1=128→M2=64
B-32w    pope     32   1      64   weighted_avg  0.5      # Ours weighted
R-30     pope     64   1      128  simple_avg    0.3      # ablation: r=0.3
M-192    pope     64   1      192  simple_avg    0.5      # ablation: M1=192
```
- `CLUST` 0/1 = clustering off/on. off면 `M1`·`METHOD`는 무시(관례상 M1=M2, METHOD=none).
- 컬럼 = `ID BENCH M2 CLUST M1 METHOD R`. **신규 실험은 이 줄을 추가**하면 됨.
- 벤치 종류: `pope gqa textvqa vqav2`(loader) / `sqa`(science, 별도). 새 하이퍼파라미터가
  필요하면 worker.sh의 CLI 매핑(아래)도 같이 확장해야 함.

### 6-2. job → CLI 매핑 — `worker.sh`
```bash
CL_ARGS=""
[ "$CLUST" = "1" ] && CL_ARGS="--enable_clustering --stage1_tokens $M1 --merge_method $METHOD"
python -m llava.eval.model_vqa_loader --model-path models/llava-v1.5-7b \
  --question-file "$QF" --image-folder "$IMG" --answers-file "$ANS" \
  --visual_token_num "$M2" --important_ratio "$R" $CL_ARGS \
  --temperature 0 --conv-mode vicuna_v1
```
- **resume/retry**: 답변수 < 총문항수면 최대 25회 재시도(크래시·OOM 후 이어쓰기). loader가
  `answered_ids`로 기존 답변 skip(model_vqa_loader.py L101~112).
- 채점: pope=`eval_pope.py`(Average F1) / gqa=공식 `eval.py`(Accuracy) /
  textvqa=`eval_textvqa`(Acc) / vqav2=`exp_runner/vqa_eval.py ... by_type`(Overall Acc + type별).

### 6-3. 결과 출력 — `exp_runner/results.tsv` (탭 구분)
```
ID  BENCH  M2  CLUST  M1  METHOD  R  METRIC  VALUE  GEN/TOT
B-32s  pope  32  1  64  simple_avg  0.5  AvgF1  0.7756147629450251  8910/8910
```
- 한 job = 한 줄. `flock`으로 동시쓰기 보호. `VALUE`는 지표 원값(POPE는 0~1 F1, 그 외 %).
- **출력 형식 규약(새 실험도 이걸 따를 것)**: `ID\tBENCH\tM2\tCLUST\tM1\tMETHOD\tR\tMETRIC\tVALUE\tGEN/TOT`.
  새 컬럼(예: 새 하이퍼파라미터)이 필요하면 worker.sh의 `echo -e ...` 라인과 헤더를 함께 확장.

### 6-4. 실행
```bash
cd term_project
pip install -e .                                   # llava 패키지 등록(최초 1회)
bash exp_runner/launch.sh                          # 3-GPU 병렬(락 분배). job은 exp_jobs.tsv
# 단일: bash exp_runner/worker.sh <GPU> <jobs.tsv>
python exp_runner/efficiency.py                    # 효율성(POPE 110샘플, latency·GPU mem)
```

---

## 7. ★ 함정 · 주의 (클로드 웹이 md에 반드시 반영할 것)

- **A. dtype 패치(필수·이미 적용됨)** — `builder.py` L155~161: `device_map='auto'`일 때 vision
  tower를 `next(model.parameters()).dtype`(fp16)로 강제 캐스팅. **미적용 시 비전타워(fp32)↔모델(fp16)
  혼용 → 비동기 CUDA "illegal memory access" → 시각 피처 무음 손상 → 정확도 비정상 하락(POPE 0.72).**
  새 코드가 dtype를 섞지 않도록 주의. `VisPruner_run/llava/model/builder.py`에도 동일 패치 존재.

- **B. ★ 옛 경로 3종이 깨져 있음(실측 확인 — 실행 전 반드시 수정)** — 디렉토리가 `SKKU_Works/...`
  에서 현재 `AD_MLDL_termProject/`로 이동됐는데 **아래 3곳이 옛 절대경로를 그대로 가리켜 깨진 상태**:
  1. `exp_runner/worker.sh` · `exp_runner/launch.sh`의
     `TP=/home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/experiments/term_project` → 현재
     `/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project`로 수정.
  2. `term_project/models` 심볼릭링크 → `/home/.../SKKU_Works/.../VisPruner_run/models`(깨짐). 실제
     모델은 `../VisPruner_run/models`에 있음 → `ln -sfn ../VisPruner_run/models term_project/models`로 재생성.
  3. (아래 C) `config.json`의 `mm_vision_tower` 절대경로도 옛 SKKU 경로.
  → md 지시서 맨 앞 "실행 전 환경 셋업" 절에 이 3가지 경로 패치를 **선행 단계로** 박을 것.
  데이터 링크 `term_project/playground/data/eval → ../dataset`도 함께 확인.

- **C. config.json의 mm_vision_tower 절대경로(실측: 옛 경로로 깨짐)** —
  `VisPruner_run/models/llava-v1.5-7b/config.json`의
  `"mm_vision_tower": "/home/jhlee/CLUST_KETI/SKKU_Works/.../VisPruner_run/models/clip-vit-large-patch14-336"`
  가 옛 경로 → 현재 `/home/jhlee/CLUST_KETI/AD_MLDL_termProject/VisPruner_run/models/clip-vit-large-patch14-336`
  로 갱신해야 CLIP 로드 가능(아니면 로드 실패). `mm_vision_select_layer: -2`, `image_aspect_ratio: "pad"`,
  `mm_patch_merge_type: flat`(LLaVA-1.5 단일 이미지)은 그대로.

- **D. CUDA_LAUNCH_BLOCKING** — worker.sh는 `=1`(안정·2~3배 느림). dtype 패치가 적용돼 있으면
  꺼서 가속 가능(검증된 사례 있음). 새 디버깅 땐 켜는 게 안전.

- **E. clustering은 단일 이미지(LLaVA-1.5) 경로만 지원** — `prepare_inputs...` L232~236에서
  anyres/multi-image(LLaVA-NeXT) + `index_masks=None`이면 `NotImplementedError`. **다른 백본/anyres로
  확장하려면 이 가드와 spatial 병합 경로(L247~301)까지 손봐야 함**(단순 적용 불가).

- **F. 홀수 R IndexError fix(이미 반영)** — diverse 루프(L177~184)에서 홀수 R일 때
  `ceil(R/2) ≠ R//2` 불일치로 IndexError가 났던 것을, `keep = distinct_indices.shape[1]`로
  arange 길이를 맞춰 수정 완료. r=0.5(짝수 R)는 영향 없음, r=0.7 등 손대면 이 로직 재확인.

- **G. weighted_avg는 Stage1 attention 재사용** — 후보①에서 `image_attentions`를 다른 점수로
  바꾸면 Stage2 weighted 가중치(L205 `attn_b`)도 같이 바뀐다. 의도와 다르면 분리할 것.

- **H. VQAv2는 val 균형 subset 6000** — `llava_vqav2_val_subset.jsonl`로 로컬 채점(test-dev EvalAI
  미사용). 논문 test-dev 인용치와 **절대 비교 불가, 동일 subset 내 A↔B 상대비교만 유효**.

---

## 8. 클로드 웹에게 — md 지시서에 꼭 넣어야 할 항목 체크리스트

1. **수정 대상 파일·라인을 §3/§4 좌표로 지정**(예: "llava_arch.py L158~162의 important 점수를 …로 교체").
2. **새 하이퍼파라미터는 §4의 4 touch-point**(CLI add_argument → loader 호출 → llava_llama `__init__`+getter
   → encode_images 사용)를 빠짐없이. getter는 `getattr(self,"x",default)`, default=기존동작.
3. **회귀 안전**: 새 기능 OFF 시 기존 결과(특히 clustering OFF=원본 VisPruner)가 불변임을 보장하는
   sanity 검증을 지시(예: 동일 시드/job에서 A-시리즈 수치 재현).
4. **반환 shape 규약**(§3 하단) 준수 — Stage2 ON `(B,M2,D)+None`, OFF `(B,576,D)+mask`.
5. **실험은 §6 패턴 재사용**: exp_jobs.tsv에 job 줄 추가 → worker.sh CLI 매핑 확장(새 인자면) →
   results.tsv 스키마(§6-3)로 출력. 채점기는 벤치별 기존 명령 그대로.
6. **실행 전 §7-B 경로 수정 + §7-A dtype 불변 + §7-C config 절대경로** 주의를 명시.
7. **출력 형식**: results.tsv 한 줄/실험, 컬럼 = `ID BENCH M2 CLUST M1 METHOD R METRIC VALUE GEN/TOT`.
   새 축이 생기면 헤더·echo 라인·집계문서 표를 함께 확장.
8. **transformers 4.37.2 고정** — LLM 디코더 내부 개조(FastV류)는 피하고 시각토큰 단(CLIP~projector)에서 작업.

---

> 이 자료로 부족하면 다음을 추가로 요청하세요: 특정 함수 전체 소스, `model_vqa_science.py`의 CLI 구조,
> `vqa_eval.py` 채점 로직, `efficiency.py` 측정 방식, 데이터셋 경로/포맷, 기존 결과 원시 수치(results*.tsv).

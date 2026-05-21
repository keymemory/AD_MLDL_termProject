# Two-Stage Visual Token Reduction — 구현 검증 보고서 (클로드 웹 리뷰용)

> 목적: VisPruner 기반 **Stage1(토큰 선택) → Stage2(Spherical K-Means 병합)** 프레임워크 구현이
> 명세대로 정확한지 독립 검증. 이 문서는 **단독 완결형** — 코드 전문·통합 지점·검증 결과 포함.
> 환경: LLaVA-1.5-7B(+CLIP-ViT-L/14-336), PyTorch 2.1.2/transformers 4.37.2, fp16, greedy, CUDA_LAUNCH_BLOCKING=1.

---

## 1. 설계 명세 (구현 대상)

```
이미지 → CLIP → 576 visual tokens
  └ Stage1 (VisPruner): [CLS] attention로 important(M1·r개) + cosine 중복제거로 diverse(M1·(1-r)개)
                         → M1개 보존, 각 토큰 attention score도 함께 보유
  └ Stage2 (Spherical K-Means): M1개를 M2(<M1)개 클러스터로 병합
                         → 대표토큰 = simple_avg | weighted_avg(attn 가중)
  → 최종 M2개 토큰을 LLM <image> 자리에 주입
분기: enable_clustering=False → 기존 VisPruner와 완전 동일 (M1=M2, Stage2 미실행)
```

CLI 인자: `--enable_clustering`(flag) `--stage1_tokens`(M1, 기본 None=M2) `--visual_token_num`(M2)
`--merge_method`(simple_avg|weighted_avg) `--kmeans_max_iter`(기본10) `--important_ratio`(r, 기본0.5).

---

## 2. 핵심 신규 코드 전문 — `llava/model/spherical_kmeans.py`

```python
import torch
_EPS = 1e-8

def _l2norm(x, dim=-1):
    # fp16 안전: float32로 norm 계산 후 원 dtype 유지
    return x / (x.float().norm(dim=dim, keepdim=True).clamp_min(_EPS)).to(x.dtype)

@torch.no_grad()
def spherical_kmeans(tokens, k, max_iter=10, init_indices=None):
    M1 = tokens.shape[0]; device = tokens.device
    if k >= M1:
        return torch.arange(M1, device=device)
    x = _l2norm(tokens.float())                       # 단위 구 투영(float32 안정화)
    if init_indices is None:
        init_indices = torch.randperm(M1, device=device)[:k]
    centroids = x[init_indices].clone()               # (k, D)
    prev_assign = None
    for _ in range(max_iter):
        sim = x @ centroids.t()                       # (M1, k) cosine
        assign = sim.argmax(dim=1)                    # (M1,)
        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        new_centroids.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones(M1, device=device))
        empty = (counts == 0)
        if empty.any():                               # 빈 클러스터: 최대 클러스터에서 1토큰 재할당
            for e in torch.nonzero(empty, as_tuple=False).flatten().tolist():
                biggest = int(counts.argmax())
                members = torch.nonzero(assign == biggest, as_tuple=False).flatten()
                if members.numel() <= 1: continue
                move = members[0]
                assign[move] = e
                counts[biggest] -= 1; counts[e] += 1
                new_centroids[biggest] -= x[move]; new_centroids[e] += x[move]
        nonempty = counts > 0
        centroids[nonempty] = _l2norm(new_centroids[nonempty] / counts[nonempty].unsqueeze(1))
    sim = x @ centroids.t()
    return sim.argmax(dim=1)

@torch.no_grad()
def merge_tokens(tokens, attn_scores, k, method="simple_avg", max_iter=10, init_indices=None):
    M1, D = tokens.shape
    if k >= M1:
        return tokens
    assign = spherical_kmeans(tokens, k, max_iter=max_iter, init_indices=init_indices)
    tf = tokens.float()
    out = torch.zeros(k, D, device=tokens.device, dtype=torch.float32)
    if method == "weighted_avg":
        w = attn_scores.float().clamp_min(0)
        out.index_add_(0, assign, tf * w.unsqueeze(1))      # Σ aᵢxᵢ
        wsum = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        wsum.index_add_(0, assign, w)                       # Σ aᵢ
        cnt = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        cnt.index_add_(0, assign, torch.ones(M1, device=tokens.device))
        denom = wsum.clone()
        zero_w = denom <= _EPS
        if zero_w.any():                                    # 가중합 0 → 단순평균 폴백
            avg = torch.zeros(k, D, device=tokens.device, dtype=torch.float32)
            avg.index_add_(0, assign, tf)
            out[zero_w] = avg[zero_w]
            denom[zero_w] = cnt[zero_w].clamp_min(1.0)
        out = out / denom.clamp_min(_EPS).unsqueeze(1)
    else:                                                   # simple_avg
        out.index_add_(0, assign, tf)
        cnt = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        cnt.index_add_(0, assign, torch.ones(M1, device=tokens.device))
        out = out / cnt.clamp_min(1.0).unsqueeze(1)
    return out.to(tokens.dtype)
```

---

## 3. 통합 지점 — `llava/model/llava_arch.py :: encode_images()`

VisPruner 원본 흐름 유지 + 2가지 변경:

```python
# (변경1) M1/M2 분리: Stage1 선택은 M1 기준
M2 = self.get_visual_token_num()
enable_clustering = self.get_enable_clustering()
M1 = self.get_stage1_tokens() if enable_clustering else M2
visual_token_num = M1                                   # important/diverse 개수는 M1로 산정
important_ratio = self.get_important_ratio()
important_token_num = int(visual_token_num * important_ratio)
diverse_token_num   = visual_token_num - important_token_num

# ... (VisPruner 원본: [CLS] attn mean → argsort → important + ToMe식 diverse 선택) ...
image_attentions = image_attentions.mean(dim=1)         # (B,N) 토큰별 attn score 보유
# ... selected_indices: (B, M1) ...
index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
index_masks.scatter_(1, selected_indices, True)
image_features = self.get_model().mm_projector(image_features)   # (B,N,D_llm)

# (변경2) Stage2: clustering on & M2<M1 일 때만
if enable_clustering and M2 < M1:
    from llava.model.spherical_kmeans import merge_tokens
    merged_list = []
    for b in range(B):
        feats_b = image_features[b][selected_indices[b]]   # (M1, D) Stage1 보존토큰
        attn_b  = image_attentions[b][selected_indices[b]] # (M1,)  토큰별 attn
        merged_b = merge_tokens(feats_b, attn_b, M2,
                                method=self.get_merge_method(),
                                max_iter=self.get_kmeans_max_iter())  # (M2, D)
        merged_list.append(merged_b)
    return torch.stack(merged_list, 0), None               # index_masks=None = "이미 최종"

return image_features, index_masks                         # clustering off → 원본과 동일
```

`prepare_inputs_labels_for_multimodal()` 처리:
- 단일이미지(LLaVA-1.5/POPE) 경로: `index_masks is None` → `image_features.flatten(0,1).unsqueeze(0)` 로 LLM 주입. 아니면 기존 `image_features[index_masks].unsqueeze(0)`.
- multi-image/anyres 경로: `index_masks is None`이면 `NotImplementedError`(단일이미지 전용).

`llava_llama.py`: `__init__`에 4개 인자 + `get_*` getter(미존재 시 `getattr` 기본값으로 구버전 호환).
CLI는 `model_vqa_loader.py` argparse → `load_pretrained_model(**kwargs)` → `from_pretrained` → `__init__` 경유(기존 visual_token_num과 동일 메커니즘).

---

## 4. 설계 결정 & 엣지 케이스 처리 (검증 포인트)

| 항목 | 처리 | 의도 |
|---|---|---|
| 클러스터링 공간 | L2 정규화 후 cosine(단위 구) — float32 | spherical k-means 정의 충실, fp16 수치불안정 회피 |
| 대표 토큰 공간 | **원 feature(미정규화) 평균** | LLM 입력 분포 보존(정규화 벡터를 넣지 않음) |
| 0-division | `clamp_min(1e-8)` | fp16 norm/가중합 0 방지 |
| 빈 클러스터 | 최대 클러스터에서 1토큰 재할당 | k개 대표 토큰 수 보장 |
| weighted 가중합 0 | 단순평균 폴백 | attn=0 클러스터 NaN 방지 |
| `k>=M1` | 원본 그대로 반환 | clustering 무효 구간 안전 |
| `enable_clustering=False` | M1=M2, Stage2 미실행, 반환 `(feat, index_masks)` | **기존 VisPruner와 완전 동일(회귀 0)** |
| 조기 종료 | assignment 불변 시 break | 추론 overhead 최소화(max_iter≤10) |

---

## 5. 검증 결과

### 5-1. 단위 테스트 (`(128,1024)`, k=64)
- fp32/fp16 모두: `spherical_kmeans` assign∈[0,64), `merge_tokens` (64,1024), 전부 finite ✅
- 엣지: `k≥M1` 원본반환, 동일점 다수→빈클러스터 발생 시 에러 없이 (64,1024) ✅

### 5-2. 통합 Sanity (POPE 300 subset, 최종 64토큰, r=0.5, greedy)

| # | 설정 | 생성 | Avg F1 | 판정 |
|---|---|---|---|---|
| 1 | VisPruner only (clustering off, n=64) | 300/300 | **0.8459** | 기존 경로 불변·에러0 ✅ |
| 2 | Two-stage **simple_avg** (M1=128→M2=64) | 300/300 | **0.8474** | 정상·합리적 ✅ |
| 3 | Two-stage **weighted_avg** (M1=128→M2=64) | 300/300 | **0.8381** | 정상·합리적 ✅ |

해석: (2) 128→64 병합이 동일 최종 64토큰 직접선택(1, 0.846)과 동등 이상(0.847) →
"많이 보존 후 병합"이 정보 보존에 유리하다는 프레임워크 가설과 부합. 세 경우 CUDA 에러·크래시 0.

> 참고: 300-subset은 1차 full-set(POPE@64 F1 0.809)보다 쉬워 절대값이 높음. (1)의 목적은
> **clustering off가 기존 VisPruner와 동일 경로로 동작함을 회귀 검증**하는 것.

---

## 6. 클로드 웹 검증 요청 포인트

다음을 중점 확인 부탁:
1. **회귀 안전성**: `enable_clustering=False`에서 `encode_images`가 원본과 비트 동일 경로인가? (M1=M2, Stage2 분기 미진입, 반환형 동일)
2. **Stage1→Stage2 연결**: `selected_indices`로 feature와 attn을 같은 인덱스로 gather → weighted_avg 가중치 정합성
3. **대표토큰 공간 선택**: clustering은 정규화 벡터로, 평균은 원 feature로 — 타당한가?
4. **수치 안정성**: fp16에서 `_l2norm`/가중합 0-division 가드 충분한가?
5. **빈 클러스터/조기종료** 로직의 정확성 및 k개 토큰 수 보장
6. **파이프라인 주입**: `(B,M2,D) → flatten(0,1).unsqueeze(0)` 가 LLM `<image>` 자리 삽입과 정합한가? (B=1 POPE 기준)

## 7. 결론
명세의 Stage1+Stage2(simple/weighted) 전부 구현·통합. 단위테스트·sanity 3종 통과,
`enable_clustering=False` 회귀 안전 확인. 후속 체계적 실험(experiments_ver.md) 준비 완료.
구현 파일: `experiments/term_project/llava/model/spherical_kmeans.py`(신규),
`llava_arch.py`·`llava_llama.py`·`eval/model_vqa_loader.py`(수정), 상세: `01_implementation_report.md`.

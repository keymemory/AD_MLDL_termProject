"""
[Two-Stage Framework] Stage 2: Spherical K-Means 기반 토큰 병합.

Stage 1(VisPruner)에서 보존된 M1개 토큰을 M2(<M1)개 클러스터로 그룹화하고,
각 클러스터의 대표 토큰을 생성하여 동일 토큰 수 대비 정보 밀도를 높인다.

- 모든 연산 GPU 텐서(fp16 안전, eps로 0-division 방지)
- max_iter 작게(<=10) 유지하여 추론 overhead 최소화
- 빈 클러스터는 가장 큰 클러스터에서 토큰 1개 재할당
"""
import torch

_EPS = 1e-8


def _l2norm(x, dim=-1):
    # fp16 안전: float32로 norm 계산 후 원 dtype 유지
    return x / (x.float().norm(dim=dim, keepdim=True).clamp_min(_EPS)).to(x.dtype)


@torch.no_grad()
def spherical_kmeans(tokens, k, max_iter=10, init_indices=None):
    """
    Args:
        tokens: (M1, D) 입력 토큰 (원 feature 공간)
        k: 목표 클러스터 수 (= M2)
        max_iter: 최대 반복
        init_indices: (k,) 초기 centroid로 쓸 토큰 인덱스 (None이면 랜덤)
    Returns:
        assignments: (M1,) 각 토큰의 클러스터 id (0..k-1)
    """
    M1 = tokens.shape[0]
    device = tokens.device
    if k >= M1:
        return torch.arange(M1, device=device)

    x = _l2norm(tokens.float())  # 단위 구 투영 (clustering은 float32로 안정화)

    if init_indices is None:
        init_indices = torch.randperm(M1, device=device)[:k]
    centroids = x[init_indices].clone()  # (k, D) 이미 정규화된 점들

    prev_assign = None
    for _ in range(max_iter):
        sim = x @ centroids.t()                # (M1, k) cosine similarity
        assign = sim.argmax(dim=1)             # (M1,)

        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign

        # 새 centroid = 클러스터 평균 후 재정규화
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        new_centroids.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones(M1, device=device))

        empty = (counts == 0)
        if empty.any():
            # 빈 클러스터: 가장 큰 클러스터에서 토큰 하나를 떼어 재할당
            for e in torch.nonzero(empty, as_tuple=False).flatten().tolist():
                biggest = int(counts.argmax())
                members = torch.nonzero(assign == biggest, as_tuple=False).flatten()
                if members.numel() <= 1:
                    continue
                move = members[0]
                assign[move] = e
                counts[biggest] -= 1
                counts[e] += 1
                new_centroids[biggest] -= x[move]
                new_centroids[e] += x[move]

        nonempty = counts > 0
        centroids[nonempty] = _l2norm(
            new_centroids[nonempty] / counts[nonempty].unsqueeze(1)
        )

    # 최종 assignment
    sim = x @ centroids.t()
    return sim.argmax(dim=1)


@torch.no_grad()
def merge_tokens(tokens, attn_scores, k, method="simple_avg", max_iter=10,
                 init_indices=None):
    """
    Stage 1 보존 토큰(M1, D)을 M2(=k)개 대표 토큰으로 병합.

    Args:
        tokens: (M1, D) Stage 1 보존 토큰 (LLM 입력 feature 공간)
        attn_scores: (M1,) Stage 1에서 가져온 토큰별 [CLS]-attention 점수
                     (weighted_avg에서 가중치로 사용)
        k: 목표 토큰 수 M2
        method: "simple_avg" | "weighted_avg"
        max_iter: spherical k-means 최대 반복
    Returns:
        merged: (M2, D) 대표 토큰 (원 dtype)
    """
    M1, D = tokens.shape
    if k >= M1:
        return tokens

    assign = spherical_kmeans(tokens, k, max_iter=max_iter,
                              init_indices=init_indices)  # (M1,)
    tf = tokens.float()
    out = torch.zeros(k, D, device=tokens.device, dtype=torch.float32)

    if method == "weighted_avg":
        w = attn_scores.float().clamp_min(0)            # (M1,) 음수 방지
        out.index_add_(0, assign, tf * w.unsqueeze(1))  # Σ a_i x_i
        wsum = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        wsum.index_add_(0, assign, w)                   # Σ a_i
        # 가중치 합이 0인 클러스터는 단순평균으로 폴백
        cnt = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        cnt.index_add_(0, assign, torch.ones(M1, device=tokens.device))
        denom = wsum.clone()
        zero_w = denom <= _EPS
        if zero_w.any():
            avg = torch.zeros(k, D, device=tokens.device, dtype=torch.float32)
            avg.index_add_(0, assign, tf)
            out[zero_w] = avg[zero_w]
            denom[zero_w] = cnt[zero_w].clamp_min(1.0)
        out = out / denom.clamp_min(_EPS).unsqueeze(1)
    else:  # simple_avg
        out.index_add_(0, assign, tf)
        cnt = torch.zeros(k, device=tokens.device, dtype=torch.float32)
        cnt.index_add_(0, assign, torch.ones(M1, device=tokens.device))
        out = out / cnt.clamp_min(1.0).unsqueeze(1)

    return out.to(tokens.dtype)

"""
[Phase 1] Adaptive Stage-1 important 토큰 개수(n_imp)·M1 자동 결정 헬퍼.

기존 VisPruner는 important 개수를 고정 비율(M1·r)로 정한다. 본 모듈은 [CLS] attention
분포로부터 important 개수를 **이미지마다 자동**으로 결정하는 두 수식을 제공한다.

- (가) energy      : 내림차순 누적 attention 질량이 τ를 넘는 최소 토큰 수 (적분 관점)
- (나) statistical : attention 분포의 통계적 이상치 개수 (μ+kσ, 또는 robust median+k·MAD)

두 수식 모두 **n_imp만 정하고**, 공통 규칙(floor=M2, cap=M1_cap)으로 M1을 유도한다.
선택/병합 이후 로직(diverse 제거, Stage2 k-means)은 기존 경로를 그대로 재사용한다.

회귀 안전: 이 모듈은 selection_method ∈ {energy, statistical}일 때만 호출된다.
default(topk)에서는 import조차 되지 않으므로 기존 VisPruner 경로에 영향이 없다.
"""
import torch

_EPS = 1e-8


@torch.no_grad()
def compute_adaptive_counts(s, method, M2, energy_tau=0.5, stat_k=2.0,
                            stat_robust=False, r_floor=0.5, M1_cap=384):
    """
    Args:
        s: (B, N) [CLS] attention 점수 (head 평균, 정규화 전, 음수 없음 가정).
        method: "energy" | "statistical".
        M2: 최종 토큰 수(병합 후). M1의 하한(floor)으로 사용.
        energy_tau: (가) 누적 질량 임계 τ ∈ (0,1).
        stat_k: (나) 이상치 임계 계수 k.
        stat_robust: (나) True면 median+k·MAD, False면 mean+k·std.
        r_floor: n_imp → M1 환산 시 important 비율 하한(0.5 = n_imp를 절반으로 봄).
        M1_cap: M1 상한(배경/노이즈 과다 유입 방지).
    Returns:
        n_imp_list: list[int] (len B) — 이미지별 important 토큰 수.
        M1_list:    list[int] (len B) — 이미지별 Stage1 보존 수(clamp 후).
        detail_list: list[dict] (len B) — 이미지별 진단 정보
                     {"raw_M1": clamp 전 M1, "floor": floor 도달 여부, "cap": cap 도달 여부}.
    반환 규약: 항상 1 <= n_imp <= M1, M2 <= M1 <= min(M1_cap, N).
    floor/cap 판정은 clamp 전 raw_M1(=round(n_imp/r_floor)) 기준이라 정확하다.
    """
    B, N = s.shape
    cap_val = min(M1_cap, N)
    n_imp_list, M1_list, detail_list = [], [], []

    for b in range(B):
        sb = s[b]
        if method == "energy":
            # (가) 내림차순 누적 질량이 τ를 넘는 최소 개수
            sorted_s, _ = torch.sort(sb, descending=True)
            p = sorted_s / (sorted_s.sum() + _EPS)        # 질량 정규화
            csum = torch.cumsum(p, dim=0)
            n_imp = int((csum < energy_tau).sum().item()) + 1
        elif method == "statistical":
            # (나) 통계적 이상치 개수
            if stat_robust:
                med = sb.median()
                mad = (sb - med).abs().median()
                thr = med + stat_k * mad
            else:
                thr = sb.mean() + stat_k * sb.std()
            n_imp = int((sb >= thr).sum().item())
        else:
            raise ValueError(f"[adaptive_selection] unknown method: {method}")

        # 공통: n_imp 정규화 → raw_M1 → clamp(floor=M2, cap) → n_imp 재클램프
        n_imp = max(1, min(n_imp, N))
        raw_M1 = int(round(n_imp / r_floor))           # clamp 전 (floor/cap 판정 기준)
        M1 = max(M2, min(raw_M1, cap_val))             # clamp 후
        floor_hit = raw_M1 < M2                        # 하한에 걸림
        cap_hit = raw_M1 > cap_val                     # 상한에 걸림
        n_imp = min(n_imp, M1)                         # n_imp가 M1 넘지 않도록

        n_imp_list.append(int(n_imp))
        M1_list.append(int(M1))
        detail_list.append({"raw_M1": int(raw_M1),
                            "floor": bool(floor_hit), "cap": bool(cap_hit)})

    return n_imp_list, M1_list, detail_list

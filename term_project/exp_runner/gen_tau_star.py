"""[Phase3 실험 B] τ* 역산 — 고정 M1(=2×M2)이 되는 energy τ*를 데이터셋별로 산출.

묶음 A의 results_phase2A.tsv(AVG_M1 컬럼) 재활용, 추가 추론 불필요.
각 (데이터셋, M2)에서 (τ, AVG_M1) 측정점을 선형 보간해 AVG_M1 = 2×M2(=기존 고정 topk M1)이 되는 τ*를 찾는다.
→ "고정 M1은 각 데이터셋에서 서로 다른 τ*에 해당한다"(고정=adaptive의 특수해) 증명.

사용: python exp_runner/gen_tau_star.py > develop_md/phase3_tau_star_result.md
"""
RES = "exp_runner/results_phase2A.tsv"

# energy simple, (bench, M2) -> [(tau, avg_M1)]
data = {}
for l in open(RES):
    p = l.rstrip("\n").split("\t")
    if len(p) < 12 or not p[0].startswith("A-en") or not p[0].endswith("s"):
        continue
    try:
        M2 = int(p[2]); avgm1 = float(p[11]); tau = int(p[0][4:6]) / 10.0
    except (ValueError, IndexError):
        continue
    data.setdefault((p[1], M2), []).append((tau, avgm1))


def interp_tau(pairs, target):
    pairs = sorted(pairs)
    for i in range(len(pairs) - 1):
        t0, m0 = pairs[i]; t1, m1 = pairs[i + 1]
        if (m0 - target) * (m1 - target) <= 0 and m1 != m0:
            return round(t0 + (target - m0) * (t1 - t0) / (m1 - m0), 3)
    # 외삽 (마지막 두 점)
    if len(pairs) >= 2:
        (t0, m0), (t1, m1) = pairs[-2], pairs[-1]
        if m1 != m0:
            return round(t0 + (target - m0) * (t1 - t0) / (m1 - m0), 3)
    return None


print("# Phase3 실험 B — τ* 역산 (고정 M1 = adaptive 수식의 특수해)")
print()
print("> 묶음 A `results_phase2A.tsv`(AVG_M1) 재활용, 추가 추론 없음. energy simple 기준.")
print("> 각 M2의 기존 고정 topk M1 = 2×M2. 그 M1이 되는 energy τ*를 데이터셋별 보간.")
print()
print("## 데이터셋 × M2 별 τ* (AVG_M1 = 2×M2 되는 지점)")
print()
print("| 데이터셋 | M2 | 고정 M1(=2×M2) | 측정점 (τ→AVG_M1) | **τ\\*** |")
print("|---|---|---|---|---|")
rows = {}
for (bench, M2), pairs in sorted(data.items()):
    target = 2 * M2
    ts = interp_tau(pairs, target)
    rows[(bench, M2)] = ts
    pstr = ", ".join(f"{t}→{m:.0f}" for t, m in sorted(pairs))
    print(f"| {bench} | {M2} | {target} | {pstr} | **{ts}** |")
print()
print("## 요약 — 데이터셋별 τ* (M1=128, 즉 M2=64 기준)")
print()
print("| 데이터셋 | τ\\* (M1=128) |")
print("|---|---|")
for bench in ("pope", "gqa", "textvqa"):
    print(f"| {bench} | {rows.get((bench, 64), '—')} |")
print()

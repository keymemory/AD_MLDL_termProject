"""[Phase 1] adaptive selection 분포 분석.

사용: python analyze_dist.py <라벨> <answers.dist.jsonl> [M2=64] [M1_cap=384]

.dist.jsonl 한 줄당: {"method","n_imp","M1","raw_M1","floor","cap"}
출력: M1·n_imp·raw_M1 통계(min/max/평균/std), floor/cap/adapt 비율, M1 히스토그램.

판정 가이드:
- floor/cap 비율이 대부분(>~80%) → "적응이 아니라 사실상 고정". 수식/하이퍼 재설계 필요.
- adapt(중간) 비율이 충분 + M1이 흩어짐 → "진짜 이미지별 적응". τ/k 정당화 근거.
"""
import sys
import json


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5
    return min(xs), max(xs), m, sd


def histogram(xs, edges):
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    counts = [0] * len(bins)
    for x in xs:
        for i, (lo, hi) in enumerate(bins):
            if lo <= x < hi or (i == len(bins) - 1 and x == hi):
                counts[i] += 1
                break
    return bins, counts


def main():
    name, f = sys.argv[1], sys.argv[2]
    M2 = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 384

    recs = [json.loads(l) for l in open(f)]
    n = len(recs)
    M1 = [r["M1"] for r in recs]
    nimp = [r["n_imp"] for r in recs]
    rawM1 = [r.get("raw_M1", r["M1"]) for r in recs]
    floor = sum(1 for r in recs if r.get("floor"))
    capc = sum(1 for r in recs if r.get("cap"))
    adapt = n - floor - capc

    print(f"\n========== {name}  (n={n}, M2={M2}, cap={cap}) ==========")
    mn, mx, mean, sd = stats(M1)
    print(f"M1     : min={mn:3d} max={mx:3d} mean={mean:6.1f} std={sd:5.1f}")
    mn, mx, mean, sd = stats(nimp)
    print(f"n_imp  : min={mn:3d} max={mx:3d} mean={mean:6.1f} std={sd:5.1f}")
    mn, mx, mean, sd = stats(rawM1)
    print(f"raw_M1 : min={mn:3d} max={mx:3d} mean={mean:6.1f} std={sd:5.1f}  (clamp 전)")
    print(f"floor (M1<={M2})  : {floor:3d}/{n} = {100*floor/n:5.1f}%")
    print(f"cap   (M1>={cap}) : {capc:3d}/{n} = {100*capc/n:5.1f}%")
    print(f"adapt (그 사이)   : {adapt:3d}/{n} = {100*adapt/n:5.1f}%   <-- 진짜 적응 비율")

    edges = [M2 + i * (cap - M2) / 8 for i in range(9)]
    bins, h = histogram(M1, edges)
    mxc = max(h) or 1
    print("M1 histogram (M2~cap 8등분):")
    for (lo, hi), c in zip(bins, h):
        bar = "#" * int(40 * c / mxc)
        print(f"  [{int(lo):3d}-{int(hi):3d}) {c:3d} {bar}")


if __name__ == "__main__":
    main()

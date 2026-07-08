"""[Phase 2 Part1-C] M2별 energy/statistical 건강 τ/k 분석.

probe_dist.py가 만든 probe_{method}_M{M2}_{param}.dist.jsonl 들을 모아
(method, M2, param)별 n_imp/M1 평균, floor%, cap%, adapt% 를 집계하고
M2별 건강 구간을 판정한다.

건강 기준: adapt% >= 90 건강(✅) / 50~90 경계(🟡) / <50 붕괴(🔴)
사용: python exp_runner/analyze_healthy.py > develop_md/phase2_healthy_range.md
"""
import glob
import json
import os

P = "playground/data/eval/pope/answers/regress"


def status(adapt):
    if adapt >= 90:
        return "✅"
    if adapt >= 50:
        return "🟡"
    return "🔴"


rows = []
for f in sorted(glob.glob(f"{P}/probe_*_M*_*.dist.jsonl")):
    name = os.path.basename(f).replace(".dist.jsonl", "")
    parts = name.split("_")  # probe energy M32 0.3
    if len(parts) < 4:
        continue
    method, m2s, param = parts[1], parts[2], parts[3]
    M2 = int(m2s[1:])
    param = float(param)
    recs = [json.loads(l) for l in open(f)]
    n = len(recs)
    if n == 0:
        continue
    M1 = [r["M1"] for r in recs]
    nimp = [r["n_imp"] for r in recs]
    floor = 100.0 * sum(1 for r in recs if r.get("floor")) / n
    cap = 100.0 * sum(1 for r in recs if r.get("cap")) / n
    adapt = 100.0 - floor - cap
    rows.append({"method": method, "M2": M2, "param": param,
                 "M1": sum(M1) / n, "nimp": sum(nimp) / n,
                 "floor": floor, "cap": cap, "adapt": adapt})

print("# Phase 2 Part1-C — M2별 energy/statistical 건강 τ/k (자동 생성)")
print()
print("> probe(LLM 디코딩 없이 vision tower+selection만, POPE 300 subset)로 분포만 측정.")
print("> 건강 기준: **adapt% ≥ 90 건강(✅) / 50~90 경계(🟡) / <50 붕괴(🔴)**.")
print("> floor=M2(작은 important로 M1이 M2 아래 → clamp), cap=384.")
print()

for method, plabel, prange in [("energy", "τ", "0.3~0.9"), ("statistical", "k", "0.2~0.8")]:
    print(f"## {method} ({plabel} 스윕 {prange})")
    print()
    for M2 in (32, 64, 128):
        sub = sorted([r for r in rows if r["method"] == method and r["M2"] == M2], key=lambda x: x["param"])
        if not sub:
            continue
        print(f"### M2 = {M2}")
        print(f"| {plabel} | n_imp | M1 | floor% | cap% | adapt% | |")
        print("|---|---|---|---|---|---|---|")
        healthy = []
        for r in sub:
            print(f"| {r['param']} | {r['nimp']:.0f} | {r['M1']:.0f} | {r['floor']:.0f} | {r['cap']:.0f} | {r['adapt']:.0f} | {status(r['adapt'])} |")
            if r["adapt"] >= 90:
                healthy.append(r["param"])
        hstr = f"{min(healthy)}~{max(healthy)}" if healthy else "없음(전 구간 붕괴)"
        print(f"\n→ **M2={M2} {method} 건강구간({plabel}, adapt≥90%): {hstr}**")
        print()

print("---")
print()
print("## 종합 — M2별 건강 구간 요약")
print()
print("| M2 | energy 건강 τ | statistical 건강 k |")
print("|---|---|---|")
for M2 in (32, 64, 128):
    eh = sorted([r["param"] for r in rows if r["method"] == "energy" and r["M2"] == M2 and r["adapt"] >= 90])
    sh = sorted([r["param"] for r in rows if r["method"] == "statistical" and r["M2"] == M2 and r["adapt"] >= 90])
    es = f"{min(eh)}~{max(eh)}" if eh else "없음"
    ss = f"{min(sh)}~{max(sh)}" if sh else "없음"
    print(f"| {M2} | {es} | {ss} |")
print()
print("> Part 2 본실험은 각 M2의 건강구간 τ/k를 사용해야 adaptive가 topk와 구별된다.")

"""[Phase2 Part2 묶음A] results_phase2A.tsv → 전 구간 성능 곡선 + best τ/k md (셸 자동 생성).

데이터셋(POPE/GQA/TextVQA) × M2(32/64/128) 별로 τ(energy)/k(statistical)에 따른 성능 곡선.
건강구간(Part1) 표시. best τ/k = POPE 성능 최고인 건강 param (묶음B용).
사용: python exp_runner/generate_phase2A_report.py > develop_md/phase2_part2_A_result.md
"""
import os
from collections import defaultdict

RES = "/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project/exp_runner/results_phase2A.tsv"

# Part1 건강구간
HEALTHY = {
    ("energy", 32): [0.6, 0.7, 0.8], ("energy", 64): [0.7, 0.8], ("energy", 128): [0.8],
    ("statistical", 32): [0.2, 0.3, 0.4, 0.5, 0.6], ("statistical", 64): [0.2, 0.3],
}
ENERGY_TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STAT_KS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def parse_id(i):
    if not i.startswith("A-"):
        return None
    m = i[2:]
    method = "energy" if m.startswith("en") else ("statistical" if m.startswith("st") else None)
    if method is None:
        return None
    rest = m[2:]                       # 07-64s
    try:
        param = int(rest[:2]) / 10.0
        tail = rest.split("-")[1]      # 64s
        merge = "weighted" if tail.endswith("w") else "simple"
        M2 = int(tail[:-1])
    except (ValueError, IndexError):
        return None
    return method, param, M2, merge


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


# 파싱 + dedup (ID,BENCH 마지막 줄 우선)
rows = {}
if os.path.exists(RES):
    for l in open(RES):
        p = l.rstrip("\n").split("\t")
        if len(p) < 9 or not p[0].startswith("A-"):
            continue
        pi = parse_id(p[0])
        if not pi:
            continue
        method, param, M2, merge = pi
        rows[(p[0], p[1])] = {
            "method": method, "param": param, "M2": M2, "merge": merge, "bench": p[1],
            "value": p[8], "avg_M1": p[11] if len(p) > 11 else "-",
            "floor": p[13] if len(p) > 13 else "-", "cap": p[14] if len(p) > 14 else "-",
        }
recs = list(rows.values())
n_done = len(recs)

out = []
out.append("# Phase2 Part2 묶음A — 전 구간 성능 곡선 + best τ/k (자동 생성)")
out.append("")
out.append(f"> 셸 자동 생성. `results_phase2A.tsv` {n_done}/210 jobs. M2=128 statistical 제외.")
out.append("> ★ = Part1 건강구간(adapt≥90%). 검증: 건강구간 성능 ≥ 붕괴구간이어야 '건강=최적' 입증.")
out.append("")


def curve(method, bench, M2, params):
    sub = {r["param"]: r for r in recs if r["method"] == method and r["bench"] == bench and r["M2"] == M2}
    if not sub:
        return None
    plabel = "τ" if method == "energy" else "k"
    healthy = HEALTHY.get((method, M2), [])
    head = "| merge | " + " | ".join(f"{plabel}={p}{'★' if p in healthy else ''}" for p in params) + " |"
    sep = "|---|" + "---|" * len(params)
    lines = [head, sep]
    for merge in ("simple", "weighted"):
        cells = []
        for p in params:
            r = sub.get(p)
            cells.append(r["value"] if (r and r["merge"] == merge) else "-")
        # merge별 행 재구성
        row = {p: (sub[p]["value"] if p in sub and sub[p]["merge"] == merge else "-") for p in params}
        # sub는 param당 1개(마지막 merge)만 — merge 구분 위해 별도 필터
        rr = {r["param"]: r["value"] for r in recs
              if r["method"] == method and r["bench"] == bench and r["M2"] == M2 and r["merge"] == merge}
        cells = [rr.get(p, "-") for p in params]
        lines.append(f"| {merge} | " + " | ".join(cells) + " |")
    # avg_M1/floor 행 (simple 기준)
    am = {r["param"]: r["avg_M1"] for r in recs
          if r["method"] == method and r["bench"] == bench and r["M2"] == M2 and r["merge"] == "simple"}
    fl = {r["param"]: r["floor"] for r in recs
          if r["method"] == method and r["bench"] == bench and r["M2"] == M2 and r["merge"] == "simple"}
    lines.append(f"| _AVG_M1_ | " + " | ".join(str(am.get(p, "-")) for p in params) + " |")
    lines.append(f"| _floor%_ | " + " | ".join(str(fl.get(p, "-")) for p in params) + " |")
    return "\n".join(lines)


for method, params in [("energy", ENERGY_TAUS), ("statistical", STAT_KS)]:
    out.append(f"## {method}")
    out.append("")
    for bench in ("pope", "gqa", "textvqa"):
        for M2 in (32, 64, 128):
            if method == "statistical" and M2 == 128:
                continue
            c = curve(method, bench, M2, params)
            if c:
                out.append(f"### {bench.upper()} · M2={M2}")
                out.append(c)
                out.append("")

# best τ/k (POPE 성능 기준, 건강구간 내)
out.append("---")
out.append("")
out.append("## best τ/k (POPE 성능 최고인 건강 param — 묶음B용)")
out.append("")
out.append("| method | M2 | merge | best param | POPE 성능 |")
out.append("|---|---|---|---|---|")
best = {}
for method in ("energy", "statistical"):
    for M2 in (32, 64, 128):
        healthy = HEALTHY.get((method, M2), [])
        if not healthy:
            continue
        for merge in ("simple", "weighted"):
            cands = [(r["param"], fnum(r["value"])) for r in recs
                     if r["method"] == method and r["bench"] == "pope" and r["M2"] == M2
                     and r["merge"] == merge and r["param"] in healthy and fnum(r["value"]) is not None]
            if not cands:
                continue
            bp, bv = max(cands, key=lambda x: x[1])
            best[(method, M2, merge)] = bp
            out.append(f"| {method} | {M2} | {merge} | {bp} | {bv} |")
out.append("")
out.append("> 위 best param으로 묶음B(전 데이터셋 비교표)를 구성한다.")
print("\n".join(out))

"""[Phase 1 E2] results_phase1.tsv -> 결과 md 자동 생성 (Claude 없이 셸에서 실행).

E2 launch 완료 후 체인에서 호출되어 develop_md/phase1_e2_result.md 를 만든다.
τ*/k* 역산(고정 M1=128 = 적응 수식의 특수해)까지 포함.

사용: python exp_runner/generate_e2_report.py > develop_md/phase1_e2_result.md
"""
import os

TP = "/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project"
RES_P1 = os.path.join(TP, "exp_runner/results_phase1.tsv")
RES_BASE = os.path.join(TP, "exp_runner/results.tsv")


def load(path, minlen=11):
    rows = []
    if os.path.exists(path):
        for l in open(path):
            p = l.rstrip("\n").split("\t")
            if len(p) >= minlen and p[0] and not p[0].startswith("ID"):
                rows.append(p)
    return rows


def parse_id(i):
    m = i.replace("E2-", "")
    if m.startswith("en"):
        method = "energy"
    elif m.startswith("st"):
        method = "statistical"
    else:
        return None
    rest = m[2:]
    merge = "weighted" if rest.endswith("w") else "simple"
    try:
        param = int(rest[:-1]) / 100.0
    except ValueError:
        return None
    return method, param, merge


# ---- E2 결과 파싱 ----
# 컬럼: 0ID 1BENCH 2M2 3CLUST 4M1 5METHOD 6R 7METRIC 8VALUE 9GEN 10SELMETHOD 11AVG_M1 12AVG_R 13FLOOR 14CAP
e2 = load(RES_P1, minlen=11)
recs = []
for r in e2:
    pi = parse_id(r[0])
    if not pi:
        continue
    method, param, merge = pi
    g = lambda i: r[i] if len(r) > i else "-"
    recs.append({"id": r[0], "method": method, "param": param, "merge": merge,
                 "bench": r[1], "metric": g(7), "value": g(8), "gen": g(9),
                 "avg_M1": g(11), "avg_r": g(12), "floor": g(13), "cap": g(14)})

# ---- baseline (B-64s/w from results.tsv) ----
base = load(RES_BASE, minlen=9)
basemap = {}
for r in base:
    if r[0] in ("B-64s", "B-64w") and r[1] in ("pope", "gqa"):
        basemap[(r[0], r[1])] = r[8]


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def table(bench, merge):
    """한 (bench, merge) 조합의 energy+statistical 표."""
    lines = []
    lines.append(f"#### {bench.upper()} · {merge}_avg")
    lines.append("| method | param | 성능 | AVG_M1 | AVG_R | FLOOR% | CAP% |")
    lines.append("|---|---|---|---|---|---|---|")
    base_id = "B-64s" if merge == "simple" else "B-64w"
    bval = basemap.get((base_id, bench), "-")
    lines.append(f"| **topk(고정 M1=128)** | — | {bval} | 128.0 | 0.5 | 0 | 0 |")
    for method, plabel in [("energy", "τ"), ("statistical", "k")]:
        rs = sorted([x for x in recs if x["method"] == method and x["bench"] == bench and x["merge"] == merge],
                    key=lambda x: x["param"])
        for x in rs:
            lines.append(f"| {method} | {plabel}={x['param']} | {x['value']} | {x['avg_M1']} | {x['avg_r']} | {x['floor']} | {x['cap']} |")
    return "\n".join(lines)


def m1_pairs(method, merge, bench):
    ps = []
    for x in recs:
        if x["method"] == method and x["merge"] == merge and x["bench"] == bench:
            m = fnum(x["avg_M1"])
            if m is not None:
                ps.append((x["param"], m))
    return sorted(ps)


def interp_param(pairs, target=128.0):
    """avg_M1=target 이 되는 param 선형보간/외삽."""
    if len(pairs) < 2:
        return None
    for i in range(len(pairs) - 1):
        p0, m0 = pairs[i]
        p1, m1 = pairs[i + 1]
        if (m0 - target) * (m1 - target) <= 0 and m1 != m0:
            return round(p0 + (target - m0) * (p1 - p0) / (m1 - m0), 3)
    # 외삽 (가장 가까운 두 점)
    (p0, m0), (p1, m1) = pairs[-2], pairs[-1]
    if m1 != m0:
        return round(p0 + (target - m0) * (p1 - p0) / (m1 - m0), 3)
    return None


def best_setting(method, bench):
    """성능 최고 param (merge 무관)."""
    cands = [x for x in recs if x["method"] == method and x["bench"] == bench and fnum(x["value"]) is not None]
    if not cands:
        return None
    b = max(cands, key=lambda x: fnum(x["value"]))
    return b


# ---- md 출력 ----
out = []
out.append("# Phase 1 E2 — energy/statistical 본실험 결과 (자동 생성)")
out.append("")
out.append(f"> 셸 자동 생성(Claude 미개입). 데이터: `results_phase1.tsv` ({len(recs)}/24 jobs 기록됨).")
out.append("> M2=64 고정, clustering on, POPE+GQA. baseline=topk M1=128(B-64s/w 인용).")
out.append("> energy τ=0.6/0.7/0.8, statistical k=0.3/0.4/0.5, 각 simple/weighted.")
out.append("")
out.append("---")
out.append("")
out.append("## 1. 결과표 (성능 + AVG_M1/AVG_R + floor/cap)")
out.append("")
for bench in ("pope", "gqa"):
    for merge in ("simple", "weighted"):
        out.append(table(bench, merge))
        out.append("")

out.append("---")
out.append("")
out.append("## 2. τ*/k* 역산 — \"고정 M1=128 = 적응 수식의 특수해\"")
out.append("")
out.append("AVG_M1이 고정 baseline(M1=128)과 같아지는 파라미터를 선형보간으로 산출:")
out.append("")
out.append("| 수식 | merge | bench | (param, AVG_M1) 측정점 | **M1=128 되는 지점** |")
out.append("|---|---|---|---|---|")
for method, plabel in [("energy", "τ*"), ("statistical", "k*")]:
    for merge in ("simple", "weighted"):
        for bench in ("pope", "gqa"):
            pairs = m1_pairs(method, merge, bench)
            if not pairs:
                continue
            star = interp_param(pairs)
            pstr = ", ".join(f"({p},{m:.0f})" for p, m in pairs)
            out.append(f"| {method} | {merge} | {bench} | {pstr} | **{plabel}={star}** |")
out.append("")
out.append("> → 적응 수식이 특정 τ*/k*에서 고정 M1=128과 동일한 평균 토큰 수를 내므로,")
out.append("> **고정 방식은 적응 수식의 한 특수해**임을 보인다(적응이 고정을 일반화).")
out.append("")
out.append("---")
out.append("")
out.append("## 3. best 설정 (성능 기준)")
out.append("")
out.append("| 수식 | bench | best param | merge | 성능 |")
out.append("|---|---|---|---|---|")
for method in ("energy", "statistical"):
    for bench in ("pope", "gqa"):
        b = best_setting(method, bench)
        if b:
            out.append(f"| {method} | {bench} | {b['param']} | {b['merge']} | {b['value']} |")
out.append("")
out.append("---")
out.append("")
out.append("## 4. 다음 스텝")
out.append("- (c) best τ/k로 **M2=32/128 확장** (단계적).")
out.append("- baseline(B-64s/w) 대비 각 수식 우열 + floor/cap 건강성 점검.")
out.append("- 원자료: `results_phase1.tsv`, 분포 `answers/EXP/E2-*/r_0.5.jsonl.dist.jsonl`.")
out.append("")

print("\n".join(out))

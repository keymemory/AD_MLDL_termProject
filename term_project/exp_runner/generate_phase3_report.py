"""[Phase3 실험A] taskaware 병합 결과 → md (셸 자동 생성).
results_phase3_merge.tsv(taskaware, k_d별) vs 묶음A energy τ=0.8 weighted baseline(results_phase2A.tsv) 비교.
판정: TextVQA taskaware > weighted (OCR 회복) & POPE/GQA taskaware ≥ weighted−0.5 (안 해침).
preserve 평균은 .dist.jsonl에서.
사용: python exp_runner/generate_phase3_report.py > develop_md/phase3_taskaware_merge_result.md
"""
import glob
import json
import os

TP = "/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project"
P3 = os.path.join(TP, "exp_runner/results_phase3_merge.tsv")
PA = os.path.join(TP, "exp_runner/results_phase2A.tsv")
ABBR = {"tv": "textvqa", "po": "pope", "gq": "gqa"}
KDS = [0.5, 1.0, 1.5, 2.0]
M2S = [32, 64, 128]


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


# taskaware 결과: (bench, M2, kd) -> value
ta = {}
if os.path.exists(P3):
    for l in open(P3):
        p = l.rstrip("\n").split("\t")
        if len(p) < 9 or not p[0].startswith("P3-"):
            continue
        try:
            _, ab, kds, m2 = p[0].split("-")
            ta[(ABBR[ab], int(m2), int(kds[2:]) / 10.0)] = p[8]
        except (ValueError, KeyError):
            continue

# baseline: 묶음A energy τ=0.8 weighted (A-en08-*w)
base = {}
if os.path.exists(PA):
    for l in open(PA):
        p = l.rstrip("\n").split("\t")
        if p and p[0].startswith("A-en08-") and p[0].endswith("w"):
            base[(p[1], int(p[2]))] = p[8]


def avg_preserve(bench, M2, kd):
    kdi = int(round(kd * 10))
    jid = f"P3-{[k for k, v in ABBR.items() if v == bench][0]}-kd{kdi:02d}-{M2}"
    f = os.path.join(TP, f"playground/data/eval/{bench}/answers/EXP/{jid}/r_0.5.jsonl.dist.jsonl")
    if not os.path.exists(f):
        return None
    ps = [json.loads(l).get("preserve") for l in open(f) if "preserve" in l]
    ps = [x for x in ps if x is not None]
    return round(sum(ps) / len(ps), 1) if ps else None


out = []
out.append("# Phase3 실험A — Task-Aware (merge-distortion) 병합 결과 (자동 생성)")
out.append("")
out.append(f"> `results_phase3_merge.tsv` ({len(ta)} jobs). baseline = 묶음A energy τ=0.8 **weighted**.")
out.append("> 가설: TextVQA(OCR)에서 taskaware > weighted, POPE/GQA는 taskaware ≥ weighted−0.5(안 해침).")
out.append("")
for bench in ("textvqa", "pope", "gqa"):
    out.append(f"## {bench.upper()}  (baseline = weighted, energy τ=0.8)")
    out.append("| M2 | weighted(base) | k_d=0.5 | 1.0 | 1.5 | 2.0 | best Δ |")
    out.append("|---|---|---|---|---|---|---|")
    for M2 in M2S:
        b = base.get((bench, M2), "-")
        bv = fnum(b)
        cells = []
        deltas = []
        for kd in KDS:
            v = ta.get((bench, M2, kd), "-")
            cells.append(v)
            fv = fnum(v)
            if fv is not None and bv is not None:
                deltas.append(fv - bv)
        best = max(deltas) if deltas else None
        bd = f"{best:+.3f}" if best is not None else "-"
        out.append(f"| {M2} | {b} | " + " | ".join(cells) + f" | {bd} |")
    # preserve 평균
    pv_row = []
    for M2 in M2S:
        for kd in KDS:
            pv = avg_preserve(bench, M2, kd)
            if pv is not None:
                pv_row.append(f"M2={M2},kd={kd}:{pv}")
    if pv_row:
        out.append(f"\n_preserve 평균: {', '.join(pv_row[:6])}_")
    out.append("")

out.append("## 판정")
out.append("- **성공**: TextVQA taskaware > weighted 이면서 POPE/GQA taskaware ≥ weighted−0.5")
out.append("- 위 표에서 TextVQA best Δ > 0 이고 POPE/GQA best Δ ≥ −0.5 이면 성공.")
out.append("- (자동 판정은 수치 확인 후 수동 해석 필요 — best Δ 부호로 1차 판단)")
print("\n".join(out))

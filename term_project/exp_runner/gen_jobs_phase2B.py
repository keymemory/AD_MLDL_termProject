"""[Phase2 Part2 묶음 B] 최종 비교표용 job tsv 생성.

- adaptive: VQAv2/SQA/MME/MMBench × energy(τ=0.8)/statistical(k=0.2) × simple/weighted × M2(32/64/128).
  M2=128 statistical 제외. (TextVQA는 묶음A 완료, best τ=0.5라 이번 대상 아님)
- topk 기준선: MME/MMBench × {VisPruner(clust off) / Ours-S / Ours-W} × M2(32/64/128).
  (POPE/GQA/TextVQA/VQAv2/SQA topk는 기존 표2/results 재활용)
11컬럼: ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST
"""
ADAPTIVE_BENCH = ["vqav2", "sqa", "mme", "mmbench"]
M2S = [32, 64, 128]
MERGES = [("simple_avg", "s"), ("weighted_avg", "w")]
ABBR = {"vqav2": "vqa2", "sqa": "sqa", "mme": "mme", "mmbench": "mmb"}

lines = []
# adaptive
for bench in ADAPTIVE_BENCH:
    ab = ABBR[bench]
    for M2 in M2S:
        M1 = M2 * 2
        for mname, msuf in MERGES:
            lines.append(f"B-{ab}-en08-{M2}{msuf}\t{bench}\t{M2}\t1\t{M1}\t{mname}\t0.5\tenergy\t0.8\t0.2\t0")
            if M2 != 128:   # M2=128 statistical 제외
                lines.append(f"B-{ab}-st02-{M2}{msuf}\t{bench}\t{M2}\t1\t{M1}\t{mname}\t0.5\tstatistical\t0.5\t0.2\t0")

# topk 기준선 (MME/MMBench만 새로)
for bench in ["mme", "mmbench"]:
    ab = ABBR[bench]
    for M2 in M2S:
        M1 = M2 * 2
        # VisPruner: topk + clustering off (M1=M2, 병합 없음)
        lines.append(f"B-{ab}-visp-{M2}\t{bench}\t{M2}\t0\t{M2}\tnone\t0.5\ttopk\t0.8\t0.2\t0")
        # Ours-S / Ours-W: topk + clustering on
        for mname, msuf in MERGES:
            lines.append(f"B-{ab}-o{msuf.upper()}-{M2}\t{bench}\t{M2}\t1\t{M1}\t{mname}\t0.5\ttopk\t0.8\t0.2\t0")

with open("exp_runner/exp_jobs_phase2_B.tsv", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"generated {len(lines)} jobs -> exp_runner/exp_jobs_phase2_B.tsv")
print(f"  adaptive: {sum(1 for l in lines if 'en08' in l or 'st02' in l)}")
print(f"  topk baseline: {sum(1 for l in lines if 'visp' in l or '-oS-' in l or '-oW-' in l)}")

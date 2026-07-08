"""[Phase2 Part2 묶음 A] 전 구간 세밀 스윕 job tsv 생성.
POPE/GQA/TextVQA × energy(τ 0.3~0.9)/statistical(k 0.2~0.8) × simple/weighted × M2(32/64/128).
M2=128 statistical 제외. 11컬럼: ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST.
M1은 placeholder(M2*2) — energy/statistical은 수식이 자동 결정.
"""
ENERGY_TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STAT_KS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
M2S = [32, 64, 128]
MERGES = [("simple_avg", "s"), ("weighted_avg", "w")]
BENCHES = ["pope", "gqa", "textvqa"]

lines = []
for bench in BENCHES:
    for M2 in M2S:
        M1 = M2 * 2
        for mname, msuf in MERGES:
            for tau in ENERGY_TAUS:
                tid = f"A-en{int(round(tau*10)):02d}-{M2}{msuf}"
                lines.append(f"{tid}\t{bench}\t{M2}\t1\t{M1}\t{mname}\t0.5\tenergy\t{tau}\t2.0\t0")
            if M2 != 128:   # M2=128 statistical 제외 (Part1 전구간 붕괴)
                for k in STAT_KS:
                    tid = f"A-st{int(round(k*10)):02d}-{M2}{msuf}"
                    lines.append(f"{tid}\t{bench}\t{M2}\t1\t{M1}\t{mname}\t0.5\tstatistical\t0.5\t{k}\t0")

with open("exp_runner/exp_jobs_phase2_A.tsv", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"generated {len(lines)} jobs -> exp_runner/exp_jobs_phase2_A.tsv")

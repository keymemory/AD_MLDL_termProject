"""[Phase3 실험 A] Task-aware 병합 job tsv 생성.
TextVQA(핵심)/POPE/GQA × merge_method=taskaware × k_d(0.5/1.0/1.5/2.0) × M2(32/64/128).
selection energy τ=0.8 (과보존+병합 → OCR 손상 큰 조건). baseline weighted는 묶음 A에서 재활용.
12컬럼: ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST KD
"""
BENCHES = ["textvqa", "pope", "gqa"]
KDS = [0.5, 1.0, 1.5, 2.0]
M2S = [32, 64, 128]
ABBR = {"textvqa": "tv", "pope": "po", "gqa": "gq"}

lines = []
for bench in BENCHES:
    ab = ABBR[bench]
    for M2 in M2S:
        M1 = M2 * 2
        for kd in KDS:
            kdi = int(round(kd * 10))
            # merge_method=taskaware, energy τ=0.8, KD 컬럼(12번째)
            lines.append(f"P3-{ab}-kd{kdi:02d}-{M2}\t{bench}\t{M2}\t1\t{M1}\ttaskaware\t0.5\tenergy\t0.8\t2.0\t0\t{kd}")

with open("exp_runner/exp_jobs_phase3.tsv", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"generated {len(lines)} taskaware jobs -> exp_runner/exp_jobs_phase3.tsv")
print(f"  benches={BENCHES}, kd={KDS}, M2={M2S}")

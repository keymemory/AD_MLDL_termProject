# Phase 3 — 구현·예약 상태 노트

> 실험 B(τ* 역산)는 **완료**, 실험 A(task-aware 병합)는 **구현·검증 완료 + 추론 예약**(묶음 B 완료 후 cron 자동).

---

## 실험 B — τ* 역산 ✅ 완료
- 결과: `develop_md/phase3_tau_star_result.md`
- 핵심: 고정 M1=128(M2=64)이 데이터셋별로 다른 τ*(POPE 0.646 / GQA 0.637 / TextVQA 0.692)에 대응.
  M2별로도 다름(0.52/0.65/0.79). → **고정 = adaptive energy의 τ=τ* 특수해** (묶음A 분포 재활용, 추가 추론 없음).
- 검증: τ* 지점 성능(≈0.815) ≈ 고정 topk 성능(0.8227), Phase1 E2와 일관.

---

## 실험 A — Task-Aware (merge-distortion) 병합 ✅ 구현·검증, ⏳ 추론 예약

### 구현 (회귀 안전)
- `spherical_kmeans.py::merge_tokens_taskaware`: K-Means 후 각 토큰의 distortion(1−cos(x,centroid)) 계산,
  통계적 이상치(μ+k_d·σ)인 토큰은 병합 제외·원본 보존, 나머지는 weighted 병합. **M2 불변**(preserve p + 병합 M2−p).
  p ≤ M2·0.5 상한(초과 시 상위 distortion만).
- `--merge_method taskaware` + `--taskaware_kd` CLI (loader/mmbench/science 3파일), llava_llama `taskaware_kd` getter,
  `encode_images` Stage2 분기. **기존 simple/weighted 경로 불변**(회귀 안전).
- worker.sh: 12번째 KD 컬럼 + `--taskaware_kd` 전달(taskaware일 때만). 기존 11컬럼 job 무해.

### 단위 테스트 (통과)
- M1=128 → M2=64: kd=0.5/1.0/1.5/2.0에서 **out.shape[0]=64 (M2 불변 OK)**, preserve=32/26/7/3 (kd↑→보존↓).
- 기존 weighted_avg 경로 보존 확인.

### 추론 예약 (묶음 B 완료 후 자동)
- job: `exp_jobs_phase3.tsv` (36 job: TextVQA/POPE/GQA × k_d 0.5~2.0 × M2 32/64/128, energy τ=0.8).
- cron: `run_phase3_chain.sh` + `run_phase3_extra.sh` (GPU 1·2). **`/tmp/phase2B_done.marker` 대기 조건** →
  묶음 B 완주 시 자동 시작. resume/flock detach.
- 완료 시 `generate_phase3_report.py` → `phase3_taskaware_merge_result.md` 자동 생성
  (k_d별 taskaware vs weighted baseline + preserve 평균 + 판정).
- baseline은 묶음 A energy τ=0.8 weighted 재활용.

### 결과 파일 (지정 준수, 기존 미덮어쓰기)
- `results_phase3_merge.tsv`, `develop_md/phase3_taskaware_merge_result.md`, `develop_md/phase3_tau_star_result.md`.

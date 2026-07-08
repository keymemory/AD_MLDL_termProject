# Phase 2 — Part 1: 데이터셋 확장 + 건강 τ/k 확인 (Claude Code 지시서)

> **목적**: stage1+2 완전한 성능 결과물을 뽑기 위한 **준비 단계**.
> (1) MME·MMBench 벤치마크 추가, (2) topk로 작동 검증, (3) 각 M2별 energy/statistical 건강 τ/k 확인.
> **본 실험(전체 성능 매트릭스)은 이 단계 결과를 보고 Part 2에서 별도 지시한다.**

---

## 0. 작업 환경 규칙 (먼저 읽고 전체에 적용)

- **GPU 최대 활용**: 사용 가능한 GPU를 최대한 다 써서 병렬화하라. `nvidia-smi`로 확인하되,
  **타 사용자가 점유 중인 GPU는 건드리지 마라**(이전 sihwang 사용자 vllm 사례 참고).
- **확인 질문 금지**: 터미널에서 yes/no 확인을 나에게 묻지 마라. 중간에 멈추지 말고 **쭉 수행**하라.
  파괴적 작업(파일 삭제·덮어쓰기)은 백업 후 알아서 진행하고 기록만 남겨라.
- **오류 자동 처리**: 오류가 나면 멈추지 말고 알아서 분석·수정하라. 무엇이 왜 났고 어떻게 고쳤는지
  `develop_md/`에 기록하라.
- **문서화**: 각 파트(구현/검증/건강값 확인)와 결과, 최종 결과를 단계별로 `develop_md/`에 문서화하라.
- **회귀 안전**: 새 기능은 항상 기존 동작을 보존해야 한다(topk default = 기존 VisPruner 비트동일 유지).

---

## 1. 배경 — 우리가 지금 무엇을 하고 있나

### 1-1. 연구 개요
LLaVA-1.5-7B의 시각 토큰(576개)을 2단계로 압축하는 연구.
- **Stage 1 (VisPruner 기반 선택)**: [CLS] attention으로 important 토큰 + cosine 다양성으로 diverse 토큰 → M1개 보존
- **Stage 2 (Spherical K-Means 병합)**: M1개를 최종 M2개(32/64/128)로 병합. 버려질 토큰 정보를 클러스터 대표 벡터에 흡수
- **변형**: VisPruner(Stage1만) / Ours-S(simple avg 병합) / Ours-W(weighted avg 병합)

### 1-2. 이미 완료된 것 (Phase 1)
- `--selection_method {topk, energy, statistical}` 구현 완료 (adaptive_selection.py)
  - **topk**: 기존 VisPruner 방식. M1 고정(예: M2의 2배)
  - **energy**: 누적 [CLS] attention 질량이 τ 넘는 최소 토큰 = important. M1 자동 결정
  - **statistical**: μ+kσ(또는 median+k·MAD) 이상치 = important. M1 자동 결정
- M1 자동 결정 공식: `M1 = clamp(round(n_imp/0.5), floor=M2, cap=384)`, r=0.5 고정
- 회귀 검증 통과: topk default = 기존 VisPruner 비트동일 (POPE 8910 text_diff=0)
- Phase 1 동작 검증에서 확인된 것 (POPE, M2=64 기준):
  - energy 건강구간 τ=0.7~0.8 (τ≤0.5 floor 붕괴, τ=0.9 cap 붕괴)
  - statistical 건강구간 k=0.3~0.5 (k≥0.8 floor 붕괴 — attention long-tail 때문)

### 1-3. 미팅 결론 — 이번에 할 일
팀 미팅에서 "stage1만으론 부족하니 **stage1+stage2까지 수행한 완전한 성능 결과물**을 뽑고 다시 논의하자"는 결론.
즉 우리 adaptive selection(energy/statistical) + K-Means 병합(simple/weighted)을
**여러 데이터셋 × M2(32/64/128)**에서 돌려 기존 결과(VisPruner/Ours-S/Ours-W)와 비교해야 한다.

### 1-4. 교수님 피드백 (실험 보강)
- 데이터셋: 현재 5개(POPE/GQA/TextVQA/VQAv2/SQA). 본인 도메인(QA·classification)에 맞는 태스크 1~2개 추가.
- 비교 모델: 2023~2025 코드 공개된 것으로 확대.
- 잘 되는 섹션에 페이퍼 포지셔닝.
- → **이번 Part 1에서 MME·MMBench 추가**로 대응 (둘 다 VisPruner 표준 + classification 계열 강점).

---

## 2. 현재 코드 구조 (참고)

- **레포 루트**: `/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project`
- **selection 로직**: `llava/model/llava_arch.py` `encode_images()` + `llava/model/adaptive_selection.py`
- **병합**: `llava/model/spherical_kmeans.py` (`merge_tokens(method=simple_avg|weighted_avg)`)
- **추론 엔트리**: `llava/eval/model_vqa_loader.py`(POPE/GQA/TextVQA/VQAv2), `model_vqa_science.py`(SQA)
- **실험 러너**: `exp_runner/` — `launch.sh`(병렬), `worker.sh`(락+resume+채점), `exp_jobs*.tsv`(job), `results*.tsv`(결과)
- **분포 probe**: `exp_runner/probe_dist.py` (LLM 디코딩 생략, vision tower+selection만 → 수십 배 빠름)
- **job tsv 컬럼 (11열)**: `ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST`
- **results tsv (13열)**: 위 + `SELMETHOD AVG_M1 AVG_R` + 성능 METRIC/VALUE

> ⚠️ 주의: CUDA_LAUNCH_BLOCKING=1 권장(이전 illegal memory access 사례). transformers 4.37.2 고정.

---

## 3. 이번 Part 1에서 할 일 (3가지)

### 작업 A — MME·MMBench 추가

현재 코드는 POPE/GQA/TextVQA/VQAv2(loader) + SQA(science)만 지원. 여기에 MME·MMBench 추가.

1. **데이터 확인**: VSCode/서버에 MME·MMBench 데이터가 있는지 먼저 확인
   (`playground/data/eval/` 또는 `dataset/` 경로). 없으면 어디서 받아야 하는지 `develop_md/`에 기록하고,
   있는 것부터 진행.
2. **추론 엔트리·채점기 추가**: 기존 `exp_runner` 구조에 맞춰 MME·MMBench 평가 경로 추가.
   - **MME**: perception/cognition 14개 하위 태스크. 공식 평가 방식(yes/no 정확도 + score 합산) 따름.
     하위 태스크별 점수도 기록(향후 task별 분석 자산).
   - **MMBench**: 객관식(A/B/C/D). **채점 방식이 기존과 다름** — 선택지 매칭 채점기 필요. circular eval 주의.
   - LLaVA 공식 레포의 MME/MMBench 평가 스크립트를 참고해 기존 구조에 통합.
3. **selection 인자 호환**: 새 벤치마크도 `--selection_method` 등 11컬럼 체계와 호환되게.

### 작업 B — topk 작동 검증

MME·MMBench를 **topk(기존 VisPruner) 모드**로 1회 돌려서 정상 작동 확인.
- 정상 채점되고 results에 기록되면 통과.
- 기존 5개 데이터셋도 topk로 한 번씩 돌려 회귀 안전 재확인(선택).

### 작업 C — ★ 각 M2별 건강 τ/k 확인 (본 실험 전 필수)

energy/statistical은 M2에 따라 floor/cap 위치가 달라져 건강 구간이 바뀐다.
**floor=M2**이므로 M2=32/64/128에서 건강 τ/k가 다를 수밖에 없다.
`probe_dist.py`(LLM 디코딩 없이 분포만, 빠름)로 확인하라.

**스윕 범위 (0.1 단위 전 구간 — 사용자 지시):**
- **energy τ = 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9** (7개)
- **statistical k = 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8** (7개)

**확인 대상:**
- M2 = 32 / 64 / 128 각각에 대해 위 전 구간을 probe로 스윕
- 데이터셋은 **POPE**로 대표 확인 (필요시 GQA도)
- 각 (M2, τ) 및 (M2, k)마다: n_imp 평균, M1 평균, floor%, cap%, **adapt%** 기록

**산출:**
- 표로 정리: "M2=32 → energy 건강 τ 구간=?, statistical 건강 k 구간=? / M2=64 → ... / M2=128 → ..."
- **건강 기준**: adapt% ≥ 90%면 건강, 50~90% 경계, <50% 붕괴
- `develop_md/phase2_healthy_range.md`에 전체 분포표 + 각 M2별 건강 구간 기록

> **왜 이게 필수인가**: τ를 잘못 고정하면(예: 모든 M2에 τ=0.7) M2=128에서 floor 붕괴해
> adaptive가 topk와 똑같아진다 → 결과 표에서 "energy = topk"로 나와 adaptive가 무의미해 보임.
> 각 M2의 건강 τ/k를 먼저 잡아야 본 실험이 정확하다.

---

## 4. 이번 Part 1 done 조건

1. ✅ MME·MMBench 추가 + topk로 작동 검증 통과 (정상 채점·기록)
2. ✅ M2=32/64/128 각각의 energy 건강 τ 구간, statistical 건강 k 구간 확정 (전 구간 0.1단위 probe → 표)
3. ✅ 모든 과정(구현/오류수정/검증/건강값) `develop_md/`에 문서화

---

## 5. 다음 단계 예고 (Part 2 — 본 실험, 이번엔 안 함)

Part 1 결과(MME·MMBench 작동 + 각 M2 건강 τ/k)를 받으면 Part 2 본 실험을 지시한다. 미리 참고:

- **전 구간 세밀 성능 실험** (POPE·GQA·TextVQA 3개):
  - energy τ=0.3~0.9 전부 + statistical k=0.2~0.8 전부 (성능까지 측정 — 양 끝 붕괴 구간도 측정해 "왜 건강구간이 최적인지" 정당화)
  - × 병합 simple/weighted × M2(32/64/128)
- **비교 표 채우기** (전 데이터셋 POPE/GQA/TextVQA/VQAv2/SQA/MME/MMBench):
  - 기존 VisPruner/Ours-S/Ours-W(topk) **유지** + best τ/k의 energy/statistical × simple/weighted 추가
  - 보고서 표 2 형식: 같은 M2·같은 데이터셋에서 고정 selection vs adaptive selection 직접 비교
- job은 `exp_jobs*.tsv`에 전부 나열 → `launch.sh` 병렬 실행 → `results.tsv` 집계 (기존 인프라 그대로)

**지금은 Part 1(작업 A·B·C)만 수행하고 결과를 보고하라. Part 2는 별도 지시한다.**
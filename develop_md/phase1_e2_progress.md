# Phase 1 E2 — 진행 중 스냅샷 + 자동완료 안내 (→ 클로드 웹)

> **VSCode 닫힘 대비 스냅샷.** E2가 백그라운드(완전 detached)로 실행 중이며,
> **완료 시 셸이 `phase1_e2_result.md`를 자동 생성**한다(Claude 미개입). ETA ~6시간.

---

## 0. 지금까지 확정된 결과 (E2 이전, 모두 완료)

| 단계 | 결과 |
|---|---|
| 구현 (단계 0~6) | `--selection_method {topk,energy,statistical}` + adaptive_selection.py |
| **회귀 2단계** | ✅ POPE 300 subset + **full 8910 비트동일**(text_diff=0, F1 16자리 동일) |
| 동작 검증 | energy(τ=0.8) adapt 100% / statistical(k=2.0) floor 100% 붕괴 |
| statistical 재검증 | **k=0.3 adapt 94%, k=0.5 52% → 생존**(percentile 불필요) |
| energy τ 스윕 | **건강구간 τ=0.7~0.8** (τ≤0.5 floor·τ=0.9 cap 붕괴 — 사용자 가정 반전) |

---

## 1. E2 본실험 (진행 중)

| 항목 | 내용 |
|---|---|
| 매트릭스 | energy τ=**0.6/0.7/0.8** + statistical k=**0.3/0.4/0.5**, 각 simple/weighted |
| 벤치 | POPE + GQA, **M2=64 고정** (24 job) |
| baseline | topk M1=128 (= 기존 B-64s/w 인용) |
| GPU | 0·1번 (GPU2는 sihwang님 것이라 제외) |
| 기록 | 성능(F1/Acc) + AVG_M1 + AVG_R + **FLOOR% + CAP%** (results_phase1.tsv 15컬럼) |

---

## 2. 자동완료 메커니즘 (VSCode 닫아도 생존)

- **실행 체인**: `launch_phase1.sh "1 0"` 완료 → `generate_e2_report.py`가
  `develop_md/phase1_e2_result.md` **자동 생성**(τ*/k* 역산 포함).
- **detach**: `setsid`로 새 세션 리더 → VSCode/SSH/Claude 세션 종료와 무관하게 계속 실행.
- **resume 안전**: 중단돼도 `answers/EXP/E2-*/`가 보존돼 이어쓰기(25회 retry).

## 3. 재접속 시 확인 방법

```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
wc -l exp_runner/results_phase1.tsv          # 24 = 완료
cat develop_md/phase1_e2_result.md           # 완료됐으면 최종 결과(역산 포함)
tail /tmp/launch_p1.log /tmp/e2_chain.log     # 진행/완료 로그
# 미완 시 진행 확인:
for d in playground/data/eval/pope/answers/EXP/E2-* playground/data/eval/gqa/answers/EXP/E2-*; do
  echo "$d: $(wc -l < $d/r_0.5.jsonl 2>/dev/null)"; done
```

## 4. 완료 후 다음 스텝
- `phase1_e2_result.md`에 결과표 + **τ*/k* 역산**(고정 M1=128 = 적응 수식의 특수해) + best τ/k.
- (c) best τ/k로 **M2=32/128 확장** (단계적).

> 만약 detach가 실패해 중단됐으면, 재접속 후 `bash exp_runner/launch_phase1.sh "1 0"`
> 재실행하면 resume으로 이어감(이미 끝난 job은 skip, answers 보존).

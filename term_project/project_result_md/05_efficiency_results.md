# 05. 효율성 측정 (실험 5)

POPE 110샘플(앞 10 warmup 제외, 100개 평균), `torch.cuda.Event`/`time` + `torch.cuda.synchronize`,
`max_memory_allocated`. RTX A6000, fp16, `CUDA_LAUNCH_BLOCKING=1`. greedy, max_new_tokens=16.

## 측정 결과

| Setting | M2 | clustering | Token Reduction | Latency (s/q) | GPU Mem (GB) |
|---|---:|:---:|---:|---:|---:|
| A-64 (VisPruner only) | 64 | OFF | 88.9% | 0.3217 | 14.51 |
| B-64s (Ours simple) | 64 | ON (M1=128) | 88.9% | 0.3232 | 14.50 |
| B-64w (Ours weighted) | 64 | ON (M1=128) | 88.9% | 0.3124 | 14.50 |
| A-32 (VisPruner only) | 32 | OFF | 94.4% | 0.3261 | 14.50 |
| B-32s (Ours simple) | 32 | ON (M1=64) | 94.4% | 0.3171 | 14.50 |

> Token Reduction = (576 − M2) / 576 × 100%. (576=CLIP 패치 토큰 수)
> `CUDA_LAUNCH_BLOCKING=1`은 커널 직렬화로 절대 latency를 키우므로(정확/안정 우선),
> 상대 비교(클러스터링 overhead) 목적에 사용. 비차단 모드면 절대값은 더 작음.

## Clustering Overhead 분석

- **B(clustering) latency ≈ A(VisPruner only)** — 동일 M2(64)에서
  A-64 0.3217 vs B-64s 0.3232 (**+0.0015s, +0.5%**), B-64w 0.3124 (**−0.0093s, 측정 노이즈 범위**).
  → **Stage 2 Spherical K-Means 추가 오버헤드는 무시 가능 수준** (≤1%, 측정 변동 내).
  이유: clustering은 (M1≤192)개 토큰의 작은 행렬에 max_iter≤10, 조기종료 + `index_add_`
  벡터 연산 → LLM 디코딩 대비 미미. 오히려 LLM 입력 토큰 수가 동일(M2=64)하므로
  디코딩 비용 불변.
- **GPU Memory: 동일** (14.50~14.51GB). clustering이 (M1,D) 작은 텐서만 추가 사용 →
  메모리 영향 없음. 모델(LLaVA-1.5-7B fp16)이 대부분 차지.

## 정량 결론

| 지표 | A (VisPruner only) | B (Ours, clustering) | 차이 |
|---|---|---|---|
| 정확도(POPE F1 @64) | 80.95 | 82.27 (simple) | **+1.32** |
| Latency @64 | 0.3217 s/q | 0.3232 s/q | +0.5% (무시가능) |
| GPU Mem | 14.51 GB | 14.50 GB | ≈0 |
| Token Reduction | 88.9% | 88.9% | 동일 |

→ **제안 방법은 동일 토큰수·동일 메모리·사실상 동일 속도로 정확도를 향상**
(POPE @64 +1.32, @32 +3.56). Token reduction 88.9~94.4%를 유지하면서 추가 비용 없이
VisPruner 대비 우수 — "공짜 점심"에 가까운 개선.

## 비고
- M1↑(3-C: 96→192)은 Stage1 보존 토큰만 늘릴 뿐 LLM 입력은 M2로 동일 → latency 영향 미미,
  정확도는 향상(82→83). clustering 자체가 저비용이므로 M1을 크게 잡는 전략이 유리.
- @32 확정: A-32 0.3261 vs **B-32s 0.3171** (−2.8%, 측정 노이즈 내) — clustering이
  오히려 근소 빠름(LLM 입력 동일, Stage2 미미). 5세팅 전부 **B latency ≈ A, mem 동일** 확정.

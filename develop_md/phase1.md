# Phase 1 — Adaptive Selection 수식 구현 + 검증 (Claude Code 지시서)

> **목적**: Stage 1 important 토큰 선택을 두 가지 새 수식으로 구현한다.
> - **(가) Energy**: 누적 [CLS] attention 질량이 τ를 넘는 최소 토큰 수를 important로 (적분 관점)
> - **(나) Statistical**: attention 분포에서 통계적 이상치(μ+kσ 또는 median+k·MAD)를 important로 (전역 통계 관점)
>
> 두 수식은 **M1(Stage1 보존 수)과 r(important 비율)을 이미지마다 자동 결정**한다.
> 기존 고정 방식(`topk`, M1=2×M2, r=0.5)은 **default로 그대로 보존**(회귀 안전).
>
> **이 Phase의 핵심 원칙**: 새 기능은 `--selection_method` 플래그로만 활성화되며,
> default(`topk`)일 때 코드 경로·출력이 기존 VisPruner와 **비트 동일**해야 한다.

---

## 0. 작업 전 필수 — 환경 셋업 (실행 안 하면 추론 자체가 실패)

아래 3개 경로가 옛 위치(`SKKU_Works/...`)를 가리켜 깨져 있다. **코드 수정 전에 먼저 패치할 것.**

### 0-1. 러너 스크립트 경로 (worker.sh, launch.sh)
```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
# TP 변수를 현재 경로로 수정
sed -i 's|/home/jhlee/CLUST_KETI/SKKU_Works/Y1_S1/Advanced_ML_DL/experiments/term_project|/home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project|g' exp_runner/worker.sh exp_runner/launch.sh
# 수정 확인
grep -n "TP=" exp_runner/worker.sh exp_runner/launch.sh
```

### 0-2. 모델 심볼릭 링크 재생성
```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
ln -sfn ../VisPruner_run/models models
ls -la models   # ../VisPruner_run/models 를 가리키는지 확인
```

### 0-3. config.json의 mm_vision_tower 절대경로
```bash
CFG=/home/jhlee/CLUST_KETI/AD_MLDL_termProject/VisPruner_run/models/llava-v1.5-7b/config.json
# 옛 SKKU 경로 → 현재 경로
sed -i 's|/home/jhlee/CLUST_KETI/SKKU_Works/.*/VisPruner_run/models/clip-vit-large-patch14-336|/home/jhlee/CLUST_KETI/AD_MLDL_termProject/VisPruner_run/models/clip-vit-large-patch14-336|g' "$CFG"
grep -n "mm_vision_tower" "$CFG"
```

### 0-4. 데이터 링크 + 패키지 등록 확인
```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
ls -la playground/data/eval   # ../dataset 가리키는지
pip install -e .              # 최초 1회 (이미 했으면 skip)
```

> ⚠️ **dtype 패치는 이미 적용됨**(`builder.py` L155~161). 새 코드가 vision tower와 model 사이
> dtype를 섞지 않도록 주의. fp32↔fp16 혼용 시 무음 손상으로 POPE가 0.72로 떨어진다.

---

## 1. 구현 개요 — 무엇을 바꾸나

### 1-1. 새 하이퍼파라미터 (4 touch-point 체인으로 추가)

| 인자 | 타입 | default | 설명 |
|---|---|---|---|
| `--selection_method` | str | `"topk"` | `topk`(기존) / `energy`(가) / `statistical`(나) |
| `--energy_tau` | float | `0.5` | (가) 누적 attention 질량 임계 τ |
| `--stat_k` | float | `2.0` | (나) 이상치 임계 계수 k (μ+kσ) |
| `--stat_robust` | flag(bool) | `False` | (나) True면 median+k·MAD (robust 버전) |

> **default `topk` = 기존 동작 100% 보존.** energy/statistical일 때만 새 분기 진입.

### 1-2. M1·r 자동 결정 규칙 (energy / statistical 공통)

새 수식은 **important 개수 n_imp를 먼저 자동 계산**하고, 나머지를 다음 규칙으로 채운다:

- `n_imp` = (가) 또는 (나) 수식으로 결정 (이미지별 가변)
- `M1 = clamp(round(n_imp / r_floor), M2, M1_cap)` where `r_floor = 0.5`, `M1_cap = 384`
  - 즉 n_imp를 important 절반으로 보고 M1 산정, 단 **하한 M2**(병합 토큰 부족 방지), **상한 384**(노이즈 유입 방지)
- `n_div = M1 - n_imp`
- `r = n_imp / M1` (자동 계산되어 결과에 기록)

> **floor/cap 이유**:
> - 쉬운 이미지에서 n_imp가 0~몇 개로 작아지면 k-means에 토큰 부족 → `M1 ≥ M2` 강제
> - 복잡한 이미지에서 n_imp가 폭증하면 배경·노이즈 과다 → `M1 ≤ 384`(Table 4에서 6× 이상은 성능 하락 확인됨)

---

## 2. 코드 수정 — 정확한 좌표

### 2-1. CLI 인자 등록
**파일**: `llava/eval/model_vqa_loader.py` (argparse, L168~175 부근)
그리고 **동일하게** `llava/eval/model_vqa_science.py`에도 추가.

```python
# 기존 인자들 아래에 추가
parser.add_argument("--selection_method", type=str, default="topk",
                    choices=["topk", "energy", "statistical"])
parser.add_argument("--energy_tau", type=float, default=0.5)
parser.add_argument("--stat_k", type=float, default=2.0)
parser.add_argument("--stat_robust", action="store_true", default=False)
```

### 2-2. 모델 생성에 전달
**파일**: `llava/eval/model_vqa_loader.py` L86~92 (그리고 `model_vqa_science.py` 동일 지점)

```python
load_pretrained_model(
    ...,  # 기존 kwargs 유지
    selection_method=args.selection_method,
    energy_tau=args.energy_tau,
    stat_k=args.stat_k,
    stat_robust=args.stat_robust,
)
```
> `builder.py`는 `**kwargs` 통로라 **수정 불필요**.

### 2-3. 모델에 저장 + getter
**파일**: `llava/model/language_model/llava_llama.py` `__init__`(L44~65) + getter(L71~89)

```python
def __init__(self, config, visual_token_num, important_ratio,
             enable_clustering=False, stage1_tokens=None,
             merge_method="simple_avg", kmeans_max_iter=10,
             selection_method="topk", energy_tau=0.5,
             stat_k=2.0, stat_robust=False):   # ← 추가
    ...
    self.selection_method = selection_method
    self.energy_tau = energy_tau
    self.stat_k = stat_k
    self.stat_robust = stat_robust

# getter (getattr+default = 구버전 체크포인트 안전)
def get_selection_method(self): return getattr(self, "selection_method", "topk")
def get_energy_tau(self):       return getattr(self, "energy_tau", 0.5)
def get_stat_k(self):           return getattr(self, "stat_k", 2.0)
def get_stat_robust(self):      return getattr(self, "stat_robust", False)
```

### 2-4. ★ 핵심 — important 선택 로직 분기
**파일**: `llava/model/llava_arch.py` `encode_images()` L158~162

**현재 코드 (L158~162):**
```python
image_attentions = image_attentions.mean(dim=1)             # (B, N)  saliency 점수
token_indices = image_attentions.argsort(dim=-1, descending=True)
important_indices = token_indices[:, :important_token_num]   # top-(M1·r)
residual_indices  = token_indices[:, important_token_num:]
```

**수정 후:** `image_attentions.mean(dim=1)`로 점수 벡터 `s`(B,N)를 얻은 뒤,
`selection_method`에 따라 **`important_token_num`(=n_imp)과 M1을 분기 계산**한다.
아래 헬퍼를 `llava_arch.py` 상단(또는 별도 모듈 `llava/model/adaptive_selection.py`)에 추가:

```python
@torch.no_grad()
def compute_adaptive_counts(s, method, M2, energy_tau=0.5, stat_k=2.0,
                            stat_robust=False, r_floor=0.5, M1_cap=384):
    """
    s: (B, N) [CLS] attention 점수 (head 평균, 정규화 전).
    반환: n_imp_list (B,), M1_list (B,)  — 이미지별 정수.
    energy/statistical 모두 n_imp만 정하고 동일 규칙으로 M1 유도.
    """
    B, N = s.shape
    n_imp_list, M1_list = [], []
    for b in range(B):
        sb = s[b]
        if method == "energy":
            # (가) 내림차순 누적이 tau 넘는 최소 개수
            sorted_s, _ = torch.sort(sb, descending=True)
            p = sorted_s / (sorted_s.sum() + 1e-8)     # 질량 정규화
            csum = torch.cumsum(p, dim=0)
            n_imp = int((csum < energy_tau).sum().item()) + 1
        elif method == "statistical":
            # (나) 이상치: mu+k*sigma  또는  median+k*MAD
            if stat_robust:
                med = sb.median()
                mad = (sb - med).abs().median()
                thr = med + stat_k * mad
            else:
                thr = sb.mean() + stat_k * sb.std()
            n_imp = int((sb >= thr).sum().item())
        else:
            raise ValueError(method)
        # 공통 floor/cap → M1 유도
        n_imp = max(1, min(n_imp, N))
        M1 = int(round(n_imp / r_floor))
        M1 = max(M2, min(M1, M1_cap, N))
        n_imp = min(n_imp, M1)                          # n_imp가 M1 넘지 않도록
        n_imp_list.append(n_imp); M1_list.append(M1)
    return n_imp_list, M1_list
```

**그리고 L158~162를 다음으로 교체:**
```python
s = image_attentions.mean(dim=1)            # (B, N)  ← 이름은 image_attentions 유지해도 됨
selection_method = self.get_selection_method()

if selection_method == "topk":
    # ===== 기존 경로 (회귀 안전: 한 글자도 바꾸지 말 것) =====
    image_attentions = s
    token_indices = image_attentions.argsort(dim=-1, descending=True)
    important_indices = token_indices[:, :important_token_num]
    residual_indices  = token_indices[:, important_token_num:]
else:
    # ===== (가)energy / (나)statistical : 이미지별 가변 n_imp, M1 =====
    image_attentions = s
    n_imp_list, M1_list = compute_adaptive_counts(
        s, selection_method, M2=visual_token_num,
        energy_tau=self.get_energy_tau(), stat_k=self.get_stat_k(),
        stat_robust=self.get_stat_robust())
    # 배치 내 이미지마다 n_imp/M1이 다르므로 per-image 처리 필요.
    # → 아래 §2-5의 per-image 분기를 따른다.
```

> ⚠️ **중요 — 배치 가변 길이 문제**: 기존 코드는 배치 전체가 같은 M1·n_imp라서 텐서 연산이
> 깔끔했다. 새 수식은 **이미지마다 M1·n_imp가 달라** 배치 일괄 텐서 처리가 안 된다.
> **가장 안전한 구현**: 평가 추론은 batch_size=1이므로(model_vqa_loader 확인할 것),
> B=1 가정으로 단순화하되, B>1이면 per-image 루프로 처리하고 마지막에 list 반환.
> **먼저 `model_vqa_loader.py`의 실제 batch_size를 확인**하고, 1이면 루프 없이 `[0]`만 쓴다.

### 2-5. diverse 및 Stage2 연결
- diverse 선택(L164~191)은 **로직 그대로 두되**, `diverse_token_num = M1 - n_imp`를
  이미지별 값으로 주입. residual에서 그만큼 보존.
- Stage2(L196~213)는 변경 없음. 단 `selected_indices`가 이미지별 M1 길이라
  per-image 처리 경로를 따르는지 확인(이미 L202~210이 이미지별 루프이므로 호환).
- **r 자동값 기록**: 결과 출력을 위해 배치 평균 `r = mean(n_imp/M1)`, 평균 M1을
  로깅 가능하도록 반환 경로에 추가(§4 참조).

---

## 3. 회귀 안전 검증 (구현 직후 반드시)

새 코드가 기존 결과를 안 깨는지 **먼저 확인**한 뒤 새 실험으로 넘어간다.

```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project

# (1) topk default가 기존 VisPruner-only와 비트 동일한지 — POPE M2=128
#     기존 A-128 결과(results.tsv)와 대조. 동일해야 함.
bash exp_runner/worker.sh 0 <(echo -e "RT-A128\tpope\t128\t0\t128\tnone\t0.5")
# → results.tsv의 RT-A128 AvgF1 이 기존 A-128과 동일(±0.0001)인지 확인

# (2) topk + clustering ON 도 기존 B 결과와 동일한지 — POPE M2=64
bash exp_runner/worker.sh 0 <(echo -e "RT-B64s\tpope\t64\t1\t128\tsimple_avg\t0.5")
# → 기존 B-64s 와 동일한지 확인
```

> **통과 기준**: selection_method 미지정(=topk default)일 때 위 두 job이 기존 수치와
> 일치. 불일치 시 §2-4 교체에서 기존 경로를 건드린 것이므로 롤백.

---

## 4. 결과 출력 — results.tsv 스키마 확장

새 수식은 M1·r이 **자동 결정**되므로, job에 적은 값이 아니라 **실제 사용된 평균값**을 기록해야 한다.

### 4-1. worker.sh CLI 매핑 확장
`CL_ARGS` 생성부에 selection 인자 추가:
```bash
SEL_ARGS=""
if [ -n "$SELMETHOD" ] && [ "$SELMETHOD" != "topk" ]; then
  SEL_ARGS="--selection_method $SELMETHOD --energy_tau $ETAU --stat_k $SK"
  [ "$SROBUST" = "1" ] && SEL_ARGS="$SEL_ARGS --stat_robust"
fi
# python 호출에 $SEL_ARGS 추가
```

### 4-2. job tsv 컬럼 확장 (exp_jobs.tsv)
기존 `ID BENCH M2 CLUST M1 METHOD R` 뒤에 `SELMETHOD ETAU SK SROBUST` 4컬럼 추가.
topk job은 뒤 4칸을 `topk 0.5 2.0 0`으로 채우면 기존과 동일 동작.

### 4-3. results.tsv 출력 확장
기존 `ID BENCH M2 CLUST M1 METHOD R METRIC VALUE GEN/TOT` 에
`SELMETHOD AVG_M1 AVG_R` 3컬럼 추가. AVG_M1/AVG_R은 추론 중 집계한 실제 평균
(energy/statistical일 때 의미 있음, topk면 job값과 동일).

> 집계 방법: encode_images에서 배치별 n_imp/M1을 누적 → 추론 종료 시
> `model_vqa_loader.py`가 평균 계산해 answers 파일 헤더 또는 별도 `.meta` 파일로 출력,
> worker.sh가 읽어 results.tsv에 기록. (가장 단순: 모델에 누적 버퍼 attribute 두고
> 추론 끝나고 평균 출력.)

---

## 5. E2 실험 — 수식 검증 (가/나/고정 비교)

회귀 검증 통과 후 실행. **목적: M1·r 자동 수식이 임의 고정값 대비 타당함 + 두 수식 비교 + 고정=특수해 확인.**

### 5-1. exp_jobs_phase1.tsv (신규 job 파일)
컬럼: `ID BENCH M2 CLUST M1 METHOD R SELMETHOD ETAU SK SROBUST`

```
# --- baseline 고정 (대조군) ---
P1-fix-64s   pope  64  1  128  simple_avg    0.5  topk        0.5  2.0  0
P1-fix-64w   pope  64  1  128  weighted_avg  0.5  topk        0.5  2.0  0
# --- (가) energy: tau 스윕 ---
P1-en50-64s  pope  64  1  128  simple_avg    0.5  energy      0.5  2.0  0
P1-en70-64s  pope  64  1  128  simple_avg    0.5  energy      0.7  2.0  0
P1-en80-64s  pope  64  1  128  simple_avg    0.5  energy      0.8  2.0  0
P1-en90-64s  pope  64  1  128  simple_avg    0.5  energy      0.9  2.0  0
# --- (나) statistical: k 스윕 (mu+k*sigma) ---
P1-st15-64s  pope  64  1  128  simple_avg    0.5  statistical 0.5  1.5  0
P1-st20-64s  pope  64  1  128  simple_avg    0.5  statistical 0.5  2.0  0
P1-st25-64s  pope  64  1  128  simple_avg    0.5  statistical 0.5  2.5  0
# --- (나) statistical robust: median+k*MAD ---
P1-st20r-64s pope  64  1  128  simple_avg    0.5  statistical 0.5  2.0  1
```

같은 매트릭스를 **gqa**에도 복제(BENCH만 변경). 두 벤치가 핵심(POPE=hallucination, GQA=compositional).
여유 되면 textvqa/vqav2/sqa까지 확장(OCR은 하락 예상 — 분석용).

### 5-2. 실행
```bash
cd /home/jhlee/CLUST_KETI/AD_MLDL_termProject/term_project
bash exp_runner/launch.sh exp_runner/exp_jobs_phase1.tsv   # 3-GPU 병렬
# 또는 단일 GPU: bash exp_runner/worker.sh 0 exp_runner/exp_jobs_phase1.tsv
```

### 5-3. τ\* / k\* 역산 (고정=특수해 증명)
별도 분석 스크립트 `exp_runner/analyze_special_case.py` 작성:
```
목적: energy/statistical의 AVG_M1이 고정값(128)과 같아지는 tau*/k* 를 찾는다.
- results.tsv에서 SELMETHOD=energy 행들의 (ETAU, AVG_M1) 쌍 수집
- AVG_M1=128이 되는 tau* 를 선형 보간으로 추정
- 동일하게 statistical의 k* 추정
- 출력: "tau*=0.XX 에서 평균 M1=128 → 고정 M1=128은 energy(tau*)의 특수해"
```

---

## 6. 완료 기준 (이 Phase의 done 조건)

1. ✅ §0 경로 패치 3종 완료, 추론 정상 실행
2. ✅ `--selection_method {topk,energy,statistical}` + 부속 인자 4 touch-point로 추가
3. ✅ §3 회귀 검증 통과 (topk default = 기존 VisPruner 비트 동일)
4. ✅ energy/statistical이 이미지별 M1·r 자동 결정, floor/cap 동작
5. ✅ results.tsv에 SELMETHOD/AVG_M1/AVG_R 기록
6. ✅ E2 매트릭스(POPE+GQA) 완주, τ\*/k\* 역산으로 "고정=특수해" 확인

---

## 7. 함정 체크 (Phase 1 한정)

- **batch_size 확인 우선**: §2-4 가변 길이 처리는 batch_size=1이면 단순. 먼저 확인.
- **weighted_avg 가중치 재사용**: §2-4에서 `image_attentions`를 점수로 쓰는데, Stage2
  weighted의 `attn_b`(L205)도 같은 벡터 재사용. energy/statistical은 점수 자체를 안 바꾸고
  "개수만" 바꾸므로 weighted 가중치는 영향 없음(안전). 단 향후 점수 정의를 바꾸면 분리할 것.
- **floor/cap 로깅**: cap(384)이나 floor(M2)에 걸리는 이미지 비율을 로깅하면
  "수식이 실제로 적응하는지 vs 항상 cap에 걸리는지" 진단 가능. 디버깅에 유용.
- **transformers 4.37.2 고정** — 시각 토큰 단(CLIP~projector)에서만 작업. LLM 디코더 미개조.
- **회귀 먼저, 실험 나중** — §3 통과 전에 §5 돌리지 말 것.

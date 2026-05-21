# [개발 프롬프트] Two-Stage Visual Token Reduction Framework 구현

## 목표
VisPruner 코드를 기반으로 Two-Stage Visual Token Reduction Framework을 구현한다.

---

## 디렉토리 규칙
- 구현 코드: `experiments/term_project/` 안에 작성
- VisPruner 원본 코드: `experiments/VisPruner/` → 필요한 파일을 term_project/로 **전부 복사**해서 사용
- PACT 참고 코드: `experiments/PACT/` → clustering 구현 참고용
- 기존 재현 환경: `experiments/VisPruner_run/` → 모델 경로, builder.py dtype 패치 등 참고
- 데이터셋: `experiments/dataset/` (pope, gqa, textvqa 등)
- VisPruner 원본 디렉토리(`experiments/VisPruner/`)는 직접 수정하지 않는다

## 하드웨어/소프트웨어
- GPU: NVIDIA RTX A6000 (48GB)
- Python 3.10, PyTorch 2.1.2+cu121, transformers 4.37.2
- 모델: LLaVA-1.5-7B (liuhaotian/llava-v1.5-7b) + CLIP-ViT-L/14-336px — 이미 로컬에 있음
- builder.py dtype 패치 필수 적용 (VisPruner_run 참고)
- CUDA_LAUNCH_BLOCKING=1 안정모드 적용

---

## 구현할 알고리즘 상세

### 전체 파이프라인

```
입력 이미지
    ↓
CLIP-ViT-L/14 (336px) → 576개 visual tokens 생성
    ↓
[Stage 1] VisPruner 기반 토큰 선택
    - [CLS] attention으로 important tokens 선택 (M1 × r 개)
    - cosine similarity로 diverse tokens 선택 (M1 × (1-r) 개)
    - 합계: M1개 토큰 보존 (M1 > 최종 목표 M2)
    ↓
[Stage 2] Spherical K-Means Clustering 기반 토큰 병합
    - M1개 토큰을 M2개 클러스터로 그룹화
    - 각 클러스터의 대표 토큰 생성 (simple avg 또는 weighted avg)
    - 결과: M2개 merged tokens
    ↓
LLM (Vicuna-7B) → 답변 생성
```

### Stage 1: VisPruner 기반 토큰 선택 (기존 코드 활용 + 수정)

VisPruner 원본 코드의 동작을 그대로 사용하되, 출력 토큰 수를 조정 가능하게 한다.

**알고리즘 상세:**

1. **[CLS] Attention 추출**: CLIP 비전 인코더의 **뒤에서 2번째 레이어**(select_layer=-2)에서 `[CLS]→patch` attention을 추출한다. 구체적으로 attention matrix의 첫 번째 행(`attentions[:, :, 0, 1:]`)을 가져온 뒤 head 차원을 평균낸다. 결과: `attn_scores` (576,) 크기의 1D 텐서.

2. **Important Token 선택**: `attn_scores`를 내림차순 정렬하여 상위 `T_imp = M1 × r`개 토큰의 인덱스를 선택한다. 이 토큰들은 visual encoder가 가장 주목하는 전경 객체/핵심 영역에 대응한다.

3. **Diverse Token 선택**: 나머지 `576 - T_imp`개 토큰 중에서, 이미 선택된 토큰과 cosine similarity가 높은(=중복인) 토큰을 반복적으로 제거한다. ToMe 스타일의 bipartite matching으로 매 반복마다 가장 유사한 토큰 쌍을 찾아 한쪽을 제거. `T_div = M1 × (1 - r)`개가 남을 때까지 반복한다. 이 토큰들은 배경/맥락 정보를 보완한다.

4. **결합**: `important(T_imp개) + diverse(T_div개) = M1개` 토큰을 원래 이미지 패치 순서대로 정렬하여 반환한다.

**기존 VisPruner 대비 수정 사항:**
- 기존: M1 = 최종 토큰 수 (프루닝 끝)
- 수정: M1을 최종 목표 M2보다 크게 설정 가능 → 2단계에서 추가 압축
- 1단계 출력 시 **각 토큰의 원래 attention score도 함께 반환** (2단계 weighted avg에서 사용)

### Stage 2: Spherical K-Means Clustering 기반 토큰 병합 (신규 구현)

1단계에서 보존된 M1개 토큰 중에서도 여전히 의미적으로 유사한 토큰이 존재할 수 있다. 이를 클러스터링으로 병합하여 동일 토큰 수 대비 더 높은 정보 밀도를 달성한다.

**Spherical K-Means 알고리즘:**

```
입력: tokens (M1, D), k = M2 (목표 클러스터 수), max_iter
출력: centroids (M2, D)

1. tokens를 L2 정규화 → 단위 구(unit sphere) 위로 투영
2. 초기 centroid: tokens에서 k개를 랜덤 선택 (또는 attention score 상위 k개)
3. for iter in range(max_iter):
     a. 각 토큰과 모든 centroid의 cosine similarity 계산
     b. 각 토큰을 가장 유사한 centroid에 할당 (assignment)
     c. 각 클러스터 내 토큰들의 평균 벡터 계산 → 새 centroid
     d. 새 centroid를 L2 정규화
     e. assignment가 변하지 않으면 조기 종료
4. centroids 반환
```

**대표 토큰 생성 (2가지 방식):**

(a) **Simple Average**: 클러스터에 할당된 토큰들의 단순 평균
```
representative_token[j] = mean(tokens[i] for i in cluster_j)
```

(b) **Weighted Average**: 1단계에서 가져온 attention score를 가중치로 사용
```
representative_token[j] = sum(attn[i] * tokens[i] for i in cluster_j) / sum(attn[i] for i in cluster_j)
```
→ 중요도 높은 토큰의 정보가 대표 토큰에 더 많이 반영됨

**구현 시 주의:**
- 모든 연산은 PyTorch GPU 텐서로 수행 (numpy 변환 불필요)
- fp16 환경: L2 norm 계산 시 `eps=1e-8` 추가하여 0 나눗셈 방지
- max_iter = 10 이하 (추론 시간 overhead 최소화)
- 빈 클러스터 처리: 빈 클러스터가 발생하면 가장 큰 클러스터에서 토큰 하나를 재할당

### VLM 파이프라인 통합

VisPruner의 `llava_arch.py`에서 토큰 선택이 끝난 직후에 clustering을 호출한다:

```python
# 기존 VisPruner 흐름
important_tokens, diverse_tokens, attn_scores_retained = vispruner_select(
    visual_tokens, attn_scores, M1, r
)
retained_tokens = concat_and_sort(important_tokens, diverse_tokens)  # (M1, D)

# ★ 2단계: clustering (enable_clustering=True이고 M2 < M1일 때만)
if enable_clustering and M2 < M1:
    merged_tokens = merge_tokens(
        retained_tokens, attn_scores_retained, M2, merge_method
    )
    final_tokens = merged_tokens  # (M2, D)
else:
    final_tokens = retained_tokens  # (M1, D) — 기존 VisPruner 동작

# final_tokens를 LLM 입력의 <image> 자리에 삽입
```

### 추가할 커맨드라인 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--enable_clustering` | flag | False | 2단계 clustering 활성화 |
| `--stage1_tokens` | int | (visual_token_num과 동일) | 1단계 보존 토큰 수 M1 |
| `--visual_token_num` | int | 576 | 최종 토큰 수 M2 (기존 인자) |
| `--merge_method` | str | "simple_avg" | "simple_avg" 또는 "weighted_avg" |
| `--kmeans_max_iter` | int | 10 | spherical k-means 최대 반복 |
| `--important_ratio` | float | 0.5 | important token 비율 r (기존 인자) |

**핵심 동작 분기:**
- `enable_clustering=False` → 기존 VisPruner와 완전히 동일 (Stage 1만 수행)
- `enable_clustering=True` → Stage 1(M1개) → Stage 2(M2개) 순차 수행

---

## 구현 순서

### Step 1: 환경 구성
1. VisPruner 코드를 term_project/로 복사
2. VisPruner_run에서 적용된 패치(builder.py dtype, 기타) 반영
3. 모델 경로 설정 (VisPruner_run과 동일)
4. `pip install -e .` 등 필요한 설치 수행
5. 데이터셋 경로가 `experiments/dataset/`을 가리키도록 설정 (심볼릭 링크 가능)

### Step 2: Clustering 모듈 구현
1. `spherical_kmeans.py` 또는 적절한 위치에 Spherical K-Means 구현
2. `merge_tokens()` 함수 구현 (simple_avg + weighted_avg)
3. 단위 테스트: 임의 텐서 (128, 1024)로 k=64 clustering이 정상 동작하는지 확인

### Step 3: 파이프라인 통합
1. `llava_arch.py`에 clustering 호출 지점 삽입
2. 커맨드라인 인자 추가
3. 1단계에서 attention score를 함께 반환하도록 수정

### Step 4: Sanity Check (3개 모두 통과 필수)

**(1) VisPruner only 검증**
```bash
# enable_clustering 없이 실행 → 기존 VisPruner와 동일해야 함
# POPE 300개 샘플, 64토큰, r=0.5
# 기대: 기존 재현 결과(F1 ~80.9)와 거의 동일
```

**(2) Two-stage simple avg 검증**
```bash
# enable_clustering, stage1_tokens=128, visual_token_num=64, merge_method=simple_avg
# POPE 300개 샘플
# 기대: 에러 없이 완료, 합리적 정확도(60~85 범위)
```

**(3) Two-stage weighted avg 검증**
```bash
# enable_clustering, stage1_tokens=128, visual_token_num=64, merge_method=weighted_avg
# POPE 300개 샘플
# 기대: 에러 없이 완료, 합리적 정확도
```

---

## 에러 대응
- 에러 발생 시 원인을 분석하고 스스로 수정하여 정상 동작할 때까지 반복한다
- CUDA 에러: CUDA_LAUNCH_BLOCKING=1 적용, dtype 확인
- 빈 클러스터: 재할당 로직으로 처리
- 메모리 부족: stage1_tokens 줄이거나 batch size 조정
- 수정 내역은 모두 기록한다

## 산출물
구현 완료 후 `experiments/test_result_md/01_implementation_report.md` 작성:
- 구현한 코드 구조 (파일 목록 + 각 역할)
- VisPruner에서 수정/추가한 부분 상세 설명
- Spherical K-Means 구현 세부사항
- 추가한 커맨드라인 인자 정리
- Sanity check 3개 결과 (정확한 수치)
- 에러 발생 및 해결 과정

**Sanity check 3개가 모두 통과하면 "구현 완료"라고 알려줘.**

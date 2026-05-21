## 목표
기존 실험 결과를 바탕으로 최종 종합 보고서 MD를 새로 작성한다.
기존 test_result_md/의 결과 데이터를 모두 활용하되, 아래 요구사항에 맞게 훨씬 자세하고 체계적으로 정리한다.
결과 파일: experiments/test_result_md/final_report.md

## 보고서 구성 및 요구사항

### 1. 프로젝트 개요
- 연구 배경: VLM에서 visual token이 왜 문제인지 (576토큰, quadratic complexity)
- 기존 방법의 한계: pruning only(VisPruner)는 잔여 중복 미처리
- 제안 방법 요약: Two-Stage(VisPruner + Spherical K-Means) 한 문단 설명
- 실험 목적: 제안 방법이 동일 토큰수에서 정보 밀도를 높이는지 검증

### 2. 제안 방법 상세 설명

#### 2-1. 전체 파이프라인
입력 이미지 → CLIP-ViT-L/14 → 576토큰 → Stage1 → Stage2 → LLM
각 단계가 무엇을 하는지 한국어로 자세히 설명.

#### 2-2. Stage 1: VisPruner 기반 토큰 선택
- [CLS] attention이 무엇인지, 왜 이것을 쓰는지 (text-visual attention의 position bias 문제)
- Important token: attention 상위 토큰, 전경/핵심 객체 대응
- Diverse token: cosine similarity 중복 제거, 배경/맥락 정보 보완
- Important ratio r의 의미와 역할

#### 2-3. Stage 2: Clustering 기반 토큰 병합
- Spherical K-Means가 무엇인지 (일반 K-Means와 차이: cosine similarity 기반)
- 왜 Stage 1 후에 추가 병합이 필요한지 (잔여 중복 토큰 압축)
- M1(Stage1 출력) → M2(최종) 압축 과정 설명

#### 2-4. Simple Average vs Weighted Average 상세 비교
이 부분을 특히 자세하게 써줘:
- Simple Average: 클러스터 내 토큰의 단순 평균. 모든 토큰을 동등하게 취급.
  - 장점: 구현 단순, 배경 정보 균등 반영
  - 단점: 중요하지 않은 토큰 정보가 희석 없이 섞임
- Weighted Average: [CLS] attention score를 가중치로 사용한 가중 평균. 
  중요도 높은 토큰의 정보가 대표 토큰에 더 많이 반영됨.
  - 장점: 핵심 시각 정보 보존, 어려운 태스크(counting, adversarial)에서 유리
  - 단점: attention score가 부정확하면 오히려 편향
- 수식도 포함:
  - simple: rep_j = (1/|C_j|) × Σ x_i (i ∈ C_j)
  - weighted: rep_j = Σ(a_i × x_i) / Σ(a_i) (i ∈ C_j, a_i = attention score)
- 실험 결과 기반 비교: 어떤 벤치마크에서 어느 쪽이 좋았는지 정리

### 3. 실험 환경
- GPU, CUDA, Python, PyTorch, transformers 버전
- 모델 정보 (LLaVA-1.5-7B 구성: Vicuna-7B + CLIP-ViT-L/14-336px)
- 추론 설정 (fp16, greedy, temperature=0)
- 안정화 조치 (CUDA_LAUNCH_BLOCKING, dtype 패치)

### 4. 데이터셋 상세 설명
각 데이터셋마다 아래 항목을 모두 포함하여 표+설명으로 정리:

#### POPE (Polling-based Object Probing Evaluation)
- 태스크 유형: 객체 환각 평가 (yes/no 이진 분류)
- 질문 형태: "Is there a [object] in the image?" → yes/no
- 총 문항: 8910문항
- 카테고리: random(2910), popular(3000), adversarial(3000)
- 카테고리 설명: random=무작위 객체, popular=빈출 객체, adversarial=동시출현 객체(가장 어려움)
- 이미지 출처: COCO val2014
- 평가 지표: F1 Score (카테고리별 + 평균)
- 이 데이터셋이 왜 중요한지: VLM의 환각 문제 평가, diverse token 보존 효과 확인

#### GQA
- 태스크 유형: 구조적 시각 추론 (open-ended VQA)
- 질문 형태: scene graph 기반 생성 질문 (공간관계, 속성, 비교 등)
- 총 문항: 12578 (testdev balanced)
- 이미지 출처: Visual Genome
- 평가 지표: Accuracy
- 특징: 공간관계/속성 추론 → 조밀한 시각 정보 필요, 토큰 감소에 민감

#### TextVQA
- 태스크 유형: 이미지 내 텍스트 인식 기반 VQA
- 질문 형태: 이미지 속 간판/라벨/텍스트를 읽어야 답할 수 있는 질문
- 총 문항: 5000 (val set)
- 이미지 출처: Open Images v3
- 평가 지표: VQA Accuracy
- 특징: OCR 능력 필요, 세밀한 텍스트 정보가 병합 시 손실될 수 있음

#### VQAv2
- 태스크 유형: 범용 시각 질의응답
- 질문 형태: 이미지에 대한 다양한 open-ended 질문
- 사용 데이터: val 균형 subset 6000 (yes-no/number/other 각 2000)
- Question type 설명:
  - yes/no: 객체 존재 확인 → 소수 토큰으로 충분, 압축에 강건
  - number: counting → 세밀한 공간 정보 필요, 압축에 가장 취약
  - other: 다양한 시각 추론 → 중간 난이도
- 이미지 출처: COCO
- 평가 지표: VQA Accuracy (공식 metric)
- 특징: question type별 분석으로 task-aware policy 근거 제공
- 주의: val subset이므로 논문 test-dev 인용값과 절대 비교 불가, A↔B 상대 비교만 유효

#### SQA-IMG (ScienceQA Image subset)
- 태스크 유형: 과학 주제 multiple-choice QA (이미지 포함)
- 질문 형태: 자연과학/사회과학/언어과학 multiple-choice
- 총 문항: 2017 (이미지 포함 test split)
- 이미지 출처: ScienceQA dataset
- 평가 지표: Accuracy
- 특징: 상식/추론 중심 → 토큰수에 둔감, clustering 효과 제한적

### 5. 실험 결과 — 메인 비교 (실험 1)

#### 5-1. 전체 결과표
9세팅 × 5벤치마크 표를 깔끔하게 정리.
각 토큰수(128/64/32)별로 최고 성능에 **bold** 표시.

#### 5-2. 개선폭 분석 (B − A)
벤치마크별 개선폭을 표로 정리하고, 패턴을 설명:
- "압축 강할수록 clustering 이득 증가" 패턴이 어떤 벤치마크에서 나타나는지
- TextVQA가 왜 예외인지 (OCR 특성 → 병합 시 텍스트 정보 손실)
- SQA-IMG가 왜 둔감한지 (상식 추론 → 소수 토큰 충분)

#### 5-3. 벤치마크별 심층 분석
각 벤치마크마다 2~3문장으로:
- 이 벤치마크에서 제안 방법이 효과적인 이유/비효과적인 이유
- 어떤 병합 방식(simple/weighted)이 더 좋았는지와 그 이유

#### 5-4. 최종 권장 설정
전체 결과를 종합하여 "어떤 상황에서 어떤 설정을 쓰면 좋은지" 정리:
- 범용: weighted_avg, r=0.5, M1=2×M2
- 환각 평가: simple_avg, r=0.3 (diverse↑)
- OCR 태스크: 보수적 압축 또는 clustering OFF
- counting: 보수적 M2 (64 이상 유지)

### 6. 실험 결과 — 기존 Baseline 비교 (실험 2)
POPE, GQA 기준으로 FastV/ToMe/SparseVLM/VisPruner/Ours 종합표.
각 baseline이 어떤 방법인지 한 줄 설명 포함:
- FastV: LLM layer 2 이후 text-visual attention 기반 프루닝
- ToMe: bipartite soft matching으로 유사 토큰 쌍 병합
- SparseVLM: text relevance 기반 sparsification + 정보 재활용
- VisPruner: [CLS] attention 기반 중요 토큰 선택 + 유사도 기반 다양성 보존
- Ours: VisPruner + Spherical K-Means clustering 병합
인용/실행 출처도 명시.

### 7. 실험 결과 — Ablation Study (실험 3)

#### 7-1. Important/Diverse Ratio (r=0.3/0.5/0.7)
결과표 + 해석: POPE에서 r↓(diverse↑)가 유리한 이유, GQA에서 무감각한 이유

#### 7-2. Clustering 유무 비교
결과표 + 해석: 동일 토큰수에서 ON이 항상 우세인 이유, 32토큰에서 효과 극대인 이유

#### 7-3. M1 민감도 (1.5×/2×/3×)
결과표 + 해석: M1↑이 왜 좋은지 (Stage2가 선택할 풀이 넓어짐)

#### 7-4. Simple vs Weighted 종합 비교
5개 벤치마크 × 3토큰수에서 어느 쪽이 이겼는지 승패표로 정리.
결론: 어떤 상황에서 어느 방식을 권장하는지.

### 8. 실험 결과 — Question-Type 분석 (실험 4)

#### 8-1. POPE 카테고리별
random/popular/adversarial 결과표.
adversarial에서 clustering이 특히 효과적인 이유: diverse 토큰이 배경 맥락을 보존하여 환각 거부에 기여.

#### 8-2. VQAv2 Question Type별
yes-no/number/other 결과표.
number가 압축에 가장 취약한 이유: counting은 세밀한 공간 정보(객체 위치, 개수)에 의존.
clustering이 number/other를 회복하는 이유: 병합이 정보 밀도를 높여 손실된 공간 정보를 부분적으로 보상.

#### 8-3. Task-Aware Policy 종합 제안
위 분석을 종합하여 question type별 권장 설정:
- yes/no → 공격적 압축(M2=32 OK), simple_avg
- number → 보수적 압축(M2≥64), weighted_avg
- other → 중간 압축, weighted_avg
- adversarial/환각 → r↓(diverse↑), simple_avg

### 9. 실험 결과 — 효율성 (실험 5)
결과표 + 핵심 결론: clustering 오버헤드 ≈ 0인 이유 설명
(작은 텐서, max_iter≤10, 조기종료, LLM 디코딩 대비 미미)

### 10. 한계점 및 향후 연구
- TextVQA(OCR) 하락 원인과 가능한 해결 방향
- Task-aware 자동 라우팅 미구현 (수동 grid search만)
- 단일 백본(LLaVA-1.5-7B)에서만 검증
- 향후: cross-attention 기반 VLM(Flamingo 등), video 입력, 자동 task-aware policy

### 11. 결론
3~4문장으로 핵심 기여와 결과 요약.

## 작성 스타일
- 한국어로 작성 (기술 용어는 영어 병기 가능)
- 표는 마크다운 테이블 사용
- 각 섹션마다 "핵심 발견" 또는 "→" 로 시작하는 요약 한 줄 포함
- 수치는 소수점 2자리까지
- 모든 결과 수치는 기존 test_result_md/의 md 파일과 exp_runner/results.tsv에서 가져올 것
- 날조 금지, 없는 수치는 "미측정"으로 표기
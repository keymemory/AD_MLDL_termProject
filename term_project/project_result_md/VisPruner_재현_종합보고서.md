# VisPruner 재현 실험 종합 보고서

> 논문: **"Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs"** (VisPruner, ICCV 2025, arXiv:2412.01818)
> 목적: 원본 코드를 로컬 환경에서 그대로 실행하여 논문 POPE 실험 결과 재현
> 작성일: 2026-05-16 · 단일 GPU(RTX A6000) 환경

---

## 1. 한눈에 보는 결론

VisPruner 원본 코드를 환경/데이터만 로컬에 맞춰 실행한 결과, **POPE 벤치마크에서 논문값과 0.2~1.5점(F1) 이내로 재현 성공**.

| Setting | 시각 토큰 | 감소율 | 논문 POPE(F1) | **재현 POPE(F1)** | 차이 |
|---|---:|---:|---:|---:|---:|
| Baseline (no prune) | 576 | – | ≈85.9 | **85.87** | ≈0.0 |
| VisPruner | 128 | ↓77.8% | 84.6 | **84.43** | −0.17 |
| VisPruner | 64 | ↓88.9% | 80.4 | **80.91** | +0.51 |
| VisPruner | 32 | ↓94.4% | 72.7 | **74.17** | +1.47 |

→ **토큰을 77.8% 줄여도 F1은 1.4점만 하락**(85.87→84.43). VisPruner의 핵심 주장(visual cue 기반 프루닝이 효과적)이 재현됨.

---

## 2. 사용한 데이터

### 2.1 벤치마크: POPE (Polling-based Object Probing Evaluation)
- 객체 환각(object hallucination) 평가. "이미지에 X가 있나요?"에 yes/no로 답.
- **총 8910문항** (논문/LLaVA 표준과 동일 구성):

| 카테고리 | 문항 수 | 부정 샘플 생성 방식 |
|---|---:|---|
| random | 2910 | 무작위 객체 |
| popular | 3000 | 데이터셋에서 빈출하는 객체 |
| adversarial | 3000 | 대상과 자주 동시출현하는 객체 (가장 어려움) |

### 2.2 이미지: COCO val2014
- POPE가 참조하는 **고유 이미지 500장만** 사용 (전체 4만 장 zip 6.2GB 대신 필요분 79MB만 개별 다운로드 — 디스크 절약, 평가 결과에는 영향 없음).
- 출처: `http://images.cocodataset.org/val2014/`

### 2.3 평가 파일 재구성 (원본 eval.zip 미사용)
원본 EVAL.md는 Google Drive `eval.zip`을 요구하나, 독립 재현을 위해 직접 재구성:
- **annotation**: POPE 공식 repo(`AoiDragon/POPE`, commit `e3e3926`)의 `output/coco/coco_pope_{random,popular,adversarial}.json` raw 다운로드.
- **`llava_pope_test.jsonl`**: 위 3개에서 직접 생성. 전역 유일 `question_id`(1..8910), LLaVA 표준 접미사 `" Answer the question using a single word or phrase."` 부가, `category`/`label` 보존. 생성 순서 = annotation 파일 순서 → `eval_pope.py` 채점 정합성 보장.
- 질문 텍스트·정답 라벨 내용은 원본과 동일. question_id 부여 방식만 다르며 카테고리별 채점이라 결과 무영향.

---

## 3. 사용한 모델

| 구성 | 모델 | 출처 | 용량 |
|---|---|---|---:|
| VLM | **LLaVA-1.5-7B** (Vicuna-7B + CLIP) | `liuhaotian/llava-v1.5-7b` (HF) | 13 GB |
| Vision Encoder | **CLIP ViT-L/14-336px** | `openai/clip-vit-large-patch14-336` (HF) | 1.6 GB |

- LLaVA `config.json`의 `mm_vision_tower`를 **로컬 CLIP 경로로 패치**(Term_project 단독 실행 보장).
- 핵심 모델 설정: `mm_vision_select_layer = -2`, `mm_vision_select_feature = patch`, `image_aspect_ratio = pad`, fp16 추론.

---

## 4. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 49GB (단일, `CUDA_VISIBLE_DEVICES=2`) |
| Driver / CUDA | 535.183.01 / 12.2 |
| Python | 3.10 (conda env `vispruner`) |
| PyTorch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| transformers | 4.37.2 |
| tokenizers / accelerate | 0.15.1 / 0.21.0 |
| 설치 | VisPruner `pyproject.toml` 그대로 `pip install -e .` (원논문 의존성) |

---

## 5. VisPruner 핵심 동작 (재현 대상 알고리즘)

목표 토큰 수 `T`(=visual_token_num), important 비율 `r`(=important_ratio, 기본 0.5).

1. **[CLS] attention 추출** (`clip_encoder.py`): CLIP 비전 인코더의 **뒤에서 2번째 레이어**(`select_layer=-2`)에서 `[CLS]→patch` attention(`attentions[:, :, 0, 1:]`)과 patch feature를 함께 추출.
2. **Important token 선택** (`llava_arch.py`): head 평균낸 [CLS] attention 점수가 높은 순으로 상위 `T_imp = T·r` 개를 무조건 보존(=시각적으로 중요한 토큰).
3. **Diverse token 선택**: 나머지(residual) 토큰들 중 서로 코사인 유사도가 높은(중복) 토큰을 ToMe식 bipartite 매칭으로 반복 제거 → 남는 `T_div = T·(1−r)` 개는 시각적으로 **다양한** 토큰.
4. **통합**: `important + diverse = T` 개만 LLM 입력 `<image>` 자리에 삽입(공간 순서 보존) → LLM 연산량/메모리 절감.

> T=576, r=0.5 → important 288 + residual 288, diverse 루프 즉시 종료 → 576개 전부 보존 = **프루닝 없는 vanilla LLaVA-1.5-7B baseline**.

평가: greedy decoding(temperature=0), `vicuna_v1` 대화 템플릿, 채점은 `eval_pope.py`(카테고리별 Accuracy/F1).

---

## 6. 실험 과정 (중요 트러블슈팅 포함)

### 6.1 절차
코드 사본 구성 → conda 환경+의존성 → 모델/데이터 다운로드 → config 로컬 경로 패치 → sanity(300) → POPE 전체(8910) × {576,128,64,32} → eval_pope 채점 → 문서화.

### 6.2 ⚠️ 결정적 이슈: dtype 혼용으로 인한 무음 시각 피처 손상
- **증상**: 프루닝 없는 576 baseline조차 POPE 정확도 ~0.72로 비정상(모델이 거의 "No"만 답). 논문 ~0.86과 큰 괴리.
- **진단 과정**:
  1. 모델 직접 호출(이미지 묘사)은 **정상**(스노보드 이미지 정확 묘사) → 모델/가중치/이미지 파이프라인 자체는 정상.
  2. 동일 이미지에서 POPE 단답 접미사 + greedy일 때만 오답 "No", 접미사 없으면 정답 "Yes" → 피처 미세 손상 정황.
  3. attention 추출 경로 직접 비교 중 **`CUDA error: an illegal memory access`**(비동기 보고) 포착.
  4. `CUDA_LAUNCH_BLOCKING=1`에서는 무에러 + attention on/off 두 경로 피처 **완전 동일(diff=0)** → 원인을 dtype 혼용으로 특정.
- **원인**: 평가 스크립트가 `device_map='auto'`로 모델 로드 → 원본 `builder.py`는 이 경우 비전 타워를 fp16으로 캐스팅하지 않아 **비전 타워 float32 + 모델/projector float16** 혼용. VisPruner의 attention 추출·argsort·matmul·boolean-mask 인덱싱 커널에서 비동기 메모리 접근 위반 발생, 추론은 완료되나 시각 피처가 조용히 손상.
- **해결**:
  - `builder.py` 패치 — `device_map=='auto'`일 때도 비전 타워를 모델 dtype으로 캐스팅.
  - 잔여 간헐 결함 대비: `CUDA_LAUNCH_BLOCKING=1`(안정모드, 검증 완료) + `model_vqa_loader` **resume**(완료 question_id 건너뜀) + 세팅별 **자동 재시도**.
- **검증**: sanity(300) 정확도 0.717 → **0.837** 정상화. baseline 전량 재생성 시 F1 0.8024 → **0.8587**.

> 이 이슈를 잡지 않았다면 모든 재현값이 논문보다 10점 이상 낮게 나와 "재현 실패"로 잘못 결론지을 수 있었던 핵심 포인트.

---

## 7. 최종 결과 (eval_pope.py 공식 출력)

### 7.1 카테고리별 상세

| n (tokens) | Acc(random) | Acc(popular) | Acc(adv) | **Avg F1** | 비고 |
|---:|---:|---:|---:|---:|---|
| **576** | 0.8814 | 0.8723 | 0.8507 | **0.8587** | baseline, 안정모드 1회 완주 |
| **128** | 0.8687 | 0.8647 | 0.8330 | **0.8443** | 안정모드 1회 완주 |
| **64** | 0.8399 | 0.8397 | 0.8110 | **0.8091** | 안정모드 1회 완주 |
| **32** | 0.7897 | 0.7933 | 0.7687 | **0.7417** | 안정모드 1회 완주 |

(F1: random/popular/adversarial 각각 — 576: 0.873/0.861/0.841 · 128: 0.859/0.851/0.823 · 64: 0.821/0.816/0.790 · 32: 0.751/0.748/0.726)

### 7.2 논문 Table 1 대비 (★ 논문과 직접 비교용)

| Setting | 시각토큰 | 감소율 | 논문 POPE | **재현 POPE(Avg F1)** | 차이 |
|---|---:|---:|---:|---:|---:|
| Baseline | 576 | – | ≈85.9 | **85.87** | ≈0.0 |
| VisPruner | 128 | ↓77.8% | 84.6 | **84.43** | −0.17 |
| VisPruner | 64 | ↓88.9% | 80.4 | **80.91** | +0.51 |
| VisPruner | 32 | ↓94.4% | 72.7 | **74.17** | +1.47 |

추론 환경: 단일 RTX A6000, fp16, GPU 메모리 약 14–15GB, 세팅당 8910문항 ~25분(안정모드).

---

## 8. 분석

- **재현 충실도**: 4개 세팅 모두 논문값과 0.2~1.5점 이내. 원논문 코드가 신뢰성 있게 재현됨.
- **토큰 감소 추세**: 576→128(−77.8%)에서 F1 −1.4점에 불과. 일부 카테고리(random/popular)는 baseline과 거의 동일 — VisPruner가 노이즈/중복 패치를 제거해 LLM이 핵심 시각정보에 집중하게 만든다는 논문 주장과 일치. 32 토큰(−94.4%)에서도 F1 74로 급격한 붕괴 없음(극단 압축 강건성 재현).
- **미세 차이 원인**: 데이터 재구성(question_id 방식만 상이, 내용 동일), transformers/torch 버전 차이에 따른 ±1점 변동(정상 범위). 32 토큰에서 차이(+1.5)가 가장 큰 것은 극단 프루닝일수록 선택 토큰의 미세 수치차가 출력에 민감하기 때문으로 해석.

---

## 9. 산출물 위치

- 본 보고서: `Term_project/VisPruner_재현_종합보고서.md`
- 세부 문서: `Term_project/vispruner_md/01~04_*.md`
- 원시 추론 결과(재채점 가능): `Term_project/VisPruner_run/playground/data/eval/pope/answers/llava_pope_test/llava-v1.5-7b/n_{576,128,64,32}/r_0.5.jsonl` (각 8910줄)
- 독립 실행 환경: `Term_project/VisPruner_run/` (코드+모델+데이터, `bash run_pope_all.sh`로 전체 재현)

## 10. 미수행 항목

- GQA(우선순위 2), VQAv2(우선순위 3): 미진행. `develop_test.md` 지침("POPE 우선, 무거우면 subset")에 따라 POPE **전체 8910 풀셋**을 4세팅 완주하는 데 집중. (디스크 가용 28GB 한계로 GQA/VQAv2 추가 다운로드 곤란.)

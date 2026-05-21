## 목표
VisPruner POPE 재현 완료 상태에서, 논문 Table 1의 나머지 벤치마크들을 추가로 재현한다.
POPE는 이미 완료됐으므로 건너뛴다.

## 현재 상태
- VisPruner 코드: Term_project/VisPruner_run/ 에 세팅 완료
- 모델: LLaVA-1.5-7B, CLIP-ViT-L/14-336px 로컬에 있음
- POPE 실험: 576/128/64/32 토큰 4세팅 완료, 논문과 0.2~1.5점 이내 재현 확인
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- 디스크 여유: ~500GB (용량 제약 없음)
- builder.py dtype 패치, CUDA_LAUNCH_BLOCKING=1 안정모드 이미 적용됨

## 데이터 다운로드 전략

### 핵심 참고자료
VisPruner는 LLaVA-1.5의 eval 체계를 그대로 사용한다.
따라서 아래 두 곳을 반드시 먼저 확인:
1. **VisPruner 코드 내**: `docs/Evaluation.md` 또는 `scripts/v1_5/eval/` 디렉토리의 각 벤치마크 실행 스크립트에서 데이터 경로와 다운로드 방법이 명시되어 있는지 확인
2. **LLaVA 공식 Evaluation 문서**: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md
   - 이 문서에 각 벤치마크별 다운로드 링크와 파일 배치 구조가 정리되어 있음
   - `web_fetch`로 이 페이지를 먼저 읽고, 벤치마크별 정확한 다운로드 URL과 디렉토리 구조를 파악할 것

### 다운로드 방법 우선순위
각 벤치마크 데이터를 구할 때 아래 순서로 시도:
1. **LLaVA Evaluation.md에 명시된 직접 다운로드 링크** (가장 신뢰)
2. **HuggingFace datasets** (`huggingface-cli download` 또는 `wget https://huggingface.co/datasets/...`)
3. **벤치마크 공식 GitHub/웹사이트의 직접 링크** (wget/curl 가능한 것만)
4. **Google Drive 링크** → `pip install gdown` 후 `gdown <file_id>` 시도

⚠️ 아래의 경우 해당 벤치마크를 스킵:
- 이메일 등록/계정 생성이 필요한 경우 (자동화 불가)
- 다운로드 링크가 만료되거나 인증 필요한 경우
- 3가지 이상의 대안 경로를 시도해도 실패한 경우
스킵 시 사유를 반드시 기록하고 다음 벤치마크로 넘어갈 것.

## 추가 벤치마크 목록 (전체 진행)

### [1] SQA-IMG (~3GB)
- ScienceQA 이미지 서브셋, 2017 multiple-choice pairs
- HuggingFace에서 다운로드 가능: `huggingface-cli download derek-thomas/ScienceQA` 또는 LLaVA eval docs 참고
- 평가 지표: accuracy on test split (이미지 포함 질문만)

### [2] GQA (~20GB)
- 12578 image-question pairs (testdev balanced)
- 이미지: https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
- 질문 파일: LLaVA eval docs에서 testdev balanced questions 경로 확인
- 평가 지표: accuracy on testdev

### [3] TextVQA (~8GB)
- 5000 image-question pairs (val set)
- 이미지: LLaVA eval docs에서 TextVQA 이미지 다운로드 경로 확인
   (보통 https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- 평가 지표: accuracy (VQA accuracy metric)

### [4] VQAv2 (~30GB)
- 107394 image-question pairs (test-dev)
- COCO test2015 이미지 필요
- 채점: EvalAI 서버 제출 필요 → 제출 불가 시 val set으로 대체하고 사유 기록

### [5] MMBench + MMBench-CN (~1GB)
- LLaVA eval docs에서 tsv 파일 다운로드 경로 확인
- 평가: 외부 서버 제출 필요할 수 있음 → 추론 결과만 저장하고 제출 가능 여부 확인

### [6] MME (~2GB)
- ⚠️ 이메일 등록 필요 가능성 높음
- LLaVA eval docs에서 다운로드 방법 먼저 확인
- HuggingFace에 미러가 있는지 탐색
- 자동 다운로드 불가 시 스킵하고 사유 기록

### [7] MM-Vet (~500MB)
- ⚠️ 이미지가 Google Drive 호스팅일 수 있음
- LLaVA eval docs 확인 후 `gdown` 시도
- ChatGPT API 필요 시 추론 결과만 저장, 채점 보류

### [8] VizWiz (~7GB)
- ❌ vizwiz.org에서 계정 등록 + 이용 동의 필요
- 자동화 불가할 가능성 높음 → 시도 후 안 되면 스킵

## 각 벤치마크 세팅 절차 (공통)
1. **LLaVA Evaluation.md를 web_fetch로 읽어서** 해당 벤치마크의 정확한 데이터 경로, 파일 구조, 실행 방법 파악
2. **VisPruner 코드 내 eval 스크립트** 확인 (경로, 인자 등)
3. **데이터 다운로드** (위 우선순위 방법대로)
4. **경로 배치**: `playground/data/eval/<벤치마크>/` 구조에 맞춰 배치 또는 심볼릭 링크
5. **Sanity check**: 소량(100~300개)으로 빠르게 테스트
6. **전체 실행**: 576/128/64/32 토큰 4세팅

important_ratio = 0.5 (POPE 재현 때와 동일한 기본값)

## 결과 기록
`vispruner_md/` 폴더에 추가 문서 작성:

### vispruner_md/05_additional_benchmarks.md
각 벤치마크별로:
- 데이터 다운로드 출처 (실제 사용한 URL), 용량, 방법
- 다운로드 실패/스킵한 벤치마크의 사유
- 데이터 배치 경로
- 실행한 명령어
- 에러 및 해결 과정

### vispruner_md/06_full_reproduction_results.md
논문 Table 1 전체 재현 비교표:

| Benchmark | Metric | 논문(576) | 재현(576) | 논문(128) | 재현(128) | 논문(64) | 재현(64) | 논문(32) | 재현(32) |
|-----------|--------|----------|----------|----------|----------|---------|---------|---------|---------|
| POPE | F1 | 85.9 | 85.87 | 84.6 | 84.43 | 80.4 | 80.91 | 72.7 | 74.17 |
| SQA-IMG | Acc | 66.8 | ? | 69.1 | ? | 69.1 | ? | 69.2 | ? |
| GQA | Acc | 62.0 | ? | 58.2 | ? | 55.4 | ? | 52.2 | ? |
| TextVQA | Acc | 58.2 | ? | 57.0 | ? | 55.8 | ? | 53.9 | ? |
| VQAv2 | Acc | 78.5 | ? | 75.8 | ? | 72.7 | ? | 67.7 | ? |
| MME | Score | 1510.7 | ? | 1461.4 | ? | 1369.9 | ? | 1271.0 | ? |
| MMBench | Acc | 64.3 | ? | 62.7 | ? | 61.3 | ? | 58.4 | ? |
| MMBench-CN | Acc | 58.3 | ? | 57.3 | ? | 55.1 | ? | 52.7 | ? |
| MM-Vet | Score | 31.1 | ? | 33.7 | ? | 32.3 | ? | 28.8 | ? |
| VizWiz | Acc | 50.0 | ? | 52.7 | ? | 53.3 | ? | 53.0 | ? |

- 스킵한 벤치마크는 "스킵(사유)" 표기
- 각 벤치마크별 소요 시간, GPU 메모리 사용량

## 주의사항
- 반드시 LLaVA Evaluation.md를 먼저 web_fetch로 읽고 시작할 것
- 다운로드 실패 시 3가지 대안까지 시도 후 스킵 → 시간 낭비 방지
- 벤치마크 하나 완료될 때마다 중간 결과를 문서에 업데이트
- 채점에 외부 API(ChatGPT, EvalAI)가 필요한 경우 추론 결과 jsonl만 저장하고 채점은 보류로 기록
- 에러 발생 시 해결 시도하고, 해결 불가하면 해당 벤치마크 스킵 후 다음으로 진행
## 목표
VisPruner 논문 "Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs"의 원본 코드를 실행하여 논문의 실험 결과를 재현한다.

## 환경 확인
1. 현재 디렉토리 구조를 확인하고 clone된 VisPruner 코드의 위치와 파일 구조를 파악해줘
2. Python 버전, CUDA 버전, GPU 종류(nvidia-smi), PyTorch 버전 등 실험 환경을 확인해줘
3. requirements.txt 또는 setup 관련 파일이 있으면 의존성을 설치해줘

## 모델 다운로드 및 준비
현재 로컬에 모델이 없으므로 직접 구해서 세팅해야 한다.

1. **LLaVA-1.5-7B 모델**:
   - HuggingFace에서 `liuhaotian/llava-v1.5-7b` 모델을 다운로드해줘
   - VisPruner 코드 내 config나 스크립트에서 모델 경로를 어떻게 지정하는지 확인하고, 다운로드한 경로에 맞게 설정해줘

2. **CLIP-ViT-L/14 (336px) visual encoder**:
   - LLaVA-1.5-7B에 포함되어 있는지 확인하고, 별도로 필요하면 `openai/clip-vit-large-patch14-336`을 다운로드해줘

3. 모델 다운로드 전 디스크 여유 공간을 반드시 확인하고 (df -h), 부족하면 알려줘
4. 다운로드 방법: huggingface-cli, git lfs, 또는 python 스크립트 중 가장 적합한 방법을 선택해줘

## 데이터셋 다운로드 및 준비
현재 로컬에 평가 데이터셋이 없으므로 직접 구해서 세팅해야 한다.

1. **VisPruner 코드 내 데이터셋 경로 확인**:
   - README, eval 스크립트, config 파일 등을 분석해서 각 벤치마크 데이터가 어떤 경로에 있어야 하는지 먼저 파악해줘
   - LLaVA evaluation 방식을 따르는지 확인 (LLaVA-1.5는 보통 `playground/data/eval/` 구조를 사용)

2. **벤치마크별 다운로드** (우선순위 순서):

   **[우선순위 1] POPE**:
   - COCO 2014 val images 필요 (http://images.cocodataset.org/zips/val2014.zip)
   - POPE 평가용 annotation 파일 (LLaVA eval 포맷)
   - LLaVA 공식 repo의 eval 데이터 구조를 참고해서 다운로드해줘
   - HuggingFace `lmms-lab/llava-bench-in-the-wild` 또는 LLaVA eval docs 확인

   **[우선순위 2] GQA**:
   - GQA 이미지셋과 질문 파일
   - https://cs.stanford.edu/people/dorarad/gqa/download.html 참고
   - LLaVA eval에서 사용하는 GQA testdev balanced 질문 파일 확인

   **[우선순위 3] VQAv2**:
   - VQAv2 test-dev 이미지 및 질문 파일
   - https://visualqa.org/download.html 참고

3. **데이터셋 경로 연결**:
   - 다운로드한 데이터를 VisPruner 코드가 기대하는 경로에 맞춰 배치하거나 심볼릭 링크를 걸어줘
   - 경로가 안 맞으면 코드 내 경로 설정을 수정해줘

4. 전체 데이터셋이 너무 크면:
   - 먼저 POPE만 준비해서 quick test를 진행
   - POPE도 무거우면 subset(1000개)으로 sanity check 먼저

## 실험 실행
1. VisPruner를 LLaVA-1.5-7B에 적용하여 아래 세팅으로 실험해줘:
   - Retained tokens: 128 (↓77.8%), 64 (↓88.9%), 32 (↓94.4%)
   - 논문 Table 1의 결과와 비교할 것

2. 실행 시 코드 내 주요 하이퍼파라미터를 확인해줘:
   - important token ratio (r): 논문에서 사용한 기본값
   - pruning이 적용되는 위치 (visual encoder의 어느 layer에서 attention을 추출하는지)
   - [CLS] attention 추출 방식

3. 실행 명령어를 코드의 README나 eval 스크립트에서 찾아서 그대로 사용하되, 모델/데이터 경로만 로컬 환경에 맞게 수정해줘

4. 만약 전체 벤치마크 실행이 너무 오래 걸리면, POPE 데이터셋의 일부 subset(예: 1000개 샘플)으로 먼저 빠르게 sanity check를 해줘

## 결과 기록 및 문서화
모든 과정과 결과를 `vispruner_md/` 폴더 안에 마크다운 파일로 정리해줘:

### vispruner_md/01_environment_setup.md
- 실험 환경 정보 (GPU, CUDA, Python, PyTorch 버전 등)
- 설치한 패키지 목록
- 모델 다운로드 경로, 용량, 다운로드 방법
- 데이터셋 다운로드 경로, 용량, 다운로드 방법
- 최종 디렉토리 구조

### vispruner_md/02_code_analysis.md
- VisPruner 코드 구조 분석 (주요 파일과 역할)
- 핵심 구현 부분 설명:
  - [CLS] attention 추출 로직
  - important token selection 로직
  - diverse token selection (similarity-based duplication removal) 로직
  - pruning이 VLM 파이프라인에 통합되는 방식
- 주요 하이퍼파라미터와 기본값 정리
- 모델/데이터 경로 설정 방법

### vispruner_md/03_reproduction_results.md
- 각 실험 세팅별 결과를 표로 정리
- 논문 Table 1의 값과 나란히 비교하는 표 작성
- 형식 예시:

| Setting | Metric | 논문 결과 | 재현 결과 | 차이 |
|---------|--------|----------|----------|------|
| 128 tokens | POPE | 84.6 | ? | ? |
| 64 tokens | POPE | 80.4 | ? | ? |
| 32 tokens | POPE | 72.7 | ? | ? |

- 실행 시간, GPU 메모리 사용량도 기록
- 재현 결과가 논문과 다를 경우 가능한 원인 분석

### vispruner_md/04_execution_log.md
- 실행한 명령어를 순서대로 기록 (복붙 가능하게)
- 에러 발생 시 에러 내용과 해결 방법 기록
- 모델/데이터 다운로드 과정 기록
- 각 단계별 소요 시간

## 주의사항
- 모델/데이터가 로컬에 전혀 없으므로, 코드 분석 → 필요 리소스 파악 → 다운로드 → 경로 설정 → 실행 순서를 지켜줘
- 에러가 나면 바로 멈추지 말고, 에러 원인을 분석하고 해결을 시도한 뒤 로그에 기록해줘
- 다운로드 URL이 막혀있거나 변경된 경우 대안 경로를 찾아줘
- GPU 메모리가 부족하면 batch size 조정, half precision 등 대안을 제시해줘
- 각 단계 완료 시 진행 상황을 알려줘
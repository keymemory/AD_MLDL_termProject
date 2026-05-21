# 05. 추가 벤치마크 데이터 준비 및 실행 로그

POPE 재현 완료 상태에서 논문 Table 1의 나머지 벤치마크를 추가 재현. 공통 설정:
LLaVA-1.5-7B, `important_ratio=0.5`, 토큰 {576,128,64,32}, greedy, `CUDA_LAUNCH_BLOCKING=1`(안정모드)+resume+자동재시도.

## 0. 공통: LLaVA eval.zip

- `gdown 1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy` (23MB) → `playground/data/eval/` 에 해제.
- 모든 벤치마크의 LLaVA 포맷 질문 파일(`llava_*`)과 디렉토리 구조 제공.

## 1. 벤치마크별 데이터 출처 (실제 사용 URL)

| 벤치마크 | 질문 파일 | 이미지/추가 데이터 | 채점 |
|---|---|---|---|
| **SQA-IMG** | eval.zip `scienceqa/llava_test_CQM-A.json` (4241, IMG 2017) | 이미지 `https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip`, `pid_splits.json`·`problems.json` (lupantech/ScienceQA raw) | 로컬 `eval_science_qa` |
| **TextVQA** | eval.zip `textvqa/llava_textvqa_val_v051_ocr.jsonl` (5000) | `https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json`, `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip` | 로컬 `eval_textvqa` |
| **GQA** | eval.zip `gqa/llava_gqa_testdev_balanced.jsonl` (12578) | 이미지 `https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip` (~20GB), 질문/eval `https://nlp.stanford.edu/data/gqa/{questions1.2,eval}.zip` | 로컬 `gqa eval/eval.py` |
| **MM-Vet** | eval.zip `mm-vet/llava-mm-vet.jsonl` (218) | `https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip` | 외부(GPT-4 API) → 추론만 |
| **MMBench** | `https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv` (이미지 tsv 내장) | – | 외부(opencompass) → 추론만 |
| **MMBench-CN** | `.../mmbench_dev_cn_20231003.tsv` | – | 외부 → 추론만 |
| **VizWiz** | eval.zip `vizwiz/llava_test.jsonl` (8000) | `https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip`, `.../vqa_data/Annotations.zip` | 외부(EvalAI) → 추론만 |
| **VQAv2** | eval.zip `vqav2/llava_vqav2_mscoco_test-dev2015.jsonl` (107394) | `http://images.cocodataset.org/zips/test2015.zip` (~12GB) | 외부(EvalAI) → 추론(대규모) |
| **MME** | eval.zip `MME/llava_mme.jsonl` (2374) | 공식 `MME_Benchmark_release_version`(구글폼/이메일 필요). HF `lmms-lab/MME`(parquet) 시도 | 아래 6 참조 |

## 2. 데이터 배치 경로

`playground/data/eval/<벤치마크>/` 하위에 LLaVA 표준 구조대로 배치 (eval.zip이 기본 골격 제공).
- SQA 이미지: `scienceqa/images/test/<id>/image.png`
- TextVQA 이미지: `textvqa/train_images/`
- GQA 이미지: `gqa/data/images/`
- MM-Vet 이미지: `mm-vet/mm-vet/images/`
- VizWiz 이미지: `vizwiz/test/`
- VQAv2 이미지: `vqav2/test2015/`

## 3. 실행 명령어

```bash
# 로컬 채점 가능 (각각 별도 GPU 병렬)
bash run_textvqa.sh        # GPU2
bash run_sqa.sh            # GPU0
bash run_rest.sh <gpu>     # MM-Vet→MMBench→MMBench-CN→VizWiz→GQA 순차
```
각 스크립트: 4세팅 × 안정모드 + resume(이미 처리한 id 건너뜀) + 최대 15회 자동 재시도.

## 4. 스킵/보류 및 사유 (실행 후 확정)

- **MM-Vet / MMBench / MMBench-CN / VizWiz / VQAv2**: 채점에 외부 서버·API(GPT-4, opencompass, EvalAI) 필요 → **추론 결과 jsonl/upload json만 생성**, 채점은 보류(논문값과 직접 수치비교 불가, 추론 산출물은 보존).
- **MME**: 공식 `MME_Benchmark_release_version`이 구글폼/이메일 동의 필요(자동화 불가). HF `lmms-lab/MME` 미러는 parquet 포맷이라 VisPruner/LLaVA가 요구하는 `<category>/<file>.png` + eval_tool 라벨 구조와 불일치. (상세 처리: 아래 6절)
- **VQAv2**: 107,394문항 × 4세팅 ≈ 43만 추론. 안정모드 처리량 한계로 전량은 시간 초과 → 처리 가능분/대체 방안 6절에 기록.

## 5. 에러 및 해결
- POPE에서 확인된 dtype 혼용 CUDA 이슈는 `builder.py` 패치로 해결된 상태에서 시작 →
  추가 벤치마크 5종 모두 안정모드+resume로 **크래시 없이 1회 완주**(재시도 0회).
- `model_vqa_science` / `model_vqa` 에도 resume 패치 추가(POPE 때 `model_vqa_loader`에 한 것과 동일).
- ScienceQA: VisPruner `sqa.sh`의 SPLIT은 `llava_test_CQM-I`이나 eval.zip은 `llava_test_CQM-A.json` 제공
  → LLaVA 공식과 동일하게 `CQM-A` 사용(`eval_science_qa --split test`, IMG-Accuracy 지표).
- GQA: stanford `eval.py`가 GQA v1.2의 누락 에셋(scene graph 등)으로 실패 →
  haotian-liu gist(`db6eddc2...`) 패치본 `eval/eval.py` 적용(scenes/choices/consistency 로드 주석화).

## 6. MME / VQAv2 처리 결과 (스킵 + 사유)

### MME — 스킵 (채점 불가)
- 공식 `MME_Benchmark_release_version`은 Google Form/이메일 동의 후 배포(자동 다운로드 불가).
- HF 미러 `lmms-lab/MME`는 **parquet 포맷**으로, VisPruner/LLaVA가 요구하는
  `playground/data/eval/MME/MME_Benchmark_release_version/<category>/<file>.png` +
  `eval_tool`의 카테고리별 정답 txt 구조와 **불일치**. parquet→해당 레이아웃 재구성 및
  `calculation.py` 호환 라벨 복원은 본 과제 범위를 크게 초과.
- 결론: develop_ver2 "이메일 등록 필요 시 스킵, 사유 기록" 지침에 따라 **스킵**.

### VQAv2 — 스킵 (compute + 채점 모두 비현실적)
- test-dev 107,394문항 × 4세팅 ≈ **43만 추론**. 안정모드(`CUDA_LAUNCH_BLOCKING=1`,
  ~3–6 q/s)에서 단일 GPU 기준 약 24시간+ 소요.
- 채점은 **EvalAI 서버 제출 전용**(오프라인 로컬 지표 없음). val 대체도 ~214k문항으로 더 큼.
- 로컬 채점이 불가능해 추론 산출물의 검증 가치가 낮고, 동일 GPU 시간을
  로컬 채점 가능 벤치마크(POPE·TextVQA·SQA·GQA, 모두 재현 완료)에 투입하는 것이 합리적.
- 결론: develop_ver2 "제출 불가 시 사유 기록" 지침에 따라 **스킵(사유 기록)**.

### 외부 채점 벤치마크 (추론 산출물만 보존)
- MM-Vet / MMBench / MMBench-CN / VizWiz: 4세팅 추론 완료(또는 진행), 답변 jsonl 및
  업로드용 변환 파일 보존. 채점은 각각 GPT-4 API / opencompass / EvalAI 필요로 수치 보류.
  산출 경로: `playground/data/eval/<벤치>/answers(_upload)/...`

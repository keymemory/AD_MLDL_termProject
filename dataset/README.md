# dataset/ — VisPruner 재현 사용 데이터셋

VisPruner 논문 Table 1 재현에 사용한 벤치마크 데이터를 벤치마크별로 정리.
원래 위치 `VisPruner_run/playground/data/eval/<benchmark>` 는 여기로 **심볼릭 링크**되어 있어
기존 스크립트(run_*.sh)가 그대로 동작함. 추가 실험 시 이 폴더만 재사용하면 됨.

| 폴더 | 벤치마크 | 핵심 내용 | 출처 | 용량 |
|---|---|---|---|--:|
| `pope/` | POPE | llava_pope_test.jsonl(8910), coco annotation, val2014 이미지 500장 | COCO + POPE repo | 92M |
| `textvqa/` | TextVQA | llava_textvqa_val_v051_ocr.jsonl(5000), TextVQA_0.5.1_val.json, train_images | dl.fbaipublicfiles.com | 14G |
| `scienceqa/` | ScienceQA | llava_test_CQM-A.json, problems.json, pid_splits.json, images/test | scienceqa.s3 + lupantech repo | 265M |
| `gqa/` | GQA | llava_gqa_testdev_balanced.jsonl(12578), data/images, questions, eval(gist패치) | downloads.cs.stanford.edu | 48G |
| `mm-vet/` | MM-Vet | llava-mm-vet.jsonl(218), mm-vet/images | github yuweihao/MM-Vet v1 | 66M |
| `mmbench/` | MMBench | mmbench_dev_20230712.tsv (이미지 base64 내장) | openmmlab | 144M |
| `mmbench_cn/` | MMBench-CN | mmbench_dev_cn_20231003.tsv | openmmlab | 143M |
| `vizwiz/` | VizWiz | llava_test.jsonl(8000), test 이미지, Annotations | vizwiz.cs.colorado.edu | 3.8G |
| `vqav2/` | VQAv2 | llava_vqav2_*.jsonl, test2015 이미지 | cocodataset (스킵: 채점 EvalAI 전용) | 13G |
| `MME/` | MME | llava_mme.jsonl(2374), hf_mme(parquet) | HF lmms-lab/MME (스킵: 포맷 불일치) | 826M |

- 각 폴더의 `answers/`·`answers_upload/` 는 본 재현의 추론/채점 결과(원시 산출물).
- 데이터 획득 상세 URL·방법: `../vispruner_md/05_additional_benchmarks.md`
- 재현 결과 종합: `../VisPruner_전체재현_최종보고서.md`, `../vispruner_md/06_full_reproduction_results.md`

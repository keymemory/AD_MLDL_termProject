# legacy_llava — LLaVA/VisPruner 원본 스크립트 (본 과제 미사용)

[LLaVA](https://github.com/haotian-liu/LLaVA) / [VisPruner](https://github.com/Theia-4869/VisPruner)
원본 저장소에서 그대로 가져온 **학습·평가 보일러플레이트**입니다.
본 과제(추론·평가 전용)에서는 사용하지 않으며, 원본 코드 보존 목적으로 격리·보관합니다.

- `finetune*.sh`, `pretrain*.sh`, `zero*.json` — LLaVA 학습/DeepSpeed 설정 (본 과제 학습 없음)
- `v1_5/eval/`, `v1_6/eval/` — LLaVA 표준 벤치마크 평가 스크립트
  (본 과제는 `../../exp_runner/` 의 자체 워커로 평가 수행)
- `convert_*.py`, `merge_lora_weights.py`, `extract_mm_projector.py` — LLaVA 제출/변환 유틸

> 본 과제가 실제로 사용하는 변환 스크립트(`convert_gqa_for_eval.py`)는 상위 `scripts/`에 있습니다.

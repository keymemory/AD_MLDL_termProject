"""[Phase 1] selection 분포 빠른 프로브 — LLM 디코딩 없이 vision tower + selection만.

이미지별 [CLS] attention 점수를 1회만 추출해 여러 파라미터(k/τ/p)에 재사용하므로
model_vqa_loader 전체 추론(LLM 디코딩 포함) 대비 수십 배 빠르다. 성능(F1)은 안 보고
분포(n_imp/M1/floor/cap)만 필요할 때 사용.

사용:
  python exp_runner/probe_dist.py statistical 2.0,0.3,0.5,0.8,1.0 [M2=64]
  python exp_runner/probe_dist.py energy      0.3,0.5,0.7        [M2=64]
  python exp_runner/probe_dist.py percentile  0.1,0.2,0.3        [M2=64]   # 상위 p% = important

정확성: clip_encoder.forward + encode_images의 `image_attentions.mean(dim=1)` +
        compute_adaptive_counts 를 그대로 재사용 → 실제 파이프라인과 동일 분포.
        (검증: statistical 2.0 결과가 기존 act_stat_k20.dist.jsonl과 일치해야 함.)
"""
import sys
import os
import json
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path
from llava.model.adaptive_selection import compute_adaptive_counts


def percentile_counts(s, M2, p, r_floor=0.5, M1_cap=384):
    """상위 p%(0<p<1) 토큰을 important로. compute_adaptive_counts와 동일한 floor/cap 규칙."""
    B, N = s.shape
    n_imp_list, M1_list, detail_list = [], [], []
    cap_val = min(M1_cap, N)
    for b in range(B):
        n_imp = max(1, min(int(round(N * p)), N))
        raw_M1 = int(round(n_imp / r_floor))
        M1 = max(M2, min(raw_M1, cap_val))
        n_imp = min(n_imp, M1)
        n_imp_list.append(n_imp)
        M1_list.append(M1)
        detail_list.append({"raw_M1": raw_M1, "floor": raw_M1 < M2, "cap": raw_M1 > cap_val})
    return n_imp_list, M1_list, detail_list


def main():
    method = sys.argv[1]
    params = [float(x) for x in sys.argv[2].split(",")]
    M2 = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    mp = "models/llava-v1.5-7b"
    # visual_token_num/important_ratio는 __init__ 필수 인자(더미; probe는 vision tower만 사용)
    tok, model, image_processor, _ = load_pretrained_model(
        mp, None, get_model_name_from_path(mp),
        visual_token_num=64, important_ratio=0.5)
    vt = model.get_model().get_vision_tower()
    device = next(model.parameters()).device

    P = "playground/data/eval/pope"
    Q = [json.loads(l) for l in open(f"{P}/llava_pope_sanity.jsonl")]
    IMGDIR = f"{P}/val2014"

    # 이미지별 [CLS] attention 점수(head 평균) 1회 추출 → 여러 param 재사용
    scores = []
    with torch.inference_mode():
        for q in Q:
            img = Image.open(os.path.join(IMGDIR, q["image"])).convert("RGB")
            it = process_images([img], image_processor, model.config)[0]
            it = it.unsqueeze(0).to(device=device, dtype=torch.float16)
            _feats, attns = vt(it, output_attentions=True)   # (1,N,C),(1,H,N)
            # GPU·fp16 유지 (실제 encode_images와 동일 — cumsum이 CPU fp16 미지원)
            scores.append(attns.mean(dim=1))                  # (1,N) on GPU
    print(f"[probe] extracted attention for {len(scores)} images")

    for p in params:
        recs = []
        for s in scores:
            if method == "statistical":
                n_imp, M1, det = compute_adaptive_counts(s, "statistical", M2, stat_k=p)
            elif method == "energy":
                n_imp, M1, det = compute_adaptive_counts(s, "energy", M2, energy_tau=p)
            elif method == "percentile":
                n_imp, M1, det = percentile_counts(s, M2, p)
            else:
                raise SystemExit(f"unknown method {method}")
            recs.append({"n_imp": int(n_imp[0]), "M1": int(M1[0]), **det[0]})
        out = f"{P}/answers/regress/probe_{method}_M{M2}_{p}.dist.jsonl"
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"[probe] {method} p={p}: {len(recs)} -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TextVQA 실험 준비: HF Arrow 데이터셋에서 이미지 추출 + annotation JSON 생성
- 이미지: /data1/heejung/datasets/textvqa_val_images/{image_id}.jpg
- annotation: /home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project/exp_runner/textvqa_val_annotations.json

실행:
  /home/dxlab/anaconda3/envs/vispruner/bin/python exp_runner/setup_textvqa.py
"""

import os
import json
from tqdm import tqdm

ARROW_PATH = "/data1/heejung/hf_datasets/TextVQA/validation"
IMG_OUT    = "/data1/heejung/datasets/textvqa_val_images"
ANN_OUT    = "/home/dxlab/jupyter/heejung/AD_MLDL_termProject/term_project/exp_runner/textvqa_val_annotations.json"

os.makedirs(IMG_OUT, exist_ok=True)

print("Loading TextVQA validation dataset from HF Arrow ...")
import datasets
ds = datasets.load_from_disk(ARROW_PATH)
print(f"  {len(ds)} items loaded")

# Extract images & build annotation
annotations = []
skipped_img = 0

for item in tqdm(ds, desc="Extracting images + annotations"):
    image_id = item["image_id"]
    img_path = os.path.join(IMG_OUT, f"{image_id}.jpg")

    # Save image (skip if already exists)
    if not os.path.exists(img_path):
        try:
            item["image"].save(img_path, format="JPEG", quality=95)
        except Exception as e:
            print(f"  [WARN] failed to save {image_id}: {e}")
            skipped_img += 1
            continue

    annotations.append({
        "image_id": image_id,
        "question":  item["question"],          # bare question text (lowercase in HF)
        "answers":   item["answers"],
    })

print(f"Images saved: {len(os.listdir(IMG_OUT))}  (skipped: {skipped_img})")

# Write annotation file
with open(ANN_OUT, "w") as f:
    json.dump({"data": annotations}, f)
print(f"Annotation JSON saved → {ANN_OUT}  ({len(annotations)} entries)")

# Quick sanity check: verify all question-file image_ids exist
QF = "/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
with open(QF) as f:
    q_items = [json.loads(l) for l in f]
missing = [q["image"] for q in q_items if not os.path.exists(os.path.join(IMG_OUT, q["image"]))]
print(f"Question file: {len(q_items)} items | missing images: {len(missing)}")
if missing:
    print("  First 5 missing:", missing[:5])
else:
    print("  All images present [OK]")

print("Done.")

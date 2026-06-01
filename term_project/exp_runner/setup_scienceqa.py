#!/usr/bin/env python3
"""
ScienceQA 실험 준비: HF 데이터셋에서 이미지 추출 + problems.json / pid_splits.json 생성
- 이미지:        /data1/heejung/datasets/scienceqa/images/test/{id}/image.png
- problems.json: /data1/heejung/datasets/scienceqa/problems.json
- pid_splits.json: /data1/heejung/datasets/scienceqa/pid_splits.json

실행:
  /home/dxlab/anaconda3/envs/vispruner/bin/python exp_runner/setup_scienceqa.py
"""

import os, json
from tqdm import tqdm

HF_DATASET   = "derek-thomas/ScienceQA"
SQA_BASE     = "/data1/heejung/datasets/scienceqa"
IMG_BASE     = f"{SQA_BASE}/images/test"   # model_vqa_science --image-folder 경로
PROBLEMS_OUT = f"{SQA_BASE}/problems.json"
SPLITS_OUT   = f"{SQA_BASE}/pid_splits.json"
LLaVA_QF     = "/home/dxlab/jupyter/heejung/VisPruner/playground/data/eval/scienceqa/llava_test_CQM-A.json"

os.makedirs(SQA_BASE, exist_ok=True)

print("Loading ScienceQA test split from HuggingFace ...")
import datasets as hf_datasets
ds = hf_datasets.load_dataset(HF_DATASET, split="test", trust_remote_code=True)
print(f"  {len(ds)} items  (images: {sum(1 for x in ds if x['image'] is not None)})")

# LLaVA question file: same order as HF dataset, provides problem IDs
print("Loading LLaVA question file ...")
with open(LLaVA_QF) as f:
    llava_qs = json.load(f)

assert len(llava_qs) == len(ds), f"Size mismatch: LLaVA={len(llava_qs)}, HF={len(ds)}"
print(f"  LLaVA qfile: {len(llava_qs)} items")

# Build problems.json + extract images
problems   = {}
split_ids  = []
skip_img   = 0

for i, (hf_item, llava_item) in enumerate(tqdm(zip(ds, llava_qs),
                                                total=len(ds),
                                                desc="Extracting")):
    pid = str(llava_item["id"])   # e.g. "4", "5", "8", ...
    split_ids.append(pid)

    # problems.json entry
    problems[pid] = {
        "question": hf_item["question"],
        "choices":  hf_item["choices"],
        "answer":   hf_item["answer"],           # integer index
        "hint":     hf_item.get("hint", ""),
    }

    # Extract image if present
    if hf_item["image"] is not None:
        img_dir  = os.path.join(IMG_BASE, pid)
        img_path = os.path.join(img_dir, "image.png")
        os.makedirs(img_dir, exist_ok=True)
        if not os.path.exists(img_path):
            try:
                hf_item["image"].save(img_path, format="PNG")
            except Exception as e:
                print(f"  [WARN] {pid}: {e}")
                skip_img += 1

# Write problems.json
with open(PROBLEMS_OUT, "w") as f:
    json.dump(problems, f)
print(f"problems.json → {PROBLEMS_OUT}  ({len(problems)} entries)")

# Write pid_splits.json  (only test split needed)
pid_splits = {"test": split_ids, "val": [], "train": []}
with open(SPLITS_OUT, "w") as f:
    json.dump(pid_splits, f)
print(f"pid_splits.json → {SPLITS_OUT}  (test: {len(split_ids)} IDs)")

# Verify image count
actual_imgs = len([p for p in os.listdir(IMG_BASE) if os.path.isdir(os.path.join(IMG_BASE, p))])
print(f"Image folders extracted: {actual_imgs}  (skipped: {skip_img})")

# Sanity check: verify LLaVA question file images match extracted paths
missing = []
for q in llava_qs:
    if "image" in q:
        expected = os.path.join(IMG_BASE, q["image"])
        if not os.path.exists(expected):
            missing.append(q["image"])
print(f"LLaVA qfile image cross-check: {len(missing)} missing" if missing else "All LLaVA image refs present [OK]")

print("Done.")

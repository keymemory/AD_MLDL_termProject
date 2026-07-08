"""[Phase2-A] MME parquet → 이미지 폴더 추출.

주의: MME는 카테고리마다 경로 체계가 다르다.
  - 일부: category/file.png (flat)         예: code_reasoning/0020.png
  - 일부: category/images/file.jpg (하위)   예: artwork/images/14777.jpg
parquet의 question_id는 flat 형태(artwork/14777.jpg)지만, 추론에 쓰는 llava_mme.jsonl의
image 필드는 실제 경로(artwork/images/14777.jpg)다. → llava_mme image 경로로 저장해야
model_vqa_loader가 찾는다. parquet bytes는 (category, basename)으로 매칭한다.
"""
import glob
import json
import os

import pandas as pd

base = "playground/data/eval/MME"
out = f"{base}/MME_Benchmark_release_version"

# parquet: (category, basename) -> image bytes
pq = {}
for f in sorted(glob.glob(f"{base}/hf_mme/data/*.parquet")):
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        qid = r["question_id"]
        cat = qid.split("/")[0]
        bn = os.path.basename(qid)
        img = r["image"]
        b = img["bytes"] if isinstance(img, dict) else img
        pq[(cat, bn)] = b

# llava_mme.jsonl image 경로로 저장
seen = set()
n = miss = 0
for l in open(f"{base}/llava_mme.jsonl"):
    d = json.loads(l)
    img_path = d["image"]              # 실제 추론 경로
    if img_path in seen:
        continue
    seen.add(img_path)
    cat = img_path.split("/")[0]
    bn = os.path.basename(img_path)
    b = pq.get((cat, bn))
    if b is None:
        print("MISS", img_path)
        miss += 1
        continue
    p = os.path.join(out, img_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fp:
        fp.write(b)
    n += 1
print(f"extracted {n} images (miss={miss}) -> {out}")

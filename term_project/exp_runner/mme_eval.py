"""[Phase2-A] MME 로컬 채점 (parquet GT 직접, 공식 eval_tool 우회).

MME 공식 점수: 각 category score = (acc + acc_plus) * 100.
  acc      = 개별 yes/no 질문 정확도
  acc_plus = 이미지당 2질문 모두 정답인 비율
perception = 10개 category 합, cognition = 4개 category 합, total = 합.

answers(question_id=category/file.png, prompt, text)와 parquet(question_id, question, answer)을
정규화 질문으로 매칭한다. (llava_mme prompt와 parquet question의 보조문구 차이를 제거)

사용: python exp_runner/mme_eval.py <answers.jsonl>
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd


def qkey(qid):
    # (category, basename) — 카테고리별 images/ 유무 차이를 흡수
    return (qid.split("/")[0], os.path.basename(qid))

PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity",
              "scene", "landmark", "artwork", "OCR"]
COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def norm_q(q):
    q = (q or "").lower()
    q = q.replace("answer the question using a single word or phrase.", "")
    q = q.replace("please answer yes or no.", "")
    q = re.sub(r"[^a-z0-9 ]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def yn(text):
    t = (text or "").strip().lower()
    if t.startswith("yes") or re.match(r"^\W*yes\b", t):
        return "yes"
    if t.startswith("no") or re.match(r"^\W*no\b", t):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return t[:3]


def main():
    ans_file = sys.argv[1]
    gt = {}
    for f in glob.glob("playground/data/eval/MME/hf_mme/data/*.parquet"):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            gt[(qkey(r["question_id"]), norm_q(r["question"]))] = (r["answer"].strip().lower(), r["category"])

    recs = defaultdict(list)   # category -> [(file, correct)]
    miss = 0
    for l in open(ans_file):
        d = json.loads(l)
        qid = d["question_id"]
        cat = qid.split("/")[0]
        file = qid.split("/")[-1]
        key = (qkey(qid), norm_q(d["prompt"]))
        if key not in gt:
            miss += 1
            continue
        gt_ans, _ = gt[key]
        recs[cat].append((file, int(yn(d["text"]) == gt_ans)))

    def cat_score(items):
        total = len(items)
        if total == 0:
            return 0.0
        acc = sum(c for _, c in items) / total
        byfile = defaultdict(list)
        for f, c in items:
            byfile[f].append(c)
        pairs = [cs for cs in byfile.values() if len(cs) >= 2]
        accp = (sum(1 for cs in pairs if all(cs)) / len(pairs)) if pairs else 0.0
        return (acc + accp) * 100

    per = sum(cat_score(recs[c]) for c in PERCEPTION if c in recs)
    cog = sum(cat_score(recs[c]) for c in COGNITION if c in recs)
    print(f"MME perception={per:.2f} cognition={cog:.2f} total={per+cog:.2f}  (unmatched={miss})")
    for c in PERCEPTION + COGNITION:
        if c in recs:
            print(f"  {c}: {cat_score(recs[c]):.2f}  (n={len(recs[c])})")


if __name__ == "__main__":
    main()

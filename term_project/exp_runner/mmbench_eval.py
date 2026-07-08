"""[Phase2-A] MMBench dev 로컬 채점.
model_vqa_mmbench 출력(answers.jsonl) + dev tsv 정답(answer 컬럼) → accuracy.
사용: python exp_runner/mmbench_eval.py <answers.jsonl> <mmbench_dev.tsv>
"""
import json
import re
import sys
import pandas as pd


def extract_choice(text, options, option_chars):
    t = (text or "").strip()
    m = re.match(r"^\s*\(?([A-D])\b", t)          # 첫 글자 A-D / (A)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])[\.\):]", t)          # A. A) A:
    if m:
        return m.group(1)
    for ch, opt in zip(option_chars or [], options or []):  # 옵션 텍스트 매칭
        if opt and str(opt).strip() and str(opt).strip().lower() in t.lower():
            return ch
    m = re.search(r"\b([A-D])\b", t)               # 단독 A-D
    if m:
        return m.group(1)
    return (t[:1].upper() if t else "?")


def main():
    ans_file, tsv_file = sys.argv[1], sys.argv[2]
    tsv = pd.read_table(tsv_file)
    gt = {int(r["index"]): str(r["answer"]).strip()
          for _, r in tsv.iterrows() if pd.notna(r.get("answer"))}
    correct = total = 0
    for l in open(ans_file):
        d = json.loads(l)
        idx = int(d["question_id"])
        if idx not in gt:
            continue
        total += 1
        pred = extract_choice(d.get("text", ""), d.get("options", []), d.get("option_char", []))
        if pred == gt[idx]:
            correct += 1
    acc = 100.0 * correct / total if total else 0.0
    print(f"MMBench Acc = {acc:.2f} ({correct}/{total})")


if __name__ == "__main__":
    main()

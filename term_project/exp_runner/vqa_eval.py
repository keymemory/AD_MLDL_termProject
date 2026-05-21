"""VQAv2 로컬 채점 — 공식 VQA accuracy(정규화 포함). subset GT 기반.
사용: python vqa_eval.py <answers.jsonl> <val_subset_gt.json> [overall|by_type]
"""
import sys, json, re

contractions = {"aint":"ain't","arent":"aren't","cant":"can't","couldve":"could've",
"couldnt":"couldn't","didnt":"didn't","doesnt":"doesn't","dont":"don't","hadnt":"hadn't",
"hasnt":"hasn't","havent":"haven't","hes":"he's","im":"i'm","isnt":"isn't","its":"it's",
"lets":"let's","mightve":"might've","mustve":"must've","shes":"she's","shouldve":"should've",
"shouldnt":"shouldn't","thats":"that's","theres":"there's","theyd":"they'd","theyre":"they're",
"wasnt":"wasn't","werent":"weren't","whats":"what's","wheres":"where's","whos":"who's",
"wont":"won't","wouldve":"would've","wouldnt":"wouldn't","youd":"you'd","youre":"you're","youve":"you've"}
manualMap = {'none':'0','zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
'six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
articles = ['a','an','the']
periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(\,)(\d)")
punct = [';','/','[',']','"','{','}','(',')','=','+','\\','_','-','>','<','@','`',',','?','!']

def processPunctuation(s):
    for p in punct:
        if (p+' ' in s or ' '+p in s) or (re.search(commaStrip, s) is not None):
            s = s.replace(p, '')
        else:
            s = s.replace(p, ' ')
    s = periodStrip.sub("", s, re.UNICODE)
    return s

def processDigitArticle(s):
    out = []
    for w in s.lower().split():
        w = manualMap.get(w, w)
        if w not in articles:
            out.append(w)
    for i, w in enumerate(out):
        if w in contractions:
            out[i] = contractions[w]
    return ' '.join(out)

def norm(s):
    s = s.replace('\n', ' ').replace('\t', ' ').strip()
    s = processPunctuation(s)
    s = processDigitArticle(s)
    return s

def main():
    ans_f, gt_f = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "overall"
    gt = json.load(open(gt_f))
    accs = {}
    by = {"yes/no": [], "number": [], "other": []}
    n = 0
    for line in open(ans_f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        qid = str(d["question_id"])
        if qid not in gt:
            continue
        res = norm(str(d["text"]))
        gts = [norm(a) for a in gt[qid]["answers"]]
        m = sum(1 for g in gts if g == res)
        acc = min(m / 3.0, 1.0)
        accs[qid] = acc
        by[gt[qid]["answer_type"]].append(acc)
        n += 1
    overall = 100.0 * sum(accs.values()) / max(len(accs), 1)
    print(f"VQAv2(subset) N={n}  Overall Acc = {overall:.2f}")
    for t in ("yes/no", "number", "other"):
        v = by[t]
        if v:
            print(f"  {t:<7}: {100.0*sum(v)/len(v):.2f}  (n={len(v)})")

if __name__ == "__main__":
    main()

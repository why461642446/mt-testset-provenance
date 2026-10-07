# -*- coding: utf-8 -*-
"""计算 600M vs 3.3B 的 chrF（论文同一实现）。"""
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
D = os.path.join(ROOT, "massive", "1.1", "data")
BETA, ORDER = 2.0, 6


def ngrams(s, n, strip_space=True):
    if strip_space:
        s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_corpus(hyps, refs, strip_space=True):
    stats = {}
    for n in range(1, ORDER + 1):
        m = h = r = 0
        for hy, re_ in zip(hyps, refs):
            ch, cr = ngrams(hy, n, strip_space), ngrams(re_, n, strip_space)
            h += sum(ch.values()); r += sum(cr.values()); m += sum((ch & cr).values())
        stats[n] = (m, h, r)
    precs = [stats[n][0] / stats[n][1] if stats[n][1] else 0.0 for n in stats]
    recs = [stats[n][0] / stats[n][2] if stats[n][2] else 0.0 for n in stats]
    pr, rc = float(np.mean(precs)), float(np.mean(recs))
    if pr + rc == 0:
        return 0.0
    return (1 + BETA ** 2) * pr * rc / (BETA ** 2 * pr + rc) * 100.0


# 解压
for nm in ["massive_mt2", "massive_mt_600"]:
    tgt = os.path.join(SP, nm)
    os.makedirs(tgt, exist_ok=True)
    with zipfile.ZipFile(os.path.join(SP, "mt2_chrf_results.zip")) as z:
        for n in z.namelist():
            if n.startswith(nm + "/") and not n.endswith("/"):
                rel = os.path.relpath(n, nm)
                out = os.path.join(tgt, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(n) as s, open(out, "wb") as d:
                    d.write(s.read())
print("  已解压")


def load(path, part="test"):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["partition"] == part:
                out.append(r)
    return out


print()
print("=" * 90)
print("chrF 对照：600M vs 3.3B（论文同一实现：n-gram 1..6, beta=2, 去空格, 语料级）")
print("=" * 90)
res = {}
for loc in ["ko-KR", "zh-CN"]:
    loc_rows = load(os.path.join(D, loc + ".jsonl"))
    res[loc] = {}
    print()
    for tag, d in [("600M", os.path.join(SP, "massive_mt_600")),
                   ("3.3B", os.path.join(SP, "massive_mt2"))]:
        rows = load(os.path.join(d, loc + ".jsonl"))
        m = {str(r["id"]): r["utt"] for r in rows}
        pairs = [(m[str(r["id"])], r["utt"]) for r in loc_rows if str(r["id"]) in m]
        hyp = [h for h, _ in pairs]
        ref = [t for _, t in pairs]
        c = chrf_corpus(hyp, ref)
        lh = sum(len(h) for h in hyp) / len(hyp)
        lr = sum(len(t) for t in ref) / len(ref)
        res[loc][tag] = dict(chrf=c, lh=lh, lr=lr, n=len(pairs))
        print("   %-6s %-5s  chrF %6.2f   译文均长 %5.2f   参考均长 %5.2f   长度差 %+5.2f   n=%d"
              % (loc, tag, c, lh, lr, lh - lr, len(pairs)))
    a, b = res[loc]["600M"], res[loc]["3.3B"]
    print("          -> **chrF 差 = %+.2f**   长度差变化 = %+.2f 字符"
          % (b["chrf"] - a["chrf"], (b["lh"] - b["lr"]) - (a["lh"] - a["lr"])))

print()
print("=" * 90)
print("结论判读")
print("=" * 90)
for loc in ["ko-KR", "zh-CN"]:
    d = res[loc]["3.3B"]["chrf"] - res[loc]["600M"]["chrf"]
    dl = (res[loc]["3.3B"]["lh"] - res[loc]["3.3B"]["lr"]) - (res[loc]["600M"]["lh"] - res[loc]["600M"]["lr"])
    if abs(d) < 1.0:
        v = "**同系统偏差**可解释 9-16% 的全部（chrF 几乎不变）"
    elif d > 0:
        v = "3.3B 更接近参考 -> 9-16% 里含「离参考更近」的成分"
    else:
        v = "3.3B 离参考更远 -> 减少量不是「离参考更近」"
    print("  %-6s  chrF %+.2f   长度差 %+.2f   -> %s" % (loc, d, dl, v))

json.dump(res, open(os.path.join(SP, "chrf_mt2_result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print()
print("  已写出 chrf_mt2_result.json")

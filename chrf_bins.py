# -*- coding: utf-8 -*-
"""关键分析：按 chrF 分箱，看 L->MT 的下降是否随 MT 质量改善而消失。

若高 chrF 箱里下降基本消失 -> 残余主要是 MT 译文质量
若高 chrF 箱里下降仍在   -> 存在系统性的分布差异
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
BETA, ORDER = 2.0, 6


def ngrams(s, n):
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_sent(hy, re_):
    st = {}
    for n in range(1, ORDER + 1):
        ch, cr = ngrams(hy, n), ngrams(re_, n)
        st[n] = (sum((ch & cr).values()), sum(ch.values()), sum(cr.values()))
    p = np.mean([st[n][0] / st[n][1] if st[n][1] else 0.0 for n in st])
    r = np.mean([st[n][0] / st[n][2] if st[n][2] else 0.0 for n in st])
    return 0.0 if p + r == 0 else (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r) * 100.0


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    print("=" * 104)
    print("%s —— 按 chrF 分箱看 L->MT 下降" % nm)
    print("=" * 104)
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])

    ss = np.array([chrf_sent(a, b) for a, b in zip(mte[:n], lte[:n])])

    # 读四格预测（字符级 TextCNN，全部 5 个 seed）
    def getok(var):
        pats = sorted(glob.glob(os.path.join(
            SP, "runs_e4_char",
            "pred_massive_%s_in-language_%s_TextCNN_char_s*.csv" % (var, lang))))
        oks = []
        for p in pats:
            d = pd.read_csv(p)
            if len(d) != n:
                continue
            oks.append((d["pred"].values == d["gold"].values))
        return np.array(oks)

    OL, OM = getok("localized"), getok("mt-test")
    if not len(OL) or not len(OM):
        print("  缺 pred")
        continue
    m = min(OL.shape[1], OM.shape[1])
    OL, OM, keep2, ss2 = OL[:, :m], OM[:, :m], keep[:m], ss[:m]

    qs = np.percentile(ss2[keep2], [0, 25, 50, 75, 100])
    print("  chrF 四分位分箱（固定子集内）: %s" % "  ".join("%.1f" % q for q in qs))
    print()
    print("  %-16s %5s %10s %10s %10s %10s" % ("chrF 箱", "n", "L->L", "L->MT", "Δ", "翻转数"))
    tot = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = keep2 & (ss2 >= lo) & (ss2 <= hi if i == 3 else ss2 < hi)
        if sel.sum() == 0:
            continue
        a = 100.0 * OL[:, sel].mean()
        b = 100.0 * OM[:, sel].mean()
        flips = int((OL[:, sel] & ~OM[:, sel]).sum() / OL.shape[0])
        print("  [%5.1f, %5.1f) %5d %9.2f%% %9.2f%% %+9.2f %10d"
              % (lo, hi, sel.sum(), a, b, b - a, flips))
        tot.append((lo, hi, sel.sum(), a, b, b - a))
    print()
    a = 100.0 * OL[:, keep2].mean()
    b = 100.0 * OM[:, keep2].mean()
    print("  %-16s %5d %9.2f%% %9.2f%% %+9.2f" % ("固定子集全部", keep2.sum(), a, b, b - a))
    print()

    # 只用高 chrF 的一半
    med = np.median(ss2[keep2])
    hi_sel = keep2 & (ss2 >= med)
    a = 100.0 * OL[:, hi_sel].mean()
    b = 100.0 * OM[:, hi_sel].mean()
    print("  只用 chrF >= 中位(%.1f) 的一半:  L->L %.2f  L->MT %.2f  Δ %+.2f"
          % (med, a, b, b - a))
    print()

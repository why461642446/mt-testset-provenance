# -*- coding: utf-8 -*-
"""第 11 条补充：chrF 与「译文长度」的关系，并按译文长度分层重算下降。

审稿人指出：MT 译文比本地化平均长 1.8 个字符，而这个长度因素此前没有检查。
（此前只检查了**英文源句长度**，r ≈ 0。）

本脚本：
  1. 报告 MT / 本地化 译文的长度分布与差值
  2. chrF 与三种长度的相关：MT 长度、本地化长度、两者之差
  3. **按「译文长度差」分层**，重算 L→MT 的下降，看 chrF 效应是否仍存在
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
SEEDS = [42, 43, 44, 45, 46]


def ngrams(s, n):
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_sent(hy, re_):
    st = {}
    for n in range(1, ORDER + 1):
        ch, cr = ngrams(hy, n), ngrams(re_, n)
        st[n] = (sum((ch & cr).values()), sum(ch.values()), sum(cr.values()))
    p = float(np.mean([st[n][0] / st[n][1] if st[n][1] else 0.0 for n in st]))
    r = float(np.mean([st[n][0] / st[n][2] if st[n][2] else 0.0 for n in st]))
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


def get_preds(lang, variant, model="TextCNN"):
    out = {}
    for d in ["runs_e4_char", "runs_ws_extra", "runs_c2"]:
        pat = os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_char_s*.csv" % (variant, lang, model))
        for p in sorted(glob.glob(pat)):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


print("=" * 104)
print("第 11 条补充：chrF 与译文长度")
print("=" * 104)

SUM = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])

    L_len = np.array([len(x) for x in lte])          # 本地化译文长度
    M_len = np.array([len(x) for x in mte])          # MT 译文长度
    ss = np.array([chrf_sent(a, b) for a, b in zip(mte[:n], lte[:n])])
    diff = M_len - L_len

    print()
    print("─" * 104)
    print("%s   固定子集 n = %d" % (nm, keep.sum()))
    print("─" * 104)
    print("  译文长度（字符）:")
    print("     本地化  均值 %.2f   中位 %.0f" % (L_len[keep].mean(), np.median(L_len[keep])))
    print("     MT      均值 %.2f   中位 %.0f" % (M_len[keep].mean(), np.median(M_len[keep])))
    print("     **MT − 本地化 均值 %+.2f**（审稿意见说 +1.8）" % diff[keep].mean())

    print()
    print("  chrF 与长度的相关（固定子集内）:")
    print("     chrF vs MT 译文长度      r = %+.3f" % np.corrcoef(ss[keep], M_len[keep])[0, 1])
    print("     chrF vs 本地化译文长度    r = %+.3f" % np.corrcoef(ss[keep], L_len[keep])[0, 1])
    print("     **chrF vs 长度差(MT−LOC)  r = %+.3f**" % np.corrcoef(ss[keep], diff[keep])[0, 1])

    # ---------- 按「长度差」分层，重算下降 ----------
    L = get_preds(lang, "localized")
    M = get_preds(lang, "mt-test")
    seeds = sorted(set(L) & set(M))
    OL = np.array([L[s] for s in seeds])
    OM = np.array([M[s] for s in seeds])
    m2 = min(OL.shape[1], OM.shape[1], len(keep))
    OL, OM, keep2, ss2, diff2 = OL[:, :m2], OM[:, :m2], keep[:m2], ss[:m2], diff[:m2]

    print()
    print("  按「长度差」三分层的 L→L / L→MT（固定子集）:")
    q = np.percentile(diff2[keep2], [0, 33.3, 66.7, 100])
    print("     %-22s %6s %10s %10s %10s %10s" % ("长度差分层", "n", "L→L", "L→MT", "Δ", "chrF均"))
    for i in range(3):
        lo, hi = q[i], q[i + 1]
        sel = keep2 & (diff2 >= lo) & (diff2 <= hi if i == 2 else diff2 < hi)
        if sel.sum() < 20:
            continue
        a = 100 * OL[:, sel].mean()
        b = 100 * OM[:, sel].mean()
        print("     [%+5.1f, %+5.1f) 字符 %6d %9.2f%% %9.2f%% %+9.2f %10.1f"
              % (lo, hi, sel.sum(), a, b, b - a, ss2[sel].mean()))

    # ---------- 长度固定窗口：只取 |长度差| <= 2 的句子 ----------
    near = keep2 & (np.abs(diff2) <= 2)
    if near.sum() >= 50:
        a = 100 * OL[:, near].mean()
        b = 100 * OM[:, near].mean()
        print()
        print("  **只取长度几乎相同（|MT−LOC| ≤ 2 字符）的句子**")
        print("     n = %d   L→L %.2f%%   L→MT %.2f%%   **Δ = %+.2f**"
              % (near.sum(), a, b, b - a))
        # 在这批句子里再按 chrF 分半
        med = np.median(ss2[near])
        for tag, s2 in [("低 chrF 半", near & (ss2 < med)), ("高 chrF 半", near & (ss2 >= med))]:
            if s2.sum() < 20:
                continue
            a2 = 100 * OL[:, s2].mean()
            b2 = 100 * OM[:, s2].mean()
            print("       %s  n=%4d  L→L %6.2f%%  L→MT %6.2f%%  Δ %+7.2f"
                  % (tag, s2.sum(), a2, b2, b2 - a2))
    SUM.append(dict(lang=lang, mt_minus_loc=round(float(diff[keep].mean()), 2),
                    r_mtlen=round(float(np.corrcoef(ss[keep], M_len[keep])[0, 1]), 3),
                    r_loclen=round(float(np.corrcoef(ss[keep], L_len[keep])[0, 1]), 3),
                    r_diff=round(float(np.corrcoef(ss[keep], diff[keep])[0, 1]), 3)))

pd.DataFrame(SUM).to_csv(os.path.join(SP, "table_chrf_targetlen.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_chrf_targetlen.csv")

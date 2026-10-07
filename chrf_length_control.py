# -*- coding: utf-8 -*-
"""chrF 的句长混淆检验。

审稿意见：Table 5d 的 Q1 即使在 L->L 下也只有约 74%，其他分位约 80-82%，
说明低 chrF 句子本身更短或更难，句长混淆未控制。

本脚本：
  1. 报告每个 chrF 分位的源句长（英文字符数 / 词数）分布
  2. 按源句长分层后，重新计算 chrF 分位的 L->MT 下降
  3. 给出 chrF 与句长的相关系数
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
    p = float(np.mean([st[n][0] / st[n][1] if st[n][1] else 0.0 for n in st]))
    r = float(np.mean([st[n][0] / st[n][2] if st[n][2] else 0.0 for n in st]))
    return 0.0 if p + r == 0 else (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r) * 100.0


def rd(loc, root):
    tr, te, src = [], [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            tr.append(r["utt"])
        elif r["partition"] == "test":
            te.append(r["utt"])
            src.append(r.get("utt_en") or "")
    return tr, te, src


# 英文源句（用于句长）
en_src = {}
for line in open(os.path.join(MASSIVE, "en-US.jsonl"), encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r["partition"] == "test":
        en_src[r["id"]] = r["utt"]

print("=" * 100)
print("chrF 的句长混淆检验")
print("=" * 100)

for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte, _ = rd(loc, MASSIVE)
    mtr, mte, _ = rd(loc, MTDIR)
    n = min(len(lte), len(mte))
    lte, mte = lte[:n], mte[:n]

    # 取英文源句长（按 id 顺序对应）
    ids = []
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "test":
            ids.append(r["id"])
    L = np.array([len(en_src.get(i, "")) for i in ids[:n]], dtype=float)
    W = np.array([len(en_src.get(i, "").split()) for i in ids[:n]], dtype=float)
    ss = np.array([chrf_sent(a, b) for a, b in zip(mte, lte)])

    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])

    print()
    print("─" * 100)
    print("%s（固定子集 n=%d）" % (nm, keep.sum()))
    print("─" * 100)
    print("  chrF 与英文源句长的相关: r(字符) = %+.3f   r(词) = %+.3f"
          % (np.corrcoef(ss[keep], L[keep])[0, 1], np.corrcoef(ss[keep], W[keep])[0, 1]))
    print()

    # 1. 各 chrF 分位的源句长
    qs = np.percentile(ss[keep], [0, 25, 50, 75, 100])
    print("  %-18s %6s %12s %12s %10s" % ("chrF 分位", "n", "源字符 均值", "源词 均值", "chrF 均值"))
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = keep & (ss >= lo) & (ss <= hi if i == 3 else ss < hi)
        if sel.sum() == 0:
            continue
        print("  Q%d [%5.1f,%5.1f) %6d %12.1f %12.1f %10.1f"
              % (i + 1, lo, hi, sel.sum(), L[sel].mean(), W[sel].mean(), ss[sel].mean()))

    # 2. 句长分层后再看 chrF 效应
    print()
    print("  【句长分层后的 chrF 分位准确率】按英文源句词数分成三层")
    wq = np.percentile(W[keep], [0, 33.3, 66.7, 100])
    for j in range(3):
        band = keep & (W >= wq[j]) & (W <= wq[j + 1] if j == 2 else W < wq[j + 1])
        if band.sum() < 50:
            continue
        inner = np.percentile(ss[band], [0, 50, 100])
        lo_half = band & (ss < inner[1])
        hi_half = band & (ss >= inner[1])
        print("    句长层%d [%.0f,%.0f) 词  n=%4d   低 chrF 半 n=%4d (chrF %.1f)   高 chrF 半 n=%4d (chrF %.1f)"
              % (j + 1, wq[j], wq[j + 1], band.sum(),
                 lo_half.sum(), ss[lo_half].mean(), hi_half.sum(), ss[hi_half].mean()))

    # 3. 用 L->L 准确率本身做对照：低 chrF 句是不是本来就难
    print()
    print("  【对照】各 chrF 分位的 L->L 准确率（若句子本身更难，L->L 也会低）")
    lp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_localized_in-language_%s_TextCNN_char_s*.csv" % lang)))
    mp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_mt-test_in-language_%s_TextCNN_char_s*.csv" % lang)))
    if not lp or not mp:
        print("     缺 pred 文件")
        continue
    OL, OM = [], []
    for p in lp:
        d = pd.read_csv(p)
        if len(d) == n:
            OL.append((d["pred"].values == d["gold"].values))
    for p in mp:
        d = pd.read_csv(p)
        if len(d) == n:
            OM.append((d["pred"].values == d["gold"].values))
    OL, OM = np.array(OL), np.array(OM)
    m = min(OL.shape[1], OM.shape[1], len(keep))
    OL, OM, keep2, ss2, L2, W2 = OL[:, :m], OM[:, :m], keep[:m], ss[:m], L[:m], W[:m]
    print("     %-16s %6s %10s %10s %10s" % ("chrF 分位", "n", "L->L", "L->MT", "Δ"))
    q2 = np.percentile(ss2[keep2], [0, 25, 50, 75, 100])
    for i in range(4):
        lo, hi = q2[i], q2[i + 1]
        sel = keep2 & (ss2 >= lo) & (ss2 <= hi if i == 3 else ss2 < hi)
        if sel.sum() == 0:
            continue
        a = 100.0 * OL[:, sel].mean()
        b = 100.0 * OM[:, sel].mean()
        print("     Q%d [%5.1f,%5.1f) %6d %9.2f%% %9.2f%% %+9.2f"
              % (i + 1, lo, hi, sel.sum(), a, b, b - a))

pd.DataFrame([{"note": "see stdout"}]).to_csv(
    os.path.join(SP, "_chrf_len_check.csv"), index=False)
print()
print("完成。")

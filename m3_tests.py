# -*- coding: utf-8 -*-
"""M3：McNemar（话语级，配对）与 Wilcoxon（种子级配对）。"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
SEEDS = [42, 43, 44, 45, 46]


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def preds(lang, variant, model="TextCNN", tok="char"):
    out = {}
    mask = "_mask" if model == "BiLSTM" else ""
    for d in ["runs_e4_char", "runs_ws_extra", "runs_c2"]:
        pat = os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv"
                           % (variant, lang, model, tok, mask))
        for p in sorted(glob.glob(pat)):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values).astype(int)
    return out


def mcnemar_exact(b, c):
    """双侧精确 McNemar（二项检验，p=0.5）。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(x, y):
    """配对 Wilcoxon 符号秩（正态近似，含零差处理）。"""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    r = np.argsort(np.argsort(np.abs(d))) + 1
    W = r[d > 0].sum()
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sd
    from math import erfc
    p = erfc(abs(z) / np.sqrt(2))
    return float(W), float(p)


def fixed_keep(lang, loc):
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = min(len(lte), len(mte))
    ls, ms = set(ltr), set(mtr)
    return np.array([(lte[i] not in ls) and (mte[i] not in ms) for i in range(n)])


print("=" * 100)
print("M3  承诺的统计检验")
print("=" * 100)

rows = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    keep = fixed_keep(lang, loc)
    print()
    print("─" * 100)
    print("%s   固定评测子集 n = %d" % (nm, keep.sum()))
    print("─" * 100)
    for model in ["TextCNN", "BiLSTM"]:
        C = {v: preds(lang, v, model) for v in ["localized", "mt-test", "mt", "mt-train"]}
        if any(len(C[v]) < 3 for v in C):
            print("   %-8s pred 不足，跳过" % model)
            continue
        seeds = sorted(set.intersection(*[set(C[v]) for v in C]))
        n = min([C[v][s].shape[0] for v in C for s in seeds] + [len(keep)])
        ok = keep[:n]
        # 多数投票（跨种子）作为话语级结果
        def maj(v):
            M = np.array([C[v][s][:n] for s in seeds])
            return (M.mean(0) > 0.5).astype(int)
        LL, LM, MM, ML = maj("localized"), maj("mt-test"), maj("mt"), maj("mt-train")
        print()
        print("   %s" % model)
        for label, a, b in [("L→L vs L→MT  (测试侧成本)", LL, LM),
                            ("L→L vs MT→L  (训练侧成本)", LL, ML),
                            ("MT→MT vs MT→L (夸大量)", MM, ML)]:
            # b = a 对 b 错；c = a 错 b 对
            bb = int(((a == 1) & (b == 0)).sum())
            cc = int(((a == 0) & (b == 1)).sum())
            p = mcnemar_exact(bb, cc)
            print("     %-28s  b=%4d  c=%4d  p=%.3g  %s"
                  % (label, bb, cc, p, "显著" if p < 0.05 else "不显著"))
            rows.append(dict(lang=lang, model=model, contrast=label, b=bb, c=cc, p=p))
        # Wilcoxon（种子级配对）
        acc = {v: [100 * C[v][s][:n][ok].mean() for s in seeds] for v in C}
        for label, ka, kb in [("L→L vs L→MT", "localized", "mt-test"),
                              ("L→L vs MT→L", "localized", "mt-train"),
                              ("MT→MT vs MT→L", "mt", "mt-train")]:
            W, p = wilcoxon(acc[ka], acc[kb])
            print("     %-28s  W=%4.1f  p=%.3g  （种子级）" % (label + " [Wilcoxon]", W, p))
            rows.append(dict(lang=lang, model=model, contrast=label + " [Wilcoxon]",
                             W=W, p=p))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(SP, "table_stats.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_stats.csv（%d 行）" % len(out))

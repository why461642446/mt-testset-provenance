# -*- coding: utf-8 -*-
"""M3 修正：Wilcoxon 用精确分布（n=5 时正态近似偏乐观）。"""
import glob
import json
import os
import sys
from itertools import product
from math import comb

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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def wilcoxon_exact(d):
    """精确双侧 Wilcoxon 符号秩（枚举 2^n 种符号组合）。"""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n < 1:
        return float("nan"), float("nan")
    r = np.argsort(np.argsort(np.abs(d))) + 1
    Wp = r[d > 0].sum()
    Wm = r[d < 0].sum()
    W = min(Wp, Wm)
    # 零假设下所有 2^n 个符号组合等概率
    cnt = 0
    for signs in product([1, -1], repeat=n):
        wp = r[np.array(signs) > 0].sum()
        wm = r[np.array(signs) < 0].sum()
        if min(wp, wm) <= W:
            cnt += 1
    return float(W), cnt / (2 ** n)


def fixed_keep(lang, loc):
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = min(len(lte), len(mte))
    ls, ms = set(ltr), set(mtr)
    return np.array([(lte[i] not in ls) and (mte[i] not in ms) for i in range(n)])


print("=" * 104)
print("M3  承诺的统计检验（Wilcoxon 用精确分布）")
print("=" * 104)
rows = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    keep = fixed_keep(lang, loc)
    print()
    print("─" * 104)
    print("%s   固定子集 n = %d" % (nm, keep.sum()))
    print("─" * 104)
    for model in ["TextCNN", "BiLSTM"]:
        C = {v: preds(lang, v, model) for v in ["localized", "mt-test", "mt", "mt-train"]}
        if any(len(C[v]) < 3 for v in C):
            continue
        seeds = sorted(set.intersection(*[set(C[v]) for v in C]))
        n = min([C[v][s].shape[0] for v in C for s in seeds] + [len(keep)])
        ok = keep[:n]

        def maj(v):
            M = np.array([C[v][s][:n] for s in seeds])
            return (M.mean(0) > 0.5).astype(int)
        LL, LM, MM, ML = maj("localized"), maj("mt-test"), maj("mt"), maj("mt-train")
        print()
        print("   %s   （%d 个种子）" % (model, len(seeds)))
        for label, a, b in [("测试侧成本  L→L vs L→MT", LL, LM),
                            ("训练侧成本  L→L vs MT→L", LL, ML),
                            ("夸大量      MT→MT vs MT→L", MM, ML)]:
            bb = int(((a == 1) & (b == 0)).sum()); cc = int(((a == 0) & (b == 1)).sum())
            p = mcnemar_exact(bb, cc)
            print("     McNemar  %-26s b=%4d c=%4d  p=%.3g" % (label, bb, cc, p))
            rows.append(dict(lang=lang, model=model, test="McNemar", contrast=label,
                             b=bb, c=cc, p=p))
        acc = {v: [100 * C[v][s][:n][ok].mean() for s in seeds] for v in C}
        for label, ka, kb in [("测试侧成本", "localized", "mt-test"),
                              ("训练侧成本", "localized", "mt-train"),
                              ("夸大量", "mt", "mt-train")]:
            d = np.array(acc[ka]) - np.array(acc[kb])
            W, p = wilcoxon_exact(d)
            print("     Wilcoxon %-26s W=%4.1f  **p=%.4f**  (n=%d, 差 %s)"
                  % (label, W, p, len(seeds), np.round(d, 2).tolist()))
            rows.append(dict(lang=lang, model=model, test="Wilcoxon", contrast=label,
                             W=W, p=p, n=len(seeds)))

pd.DataFrame(rows).to_csv(os.path.join(SP, "table_stats.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_stats.csv")
print()
print("  说明：n=5 时双侧精确 Wilcoxon 的最小可能 p = 2/2^5 = 0.0625，因此即使 5 个种子全部同向也无法达到 0.05。")

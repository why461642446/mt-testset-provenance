# -*- coding: utf-8 -*-
"""E2：给 mBERT / XLM-R 的夸大量补 bootstrap 区间（固定子集，3 seeds）。"""
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
SEEDS = [42, 43, 44]
NB = 2000
RNG = np.random.default_rng(20240501)


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def get(d, lang, variant, model):
    pat = os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_subword_s*.csv" % (variant, lang, model))
    out = {}
    for p in sorted(glob.glob(pat)):
        s = int(p.rsplit("_s", 1)[1].split(".")[0])
        if s in SEEDS:
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


def boot(a, b, mask, nboot=NB):
    idx = np.where(mask)[0]
    av = a[:, idx].mean(axis=0)
    bv = b[:, idx].mean(axis=0)
    dd = (bv - av) * 100
    n = len(idx)
    bs = np.empty(nboot)
    for i in range(nboot):
        s = RNG.integers(0, n, n)
        bs[i] = dd[s].mean()
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("=" * 100)
print("E2  预训练编码器的 bootstrap 区间（固定评测子集，3 seeds，重采样测试句 %d 次）" % NB)
print("=" * 100)

for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])
    print()
    print("─" * 100)
    print("%s  固定子集 n = %d" % (nm, keep.sum()))
    print("─" * 100)
    for d, model in [("runs_v4a", "mBERT"), ("runs_v4c", "XLMR")]:
        L = get(d, lang, "localized", model)
        M = get(d, lang, "mt-test", model)
        A = get(d, lang, "mt", model)
        T = get(d, lang, "mt-train", model)
        if not (L and M and A and T):
            print("   %-8s 缺 pred（%d/%d/%d/%d）" % (model, len(L), len(M), len(A), len(T)))
            continue
        seeds = sorted(set(L) & set(M) & set(A) & set(T))
        m = min([x.shape[0] for x in list(L.values()) + list(M.values()) + list(A.values()) + list(T.values())] + [len(keep)])
        ok = keep[:m]
        cL = np.array([L[s][:m] for s in seeds])
        cM = np.array([M[s][:m] for s in seeds])
        cA = np.array([A[s][:m] for s in seeds])
        cT = np.array([T[s][:m] for s in seeds])
        print()
        print("   %s  seeds %s  n=%d" % (model, seeds, ok.sum()))
        print("      L→L %6.2f   L→MT %6.2f   MT→MT %6.2f   MT→L %6.2f"
              % (100 * cL[:, ok].mean(), 100 * cM[:, ok].mean(), 100 * cA[:, ok].mean(), 100 * cT[:, ok].mean()))
        for name, a, b in [("L→MT 代价", cL, cM), ("**夸大量**", cT, cA), ("MT→L 代价", cL, cT)]:
            dd, lo, hi = boot(a, b, ok)
            print("      %-14s %+7.2f  [%+7.2f, %+7.2f]" % (name, dd, lo, hi))

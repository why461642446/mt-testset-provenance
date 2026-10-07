# -*- coding: utf-8 -*-
"""v4b：固定评测子集上的 MT->MT2 分析 + bootstrap 区间。"""
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


def get(lang, variant, model):
    out = {}
    for d in ["runs_v4b", "runs_e4_char", "runs_ws_extra"]:
        mask = "_mask" if model == "BiLSTM" else ""
        pat = os.path.join(SP, d,
                           "pred_massive_%s_in-language_%s_%s_char%s_s*.csv"
                           % (variant, lang, model, mask))
        for p in sorted(glob.glob(pat)):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


def boot(a, b, mask, nboot=NB):
    idx = np.where(mask)[0]
    dd = ((b[:, idx].mean(0)) - (a[:, idx].mean(0))) * 100
    n = len(idx)
    bs = np.empty(nboot)
    for i in range(nboot):
        s = RNG.integers(0, n, n)
        bs[i] = dd[s].mean()
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("=" * 100)
print("v4b  第二个 MT 系统（600M 训练 → 3.3B 测试）")
print("=" * 100)

rows = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])
    print()
    print("─" * 100)
    print("%s   固定子集 n = %d" % (nm, keep.sum()))
    print("─" * 100)
    for model in ["TextCNN", "BiLSTM"]:
        cells = {}
        for var in ["mt", "mt2", "mt-train"]:
            d = get(lang, var, model)
            if len(d) < 3:
                cells[var] = None
                continue
            m = min(x.shape[0] for x in d.values())
            cells[var] = np.array([d[s][:m] for s in SEEDS])
        if any(v is None for v in cells.values()):
            print("   %-8s 缺 pred" % model); continue
        m = min(v.shape[1] for v in cells.values())
        m = min(m, len(keep))
        ok = keep[:m]
        A, B2, L = cells["mt"][:, :m], cells["mt2"][:, :m], cells["mt-train"][:, :m]
        a, b, l = 100 * A[:, ok].mean(), 100 * B2[:, ok].mean(), 100 * L[:, ok].mean()
        infl1 = a - l
        infl2 = b - l
        d1, lo1, hi1 = boot(L, A, ok)
        d2, lo2, hi2 = boot(L, B2, ok)
        dd, dlo, dhi = boot(A, B2, ok)
        print()
        print("   %s   (n=%d, 3 seeds)" % (model, ok.sum()))
        print("      MT→MT   %6.2f    MT→MT2  %6.2f    MT→L  %6.2f" % (a, b, l))
        print("      夸大量(同系统)  %+6.2f  [%+6.2f, %+6.2f]" % (d1, lo1, hi1))
        print("      夸大量(跨系统)  %+6.2f  [%+6.2f, %+6.2f]" % (d2, lo2, hi2))
        print("      **MT→MT2 − MT→MT**  %+6.2f  [%+6.2f, %+6.2f]   -> 缩减 %.0f%%"
              % (dd, dlo, dhi, 100 * dd / d1 if d1 else float("nan")))
        rows.append(dict(lang=lang, model=model, mt=a, mt2=b, mtL=l,
                         infl_same=round(d1, 2), infl_cross=round(d2, 2),
                         same_system_gain=round(dd, 2),
                         shrink_pct=round(100 * dd / d1, 1) if d1 else None))

pd.DataFrame(rows).to_csv(os.path.join(SP, "table_v4b.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_v4b.csv")

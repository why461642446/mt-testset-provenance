# -*- coding: utf-8 -*-
"""固定评测子集上的完整 2x2（char 与 whitespace），5 seeds，pred 文件逐句配对。"""
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
DIRS = ["runs_e4_char", "runs_ws_extra", "runs_c2", "runs_v2_kaggle", "_ws_pred_tmp"]


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def get(lang, variant, model, tok):
    """返回 {seed: 正确性数组}"""
    pat = "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv" % (
        variant, lang, model, tok, "_mask" if model == "BiLSTM" else "")
    out = {}
    for d in DIRS:
        for p in sorted(glob.glob(os.path.join(SP, d, pat))):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


def acc5(d):
    """5 个 seed 的平均准确率与样本标准差（返回 None 若不足 5）"""
    if len(d) < 5:
        return None
    a = np.array([100.0 * d[s].mean() for s in SEEDS])
    return a.mean(), a.std(ddof=1)


print("=" * 104)
print("固定评测子集上的 2x2（逐句配对，5 seeds，sd = ddof=1）")
print("=" * 104)
RES = {}
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])
    print()
    print("─" * 104)
    print("%s   固定子集 n = %d / %d" % (nm, keep.sum(), n))
    print("─" * 104)
    for tok in ["char", "whitespace"]:
        for model in ["TextCNN", "BiLSTM"]:
            cells = {}
            for name, var in [("L->L", "localized"), ("L->MT", "mt-test"),
                              ("MT->MT", "mt"), ("MT->L", "mt-train")]:
                d = get(lang, var, model, tok)
                if len(d) < 5:
                    cells[name] = None
                    continue
                a, s = acc5({k: v[keep] for k, v in d.items()})
                cells[name] = (a, s)
            if any(v is None for v in cells.values()):
                print("   %-11s %-8s  —— 缺 pred（%s）" % (
                    tok, model, {k: (0 if v is None else 1) for k, v in cells.items()}))
                continue
            print("   %-11s %-8s  L->L %6.2f±%.2f   L->MT %6.2f±%.2f   MT->MT %6.2f±%.2f   MT->L %6.2f±%.2f"
                  % (tok, model, cells["L->L"][0], cells["L->L"][1],
                     cells["L->MT"][0], cells["L->MT"][1],
                     cells["MT->MT"][0], cells["MT->MT"][1],
                     cells["MT->L"][0], cells["MT->L"][1]))
            RES[(lang, tok, model)] = cells
            if tok == "whitespace":
                infl = cells["MT->MT"][0] - cells["MT->L"][0]
                print("               → 空格夸大量 (MT->MT − MT->L) = %+.2f" % infl)
            else:
                infl = cells["MT->MT"][0] - cells["MT->L"][0]
                print("               → 字符夸大量 (MT->MT − MT->L) = %+.2f" % infl)

import pickle
pickle.dump({k: {c: (None if v is None else v) for c, v in val.items()} for k, val in RES.items()},
            open(os.path.join(SP, "_fixed52.pkl"), "wb"))
print()
print("已缓存到 _fixed52.pkl")

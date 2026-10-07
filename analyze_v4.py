# -*- coding: utf-8 -*-
"""v4a (mBERT) / v4c (XLM-R) 的字符级 2x2 分析。

同时给全量测试集与固定评测子集两套数，并与非预训练模型并列。
"""
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


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


FRAMES = {}
for d in ["runs_v4a", "runs_v4c"]:
    p = os.path.join(SP, d, "results_v2.csv")
    if os.path.exists(p):
        FRAMES[d] = pd.read_csv(p)

VARIANTS = [("localized", "L→L"), ("mt-test", "L→MT"), ("mt", "MT→MT"), ("mt-train", "MT→L")]

print("=" * 104)
print("一、全量测试集（2,974 句）—— 直接来自 results_v2.csv，含逐 seed")
print("=" * 104)
for d, df in FRAMES.items():
    model = df["model"].iloc[0]
    print()
    print("  %s  (%s)" % (d, model))
    print("  %-6s %-8s %10s %10s %10s %10s" % ("lang", "变体", "mean", "sd(ddof1)", "min seed", "max seed"))
    for lang in ["ko", "zh"]:
        for var, tag in VARIANTS:
            s = df[(df.test_lang == lang) & (df.data_variant == var)].sort_values("seed")
            a = s.accuracy.values * 100
            print("  %-6s %-8s %10.2f %10.2f %10.2f %10.2f   seeds=%s"
                  % (lang, tag, a.mean(), a.std(ddof=1) if len(a) > 1 else 0,
                     a.min(), a.max(), list(s.seed)))

print()
print("=" * 104)
print("二、全量测试集上的关键对比")
print("=" * 104)
print("  %-14s %-6s %8s %8s %8s %8s   %10s %10s" % (
    "checkpoint", "lang", "L→L", "L→MT", "MT→MT", "MT→L", "L→MT代价", "夸大量"))
summary = []
for d, df in FRAMES.items():
    model = df["model"].iloc[0]
    for lang in ["ko", "zh"]:
        v = {}
        for var, tag in VARIANTS:
            s = df[(df.test_lang == lang) & (df.data_variant == var)]
            v[tag] = (s.accuracy.values * 100).mean()
        gap = v["L→MT"] - v["L→L"]
        infl = v["MT→MT"] - v["MT→L"]
        print("  %-14s %-6s %8.2f %8.2f %8.2f %8.2f   %+10.2f %+10.2f"
              % (model, lang, v["L→L"], v["L→MT"], v["MT→MT"], v["MT→L"], gap, infl))
        summary.append(dict(checkpoint=model, lang=lang, **v, gap=gap, inflation=infl))

# 非预训练参照（已有，从 runs_loader 取）
sys.path.insert(0, SP)
from runs_loader import load_runs
R, _ = load_runs()
D = R[(R["model"].isin(["TextCNN", "BiLSTM"])) & (R["mode"] == "in-language")]
D = D[(D["tokenizer"] == "char") & ((D["model"] != "BiLSTM") | (D.bilstm_mask == True))]
print()
print("  --- 非预训练参照（字符级，全量测试集，5 seeds）---")
for m in ["TextCNN", "BiLSTM"]:
    for lang in ["ko", "zh"]:
        v = {}
        for var, tag in VARIANTS:
            s = D[(D["model"] == m) & (D.test_lang == lang) & (D.data_variant == var)]
            v[tag] = (s.accuracy.values * 100).mean() if len(s) else float("nan")
        gap = v["L→MT"] - v["L→L"]
        infl = v["MT→MT"] - v["MT→L"]
        print("  %-14s %-6s %8.2f %8.2f %8.2f %8.2f   %+10.2f %+10.2f"
              % (m, lang, v["L→L"], v["L→MT"], v["MT→MT"], v["MT→L"], gap, infl))
        summary.append(dict(checkpoint=m, lang=lang, **v, gap=gap, inflation=infl))

pd.DataFrame(summary).to_csv(os.path.join(SP, "table_v4_pretrained_2x2.csv"),
                             index=False, encoding="utf-8")

# ---------------- 固定评测子集 ----------------
print()
print("=" * 104)
print("三、固定评测子集上的 2x2（逐句配对）")
print("=" * 104)
import glob
PRIORITY = ["runs_v4a", "runs_v4c", "runs_e4_char", "runs_ws_extra", "runs_c2"]


def get_pred(d, lang, variant, model, seed):
    pat = "pred_massive_%s_in-language_%s_%s_subword_s%d.csv" % (variant, lang, model, seed)
    p = os.path.join(SP, d, pat)
    return pd.read_csv(p) if os.path.exists(p) else None


rows = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])
    print()
    print("  %s  固定子集 n = %d / %d" % (nm, keep.sum(), n))
    for d in ["runs_v4a", "runs_v4c"]:
        df = FRAMES[d]
        model = df["model"].iloc[0]
        v, sds = {}, {}
        ok_all = True
        for var, tag in VARIANTS:
            accs = []
            for seed in SEEDS:
                pr = get_pred(d, lang, var, model, seed)
                if pr is None:
                    ok_all = False
                    break
                accs.append(100.0 * pr["pred"].values[keep][:keep.sum()].__eq__(
                    pr["gold"].values[keep][:keep.sum()]).mean())
            if not ok_all:
                break
            v[tag] = float(np.mean(accs))
            sds[tag] = float(np.std(accs, ddof=1))
        if not ok_all:
            print("     %-8s 缺 pred" % model)
            continue
        gap = v["L→MT"] - v["L→L"]
        infl = v["MT→MT"] - v["MT→L"]
        print("     %-8s L→L %6.2f±%.2f  L→MT %6.2f±%.2f  MT→MT %6.2f±%.2f  MT→L %6.2f±%.2f"
              % (model, v["L→L"], sds["L→L"], v["L→MT"], sds["L→MT"],
                 v["MT→MT"], sds["MT→MT"], v["MT→L"], sds["MT→L"]))
        print("              → L→MT 代价 %+.2f    夸大量 %+.2f" % (gap, infl))
        rows.append(dict(lang=lang, model=model, subset="fixed", **v, gap=gap, inflation=infl))

pd.DataFrame(rows).to_csv(os.path.join(SP, "table_v4_fixed_2x2.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_v4_pretrained_2x2.csv 与 table_v4_fixed_2x2.csv")

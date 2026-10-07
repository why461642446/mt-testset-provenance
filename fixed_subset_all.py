# -*- coding: utf-8 -*-
"""在【固定评测子集】上重算所有头条量，统一口径。

固定子集：本地化文本 ∉ 本地化训练集 且 MT 文本 ∉ MT 训练集。
"""
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
LOC = {"ko": "ko-KR", "zh": "zh-CN"}
MARK = {"TextCNN": "", "BiLSTM": "_mask"}


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


D = {}
for l, loc in LOC.items():
    a, b = rd(loc, MASSIVE)
    c, d = rd(loc, MTDIR)
    D[l] = {"ltr": set(a), "lte": b, "mtr": set(c), "mte": d}


def preds(lang, model, tok, var):
    mk = MARK[model] if model == "BiLSTM" else ""
    dirs = {"localized": ["runs_pred_loc", "runs_pred_loc_mask", "runs_e4_char"],
            "mt-test": ["runs_pred_mt", "runs_pred_mt_mask", "runs_e4_char"],
            "mt": ["runs_c1b", "runs_e4_char", "runs_pred_mt_mask"],
            "mt-train": ["runs_e4_char"]}[var]
    seen = {}
    for d in dirs:
        for p in glob.glob(os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv"
                                        % (var, lang, model, tok, mk))):
            rid = os.path.basename(p)[5:-4]
            seen.setdefault(rid, p)
            if "runs_e4_char" in p:
                seen[rid] = p
    return [seen[k] for k in sorted(seen)]


print("=" * 104)
print("固定评测子集上的全部头条量（统一口径）")
print("=" * 104)
ALL = {}
for lang in ["ko", "zh"]:
    lte, mte = D[lang]["lte"], D[lang]["mte"]
    keep = np.array([(lte[i] not in D[lang]["ltr"]) and (mte[i] not in D[lang]["mtr"])
                     for i in range(len(lte))])
    print("  %-3s 固定子集 %d / %d (%.2f%%)" % (lang, keep.sum(), len(keep), 100.0 * keep.mean()))
    for tok in ["whitespace", "char"]:
        for model in ["TextCNN", "BiLSTM"]:
            P = {}
            for var in ["localized", "mt-test", "mt", "mt-train"]:
                ps = preds(lang, model, tok, var)
                if len(ps) != 3:
                    P[var] = None
                    continue
                O = []
                for p in ps:
                    df = pd.read_csv(p)
                    if len(df) != len(lte):
                        O = None
                        break
                    O.append((df["pred"].values == df["gold"].values))
                P[var] = np.array(O) if O else None
            ALL[(lang, tok, model)] = (P, keep)
print()

print("=" * 104)
print("§2.1 修版：固定子集上的四格 + L->MT 差值")
print("=" * 104)
print("%-4s %-11s %-9s %10s %10s %10s %10s %10s" % (
    "语言", "分词", "模型", "L->L", "L->MT", "MT->MT", "MT->L", "Δ(L->MT)"))
for lang in ["ko", "zh"]:
    for tok in ["whitespace", "char"]:
        for model in ["TextCNN", "BiLSTM"]:
            P, keep = ALL[(lang, tok, model)]
            if P["localized"] is None or P["mt-test"] is None:
                continue
            f = lambda v: "—" if P[v] is None else "%9.2f" % (100.0 * P[v][:, keep].mean())
            d = (P["mt-test"][:, keep].mean(axis=1) - P["localized"][:, keep].mean(axis=1)) * 100
            print("%-4s %-11s %-9s %10s %10s %10s %10s %+10.2f" % (
                lang, tok, model, f("localized"), f("mt-test"), f("mt"), f("mt-train"), d.mean()))
    print()

print("=" * 104)
print("§2.3 修版：伪影量级（固定子集，同口径）")
print("=" * 104)
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        ws = ALL.get((lang, "whitespace", model))
        ch = ALL.get((lang, "char", model))
        if not ws or not ch:
            continue
        kw, kw2 = ws[1], ws[1]
        dw = (ws[0]["mt-test"][:, ws[1]].mean(axis=1) - ws[0]["localized"][:, ws[1]].mean(axis=1)).mean() * 100
        dc = (ch[0]["mt-test"][:, ch[1]].mean(axis=1) - ch[0]["localized"][:, ch[1]].mean(axis=1)).mean() * 100
        # 注意：两个分词方案的固定子集相同（子集只依赖文本与训练集，与分词无关）
        print("  %-3s %-9s  空格 %+7.2f   字符 %+7.2f   差 %+7.2f 点" % (lang, model, dw, dc, dw - dc))
        if dw < 0 and dc < 0:
            print("                   -> 空格相对字符 %s %.2f 点（占空格的 %.1f%%）" % (
                "夸大" if abs(dw) > abs(dc) else "缩小", abs(dw) - abs(dc),
                100.0 * (abs(dw) - abs(dc)) / abs(dw)))
print()

print("=" * 104)
print("你的第 2 点：真实场景下的夸大（同一 MT 训练模型，MT 测试 vs 本地化测试）")
print("=" * 104)
print("%-4s %-9s %10s %10s %10s" % ("语言", "模型", "MT->MT", "MT->L", "夸大"))
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        P, keep = ALL[(lang, "char", model)]
        if P["mt"] is None or P["mt-train"] is None:
            continue
        a = 100.0 * P["mt"][:, keep].mean()
        b = 100.0 * P["mt-train"][:, keep].mean()
        print("%-4s %-9s %9.2f %10.2f %+10.2f" % (lang, model, a, b, a - b))
print()
print("  以及反向：同一本地化训练模型，MT 测试 vs 本地化测试 = L->MT 差值（缩小幅度）")
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        P, keep = ALL[(lang, "char", model)]
        d = (P["mt-test"][:, keep].mean(axis=1) - P["localized"][:, keep].mean(axis=1)).mean() * 100
        print("    %-3s %-9s  %+.2f" % (lang, model, d))

# -*- coding: utf-8 -*-
"""补跑完成后：核对 36 个目标 run 是否齐，并重算表格数字。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
sys.stdout.reconfigure(encoding="utf-8")
from runs_loader import load_runs

SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
R, _ = load_runs()

TARGETS = []
for tok, variants, langs, models in [
    ("whitespace", ["localized", "mt-test", "mt"], ["ko", "zh"], ["TextCNN", "BiLSTM"]),
    ("morph", ["localized"], ["ko", "zh"], ["TextCNN", "BiLSTM"]),
    ("char", ["localized"], ["en"], ["TextCNN", "BiLSTM"]),
]:
    for v in variants:
        for lg in langs:
            for m in models:
                for s in (45, 46):
                    TARGETS.append((tok, v, lg, m, s))

D = R[(R["model"].isin(["TextCNN", "BiLSTM"])) & (R["mode"] == "in-language")]
D = D[(D["model"] != "BiLSTM") | (D["bilstm_mask"] == True)]
have = set(zip(D.tokenizer, D.data_variant, D.test_lang, D["model"], D.seed.astype(int)))

miss = [t for t in TARGETS if t not in have]
print("目标 36 个 run —— 已完成 %d，缺 %d" % (36 - len(miss), len(miss)))
for m in miss:
    print("   缺:", m)
print()

print("=" * 96)
print("重算表格（n = 5，样本标准差 ddof=1，**全量测试集**）")
print("=" * 96)


def cell(tok, var, lang, model):
    d = D[(D.tokenizer == tok) & (D.data_variant == var) & (D.test_lang == lang) & (D["model"] == model)]
    a = d.accuracy.dropna().values * 100
    return (a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, len(a), sorted(set(d.seed)))


print()
print("--- Table 6：分词消融（localized，全量测试集）---")
print("%-6s %-9s %-9s %8s %8s %4s  %s" % ("locale", "model", "tokenizer", "mean%", "sd", "n", "seeds"))
for lang in ["en", "ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        for tok in ["whitespace", "morph", "char"]:
            if lang == "en" and tok == "morph":
                continue
            m, s, n, sd = cell(tok, "localized", lang, model)
            print("%-6s %-9s %-9s %8.2f %8.3f %4d  %s" % (lang, model, tok, m, s, n, sd))

print()
print("--- Table 5b：空格 2x2（全量测试集，n=5）---")
print("%-4s %-8s %9s %9s %9s %9s" % ("lang", "model", "L->L", "L->MT", "MT->MT", "MT->L"))
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        vals = []
        for v in ["localized", "mt-test", "mt", "mt-train"]:
            m, s, n, sd = cell("whitespace", v, lang, model)
            vals.append("%6.2f±%.2f" % (m, s))
        print("%-4s %-8s %s" % (lang, model, " ".join("%9s" % x for x in vals)))

print()
print("--- 掩码 BiLSTM 的 whitespace 值（供 §4.3 / Table 6 用）---")
for lang in ["ko", "zh"]:
    m, s, n, sd = cell("whitespace", "localized", lang, "BiLSTM")
    print("   %s  whitespace(masked) = %.2f ± %.2f  n=%d" % (lang, m, s, n))

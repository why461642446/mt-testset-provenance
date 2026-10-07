# -*- coding: utf-8 -*-
"""诊断 σ 口径：找出表里/正文里用的是 ddof=0 还是 ddof=1。"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
sys.stdout.reconfigure(encoding="utf-8")
from runs_loader import load_runs

R, _ = load_runs()
print("总行数 %d" % len(R))
print()

# 审稿人点名的三处
CHECKS = [
    ("ko TextCNN whitespace (localized)", dict(test_lang="ko", model="TextCNN", tokenizer="whitespace", data_variant="localized")),
    ("ko TextCNN char (localized)",       dict(test_lang="ko", model="TextCNN", tokenizer="char", data_variant="localized")),
    ("zh TextCNN char (localized)",       dict(test_lang="zh", model="TextCNN", tokenizer="char", data_variant="localized")),
    ("ko BiLSTM masked ws (localized)",   dict(test_lang="ko", model="BiLSTM", tokenizer="whitespace", data_variant="localized", bilstm_mask=True)),
    ("zh BiLSTM masked char (localized)", dict(test_lang="zh", model="BiLSTM", tokenizer="char", data_variant="localized", bilstm_mask=True)),
    ("en TextCNN whitespace",             dict(test_lang="en", model="TextCNN", tokenizer="whitespace", data_variant="localized")),
    ("en TextCNN char",                   dict(test_lang="en", model="TextCNN", tokenizer="char", data_variant="localized")),
]

print("%-34s %3s %8s %8s %8s %8s  %s" % ("配置", "n", "ddof=0", "ddof=1", "比值", "mean%", "判定"))
for name, q in CHECKS:
    d = R.copy()
    for k, v in q.items():
        d = d[d[k] == v] if d[k].dtype == bool else d[d[k] == v]
    if "bilstm_mask" not in q:
        d = d[(d.model != "BiLSTM") | (d.bilstm_mask == False)]
    acc = d.accuracy.dropna().values * 100
    if len(acc) < 2:
        print("%-34s %3d  —— 样本不足" % (name, len(acc)))
        continue
    s0, s1 = acc.std(ddof=0), acc.std(ddof=1)
    ratio = s1 / s0 if s0 else float("nan")
    print("%-34s %3d %8.3f %8.3f %8.4f %8.2f  %s"
          % (name, len(acc), s0, s1, ratio, acc.mean(),
             "比值≈%.3f" % ratio))

print()
print("√(5/4) = %.4f   （若比值都接近它，说明表与正文混用了 ddof=0 与 ddof=1）" % np.sqrt(5 / 4))
print()
print("=== 各配置的种子覆盖（找出哪些格不是 5 seeds）===")
for tok in ["whitespace", "morph", "char"]:
    for m in ["TextCNN", "BiLSTM"]:
        d = R[(R.tokenizer == tok) & (R.model == m) & (R.data_variant == "localized") & (R.mode == "in-language")]
        if len(d):
            g = d.groupby("test_lang").seed.apply(lambda s: sorted(set(s)))
            print("  %-11s %-8s %s" % (tok, m, dict(g)))

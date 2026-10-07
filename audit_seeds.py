# -*- coding: utf-8 -*-
"""列出每个 (tokenizer, model, data_variant, lang) 格的种子覆盖与 σ。"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
sys.stdout.reconfigure(encoding="utf-8")
from runs_loader import load_runs

R, _ = load_runs()
# 注意：R.mode 拿到的是 DataFrame.mode 方法，必须用 R["mode"]
D = R[(R["mode"] == "in-language") & (R["dataset"] == "massive")].copy()
D = D[D["model"].isin(["TextCNN", "BiLSTM"])]
D = D[(D["model"] != "BiLSTM") | (D["bilstm_mask"] == True)]   # 只用掩码 BiLSTM

print("=== 非预训练、in-language、掩码 BiLSTM 的种子覆盖 ===")
print()
hdr = "%-11s %-8s %-10s %-4s %4s %-22s %8s %8s" % (
    "tokenizer", "model", "variant", "lang", "n", "seeds", "mean%", "sd(ddof=1)")
print(hdr)
print("-" * len(hdr))
rows = []
for tok in ["whitespace", "morph", "char"]:
    for m in ["TextCNN", "BiLSTM"]:
        for var in ["localized", "mt-test", "mt", "mt-train"]:
            for lang in ["en", "ko", "zh"]:
                d = D[(D.tokenizer == tok) & (D.model == m) &
                      (D.data_variant == var) & (D.test_lang == lang)]
                if not len(d):
                    continue
                seeds = sorted(set(d.seed))
                acc = d.accuracy.dropna().values * 100
                rows.append(dict(tokenizer=tok, model=m, variant=var, lang=lang,
                                 n=len(acc), seeds=seeds, mean=acc.mean(),
                                 sd=acc.std(ddof=1) if len(acc) > 1 else 0.0))
                print("%-11s %-8s %-10s %-4s %4d %-22s %8.2f %8.3f" % (
                    tok, m, var, lang, len(acc),
                    ",".join(str(s) for s in seeds), acc.mean(),
                    acc.std(ddof=1) if len(acc) > 1 else 0.0))

F = pd.DataFrame(rows)
print()
print("=== 汇总：哪些格种子不全 ===")
for tok in ["whitespace", "morph", "char"]:
    s = F[F.tokenizer == tok]
    if not len(s):
        continue
    print("  %-11s n 的取值分布: %s" % (tok, dict(s.n.value_counts().sort_index())))
    bad = s[s.n < 5]
    if len(bad):
        print("     不足 5 seeds 的格有 %d 个，例如:" % len(bad))
        for _, r in bad.head(8).iterrows():
            print("       %-8s %-10s %-3s n=%d seeds=%s" % (
                r.model, r.variant, r.lang, r.n, r.seeds))
F.to_csv(r"D:\yanjiubaogaoxiangmu\sci_paper\_sigma_audit.csv", index=False, encoding="utf-8")
print()
print("已写出 _sigma_audit.csv")

# -*- coding: utf-8 -*-
"""配对检验：本地化测试 vs 机翻测试（同一批源句，逐条配对）

为什么比之前的 Wilcoxon 更合适：
  之前把「每个微调运行」当作一个观测，只有 12 个观测，而且混合了
  两个架构、两个语言、以及一个已被判定为无信息量的中文条件。
  这里改用**真正的评测单位** —— 2,974 个测试句 —— 做配对二分类检验
  （McNemar），统计模型在同一句源句上的"对/错"是否随评测语料而改变。

要求：实验时加 --save-predictions，生成 pred_<run_id>.csv（列：idx,gold,pred）。
"""
import glob
import os
import sys

import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"


def load_preds(outdir, lang, model, seed, variant):
    rid = f"massive_{variant}_in-language_{lang}_{model}_whitespace_s{seed}"
    p = os.path.join(SP, outdir, f"pred_{rid}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def mcnemar(a, b):
    """a, b: 两个条件下的布尔正确性序列（同序配对）"""
    b01 = int(((~a) & b).sum())   # 条件1错、条件2对
    b10 = int((a & (~b)).sum())   # 条件1对、条件2错
    n = b01 + b10
    if n == 0:
        return b01, b10, None, None
    # 连续性校正
    chi2 = (abs(b10 - b01) - 1) ** 2 / n
    p = stats.chi2.sf(chi2, 1)
    return b01, b10, chi2, p


print("=" * 96)
print("配对 McNemar 检验：本地化测试 vs 机翻测试（逐句配对，2,974 个测试句）")
print("=" * 96)
print("b01 = 本地化错/机翻对   b10 = 本地化对/机翻错   两者都是模型在两种语料上表现不一致的句子数")
print()

rows = []
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        for seed in [42, 43, 44]:
            lo = load_preds("runs_pred_loc", lang, model, seed, "localized")
            mt = load_preds("runs_pred_mt", lang, model, seed, "mt-test")
            if lo is None or mt is None:
                continue
            if not (lo["idx"].equals(mt["idx"])):
                print("  !! idx 未对齐，跳过", lang, model, seed)
                continue
            a = (lo["pred"].values == lo["gold"].values)
            b = (mt["pred"].values == mt["gold"].values)
            b01, b10, chi2, p = mcnemar(a, b)
            acc_l, acc_m = a.mean() * 100, b.mean() * 100
            rows.append(dict(lang=lang, model=model, seed=seed, n=len(a),
                             acc_loc=round(acc_l, 2), acc_mt=round(acc_m, 2),
                             delta=round(acc_m - acc_l, 2),
                             b01=b01, b10=b10,
                             chi2=None if chi2 is None else round(chi2, 1),
                             p=None if p is None else float(f"{p:.3g}")))

if not rows:
    print("  未找到 pred_*.csv —— 请先用 --save-predictions 重跑本地化与 mt-test 两组")
    sys.exit(0)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print()

# 汇总：按语言合并种子
print("=" * 96)
print("按语言汇总（每个种子一次检验，报告 p 值范围）")
print("=" * 96)
for lang in sorted(df.lang.unique()):
    for model in sorted(df.model.unique()):
        s = df[(df.lang == lang) & (df.model == model)]
        if not len(s):
            continue
        print("  %-4s %-9s  n=%d seeds   准确率 %.2f -> %.2f (Δ %+.2f)   "
              "McNemar p = %s" % (
                  lang, model, len(s), s.acc_loc.mean(), s.acc_mt.mean(),
                  s.delta.mean(), ", ".join(str(x) for x in s.p)))

out = os.path.join(SP, "mcnemar_results.csv")
df.to_csv(out, index=False, encoding="utf-8")
print("\n已写出:", out)

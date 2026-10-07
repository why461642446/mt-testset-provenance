# -*- coding: utf-8 -*-
"""E4a 的 McNemar 配对检验：掩码版与未掩码版并列

配对依据：本地化与机翻测试集来自同一批源句，ID 集合与顺序均已验证一致，
因此同一 idx 在两个条件下对应同一句源句。
"""
import os
import sys

import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"


def load(d, rid):
    p = os.path.join(SP, d, f"pred_{rid}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def mcnemar(a, b):
    b01 = int(((~a) & b).sum())      # 条件1错 / 条件2对
    b10 = int((a & (~b)).sum())      # 条件1对 / 条件2错
    n = b01 + b10
    if n == 0:
        return b01, b10, None, None
    chi2 = (abs(b10 - b01) - 1) ** 2 / n
    return b01, b10, chi2, stats.chi2.sf(chi2, 1)


def run(label, loc_dir, mt_dir, suffix):
    print("=" * 90)
    print(label)
    print("=" * 90)
    print("%-4s %-3s %8s %8s %8s %8s %8s %12s" % (
        "语言", "种子", "本地化", "机翻", "Δ", "b10", "b01", "p"))
    rows = []
    for lang in ["ko", "zh"]:
        for seed in [42, 43, 44]:
            a = load(loc_dir, f"massive_localized_in-language_{lang}_BiLSTM_whitespace{suffix}_s{seed}")
            b = load(mt_dir, f"massive_mt-test_in-language_{lang}_BiLSTM_whitespace{suffix}_s{seed}")
            if a is None or b is None:
                continue
            ca = (a["pred"].values == a["gold"].values)
            cb = (b["pred"].values == b["gold"].values)
            b01, b10, chi2, p = mcnemar(ca, cb)
            acc_a, acc_b = ca.mean() * 100, cb.mean() * 100
            print("%-4s %-3d %7.2f%% %7.2f%% %+7.2f %8d %8d %12s" % (
                lang, seed, acc_a, acc_b, acc_b - acc_a, b10, b01,
                "—" if p is None else "%.2g" % p))
            rows.append(dict(lang=lang, seed=seed, acc_loc=acc_a, acc_mt=acc_b,
                             delta=acc_b - acc_a, b10=b10, b01=b01, p=p))
    df = pd.DataFrame(rows)
    print()
    for lang in ["ko", "zh"]:
        s = df[df.lang == lang]
        if len(s):
            print("  %-3s  本地化 %.2f -> 机翻 %.2f  (Δ %+.2f)   比率 b10:b01 = %.1f:1" % (
                lang, s.acc_loc.mean(), s.acc_mt.mean(), s.delta.mean(),
                s.b10.mean() / max(1, s.b01.mean())))
    print()
    return df


masked = run("掩码 BiLSTM（新跑，E4a 的可用配置）",
             "runs_pred_loc_mask", "runs_pred_mt_mask", "_mask")
plain = run("未掩码 BiLSTM（原配置，已知被填充缺陷污染）",
            "runs_pred_loc", "runs_pred_mt", "")

masked.to_csv(os.path.join(SP, "mcnemar_e4a_masked.csv"), index=False, encoding="utf-8")
plain.to_csv(os.path.join(SP, "mcnemar_e4a_unmasked.csv"), index=False, encoding="utf-8")

print("=" * 90)
print("结论对照")
print("=" * 90)
for name, d in [("掩码", masked), ("未掩码", plain)]:
    ko = d[d.lang == "ko"]
    print("  %-6s 韩语 Δ = %+.2f（逐种子 %s）" % (
        name, ko.delta.mean(), [round(x, 2) for x in ko.delta]))
print()
print("  已写: mcnemar_e4a_masked.csv / mcnemar_e4a_unmasked.csv")

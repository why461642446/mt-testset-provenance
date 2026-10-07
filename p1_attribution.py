# -*- coding: utf-8 -*-
"""P1 归因分析：按人工判定类别，看 L->L 与 L->MT 的准确率。

核心问题（审稿人第 3 条）：
  C 类（意图已变）句子是否构成残余下降的主要部分？
  更关键：**在 A 类（翻译完好）句子上，L->MT 的下降还剩多少？**
  若 A 类上仍有可观下降 -> MT 错误解释不了全部，存在真 translationese。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
P1 = os.path.join(SP, "p1_judging")
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
ENCD = {"ko": "cp949", "zh": "gbk"}
SEEDS = [42, 43, 44, 45, 46]


def test_order(loc):
    """按 jsonl 出现顺序返回测试集的 (id, utt) —— 与 experiment_v2.load_data 一致。"""
    out = []
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "test":
            out.append((str(r["id"]), r["utt"]))
    return out


def get_preds(lang, variant):
    """返回 {seed: 正确性数组}（字符级 TextCNN）"""
    out = {}
    for d in ["runs_e4_char", "runs_ws_extra", "runs_c2"]:
        pat = os.path.join(SP, d,
                           "pred_massive_%s_in-language_%s_TextCNN_char_s*.csv" % (variant, lang))
        for p in sorted(glob.glob(pat)):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


print("=" * 104)
print("P1 归因分析：人工判定类别 × 模型准确率（字符级 TextCNN，固定子集，5 seeds）")
print("=" * 104)

ROWS = []
for lang, loc, nm, enc in [("ko", "ko-KR", "Korean", "cp949"), ("zh", "zh-CN", "Chinese", "gbk")]:
    order = test_order(loc)
    idx = {i: k for k, (i, u) in enumerate(order)}
    L = get_preds(lang, "localized")
    M = get_preds(lang, "mt-test")
    seeds = sorted(set(L) & set(M))
    n_test = len(order)

    d = pd.read_csv(os.path.join(P1, "P1_%s_blind.csv" % lang), encoding=enc)
    key = pd.read_csv(os.path.join(P1, "P1_%s_key_PRIVATE.csv" % lang), encoding="utf-8")
    d["j1"] = d["judge1_category"].astype(str).str.strip().str.upper()
    d["j2"] = d["judge2_category"].astype(str).str.strip().str.upper()
    d = d.merge(key[["item_id", "group", "src_id"]], on="item_id", how="left")
    d = d[~d["group"].str.endswith("_repeat")].copy()          # 去掉重复项
    d["row"] = d["src_id"].astype(str).map(idx)
    d = d[d["row"].notna() & (d["row"] < n_test)].copy()
    d["row"] = d["row"].astype(int)

    print()
    print("═" * 104)
    print("%s   有效判定 %d 条（去掉 12 条重复项）" % (nm, len(d)))
    print("═" * 104)

    for who, col in [("评判者1", "j1"), ("评判者2", "j2")]:
        print()
        print("  【%s】" % who)
        print("  %-8s %5s %10s %10s %10s   %s" % ("类别", "n", "L→L", "L→MT", "Δ", "占全部错误的比例"))
        # 基线（全部条目）
        r = d["row"].values
        accL = np.mean([L[s][r].mean() for s in seeds]) * 100
        accM = np.mean([M[s][r].mean() for s in seeds]) * 100
        print("  %-8s %5d %9.2f%% %9.2f%% %+9.2f   —" % ("全部", len(d), accL, accM, accM - accL))
        for cat in ["A", "B", "C"]:
            sub = d[d[col] == cat]
            if len(sub) == 0:
                continue
            rr = sub["row"].values
            a = np.mean([L[s][rr].mean() for s in seeds]) * 100
            b = np.mean([M[s][rr].mean() for s in seeds]) * 100
            sd = np.mean([L[s][rr].std(ddof=0) for s in seeds]) * 0
            # 该类别贡献的"净错误数"占全部净错误数的比例
            wrongL = ((1 - np.array([L[s][rr] for s in seeds])).mean()) * len(rr)
            wrongM = ((1 - np.array([M[s][rr] for s in seeds])).mean()) * len(rr)
            print("  %-8s %5d %9.2f%% %9.2f%% %+9.2f" % (cat, len(sub), a, b, b - a))

        # 关键：只用 A 类
        subA = d[d[col] == "A"]
        if len(subA) >= 10:
            rr = subA["row"].values
            a = np.mean([L[s][rr].mean() for s in seeds]) * 100
            b = np.mean([M[s][rr].mean() for s in seeds]) * 100
            print()
            print("  **关键：只用 A 类（翻译完好）句子**")
            print("     n = %d    L→L %.2f%%   L→MT %.2f%%   **Δ = %+.2f**" % (len(subA), a, b, b - a))
            print("     （若这个 Δ 仍显著为负，说明 MT 错误解释不了全部下降）")
        ROWS.append(dict(lang=lang, judge=who, n=len(d), all_L=accL, all_M=accM, all_d=accM - accL,
                         nA=len(subA)))

pd.DataFrame(ROWS).to_csv(os.path.join(SP, "table_p1_attribution.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_p1_attribution.csv")

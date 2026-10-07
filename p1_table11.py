# -*- coding: utf-8 -*-
"""C-1：P1 的 A 类 / (B+C) 类成本 + bootstrap 区间。"""
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
SEEDS = [42, 43, 44, 45, 46]
NB = 20000
RNG = np.random.default_rng(20240505)


def get_preds(lang, variant):
    out = {}
    for d in ["runs_e4_char", "runs_ws_extra", "runs_c2"]:
        for p in sorted(glob.glob(os.path.join(
                SP, d, "pred_massive_%s_in-language_%s_TextCNN_char_s*.csv" % (variant, lang)))):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values)
    return out


def order_of(loc):
    o = []
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if line and json.loads(line)["partition"] == "test":
            o.append(str(json.loads(line)["id"]))
    return o


def boot_diff(L, M, rows, nboot=NB):
    """配对差 bootstrap：重采样句子"""
    l = L[:, rows].mean(0)
    m = M[:, rows].mean(0)
    d = (m - l) * 100
    n = len(rows)
    bs = np.empty(nboot)
    for i in range(nboot):
        s = RNG.integers(0, n, n)
        bs[i] = d[s].mean()
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("=" * 100)
print("C-1  Table 11 重做：A 类 与 B+C 类 的成本，含 bootstrap 95% 区间")
print("=" * 100)

summary = []
for lang, loc, nm, enc in [("ko", "ko-KR", "Korean", "cp949"), ("zh", "zh-CN", "Chinese", "gbk")]:
    order = order_of(loc)
    idx = {i: k for k, i in enumerate(order)}
    Ld, Md = get_preds(lang, "localized"), get_preds(lang, "mt-test")
    seeds = sorted(set(Ld) & set(Md))
    L = np.array([Ld[s] for s in seeds])
    M = np.array([Md[s] for s in seeds])

    d = pd.read_csv(os.path.join(SP, "p1_judging", "P1_%s_blind.csv" % lang), encoding=enc)
    key = pd.read_csv(os.path.join(SP, "p1_judging", "P1_%s_key_PRIVATE.csv" % lang), encoding="utf-8")
    d["j1"] = d["judge1_category"].astype(str).str.strip().str.upper()
    d["j2"] = d["judge2_category"].astype(str).str.strip().str.upper()
    d = d.merge(key[["item_id", "group", "src_id"]], on="item_id", how="left")
    d["row"] = d["src_id"].astype(str).map(idx)
    d = d[d["row"].notna()].copy()
    d["row"] = d["row"].astype(int)
    d = d[~d["group"].str.endswith("_repeat")]
    rand = d[d["group"] == "random"].copy()

    print()
    print("─" * 100)
    print("%s   随机样本 n = %d" % (nm, len(rand)))
    print("─" * 100)
    for who in ["j1", "j2"]:
        allr = rand["row"].values
        aA = rand[rand[who] == "A"]["row"].values
        aBC = rand[rand[who].isin(["B", "C"])]["row"].values
        d_all, lo_all, hi_all = boot_diff(L, M, allr)
        d_A, lo_A, hi_A = boot_diff(L, M, aA)
        d_BC, lo_BC, hi_BC = boot_diff(L, M, aBC)
        share = 100 * d_A / d_all
        # 份额的区间：对 (A - all) 做 bootstrap
        ratio_bs = np.empty(NB)
        for i in range(NB):
            sa = RNG.integers(0, len(aA), len(aA))
            sall = RNG.integers(0, len(allr), len(allr))
            da = ((M[:, aA].mean(0) - L[:, aA].mean(0)) * 100)[sa].mean()
            dall = ((M[:, allr].mean(0) - L[:, allr].mean(0)) * 100)[sall].mean()
            ratio_bs[i] = 100 * da / dall if dall else np.nan
        rlo, rhi = np.nanpercentile(ratio_bs, [2.5, 97.5])
        # 可归因于译文错误的份额 = (all - A) / all
        attrib = 100 * (d_all - d_A) / d_all
        print()
        print("   %s" % who)
        print("      全体      Δ %+7.2f  [%+7.2f, %+7.2f]   n=%d" % (d_all, lo_all, hi_all, len(allr)))
        print("      A 类      Δ %+7.2f  [%+7.2f, %+7.2f]   n=%d" % (d_A, lo_A, hi_A, len(aA)))
        print("      B+C 类    Δ %+7.2f  [%+7.2f, %+7.2f]   n=%d" % (d_BC, lo_BC, hi_BC, len(aBC)))
        print("      **A/全体 = %.0f%%  [%.0f%%, %.0f%%]**" % (share, rlo, rhi))
        print("      **可归因于译文错误的份额 = %.0f%%**" % attrib)
        summary.append(dict(lang=lang, judge=who, n_all=len(allr), n_A=len(aA), n_BC=len(aBC),
                            d_all=round(d_all, 2), d_A=round(d_A, 2), d_BC=round(d_BC, 2),
                            share=round(share, 1), share_lo=round(rlo, 1), share_hi=round(rhi, 1),
                            attributable=round(attrib, 1)))

pd.DataFrame(summary).to_csv(os.path.join(SP, "table11_revised.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table11_revised.csv")

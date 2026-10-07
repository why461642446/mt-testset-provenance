# -*- coding: utf-8 -*-
"""2x2（字符级）+ 固定评测子集 + 配对 bootstrap 区间。

固定子集定义（用户指定）：
  一个测试条目被保留，当且仅当
    本地化版本文本 ∉ 本地化训练集  且  MT 版本文本 ∉ MT 训练集
  四个格子都只在这个子集上评测，避免"各格各自去重"引入样本差异。
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
VAR = {"L->L": "localized", "L->MT": "mt-test", "MT->MT": "mt", "MT->L": "mt-train"}
rng = np.random.default_rng(20261004)


def rd(loc, root):
    tr, te, ids = [], [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            tr.append(r["utt"])
        elif r["partition"] == "test":
            te.append(r["utt"])
            ids.append(r.get("id", r.get("utt_id", "")))
    return tr, te, ids


for lang, loc in LOC.items():
    ltr, lte, lid = rd(loc, MASSIVE)
    mtr, mte, mid = rd(loc, MTDIR)
    # 对齐性检查
    aligned = (len(lte) == len(mte)) and all(a == b for a, b in zip(lid, mid)) if lid[0] else (len(lte) == len(mte))
    print("=" * 104)
    print("语言 %s   本地化测试 %d  MT 测试 %d   ID 顺序对齐: %s" % (lang, len(lte), len(mte), aligned))
    print("=" * 104)

    Ltr, Mtr = set(ltr), set(mtr)
    keep = np.array([(lte[i] not in Ltr) and (mte[i] not in Mtr) for i in range(len(lte))])
    print("  固定评测子集: %d / %d 条 (%.2f%%)" % (keep.sum(), len(keep), 100.0 * keep.mean()))
    print()

    # 读取四格预测
    data = {}
    for cell, var in VAR.items():
        for model in ["TextCNN", "BiLSTM"]:
            pats = sorted(glob.glob(os.path.join(
                SP, "runs_e4_char",
                "pred_massive_%s_in-language_%s_%s_char%s_s*.csv" % (var, lang, model, MARK[model]))))
            if len(pats) != 3:
                print("  !! %-6s %-9s 期望 3 个 pred，实得 %d" % (cell, model, len(pats)))
                continue
            oks = []
            for p in pats:
                df = pd.read_csv(p)
                if len(df) != len(lte):
                    print("  !! %s 长度 %d != %d" % (os.path.basename(p), len(df), len(lte)))
                    break
                oks.append((df["pred"].values == df["gold"].values))
            else:
                data[(cell, model)] = np.array(oks)      # (3, N)

    print("  %-8s %-9s %14s %14s" % ("格子", "模型", "全测试集", "固定子集"))
    acc = {}
    for cell in VAR:
        for model in ["TextCNN", "BiLSTM"]:
            if (cell, model) not in data:
                continue
            O = data[(cell, model)]
            a_all = 100.0 * O.mean()
            a_sub = 100.0 * O[:, keep].mean()
            acc[(cell, model)] = (a_all, a_sub, O)
            print("  %-8s %-9s %13.2f%% %13.2f%%" % (cell, model, a_all, a_sub))
    print()

    # 配对差值 + bootstrap（在同一固定子集上，逐句配对）
    print("  配对差值（MT 相对本地化，同句配对，固定子集）:")
    for model in ["TextCNN", "BiLSTM"]:
        if ("L->L", model) not in data:
            continue
        loc_ok = data[("L->L", model)]
        for cell in ["L->MT", "MT->MT", "MT->L"]:
            if (cell, model) not in data:
                continue
            oth = data[(cell, model)]
            # 逐 seed 配对差值（同一 seed 同一句）
            if cell == "MT->L":
                # 训练来源不同，但测试子集相同；按 seed 对齐仍有意义（同 seed 同句）
                pass
            d = (oth[:, keep].mean(axis=1) - loc_ok[:, keep].mean(axis=1)) * 100
            # bootstrap 重采样句子
            idx = np.arange(keep.sum())
            boots = []
            for _ in range(2000):
                b = rng.choice(idx, size=len(idx), replace=True)
                a = oth[:, keep][:, b].mean()
                c = loc_ok[:, keep][:, b].mean()
                boots.append((a - c) * 100)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print("    %-9s %-6s  逐seed Δ = [%s]   均值 %+.2f   95%% 区间 [%+.2f, %+.2f]"
                  % (model, cell, ", ".join("%+.2f" % x for x in d), d.mean(), lo, hi))
    print()

# -*- coding: utf-8 -*-
"""E4a 的干净估计：两侧都剔除与本地化训练集完全相同的测试句，再配对相减。
同时给出你指出的百分比校正。"""
import glob
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
LOC = {"ko": "ko-KR", "zh": "zh-CN"}


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            tr.append(r["utt"])
        elif r["partition"] == "test":
            te.append(r["utt"])
    return tr, te


D = {}
for l, loc in LOC.items():
    tr, te = rd(loc, MASSIVE)
    D[l] = {"tr": set(tr), "te": te}
    D[l]["mtr"], D[l]["mte"] = rd(loc, MTDIR)

MARK = {"TextCNN": "", "BiLSTM": "_mask"}


def analyze(lang, tok):
    print("=" * 100)
    print("E4a 干净估计 — %s / %s" % (lang, tok))
    print("=" * 100)
    lte, trset = D[lang]["te"], D[lang]["tr"]
    mte = D[lang]["mte"]
    dup_l = [t in trset for t in lte]
    dup_m = [t in trset for t in mte]
    print("  本地化测试 %d 句，其中与训练集重复 %d 句 (%.2f%%) -> 保留 %d"
          % (len(lte), sum(dup_l), 100.0 * sum(dup_l) / len(lte), len(lte) - sum(dup_l)))
    print("  机翻测试   %d 句，其中与训练集重复 %d 句 (%.2f%%) -> 保留 %d"
          % (len(mte), sum(dup_m), 100.0 * sum(dup_m) / len(mte), len(mte) - sum(dup_m)))
    print()

    for model in ["TextCNN", "BiLSTM"]:
        mk = MARK[model]
        pl = sorted(glob.glob(os.path.join(
            SP, "runs_e4_char" if tok == "char" else "runs_pred_loc",
            "pred_massive_localized_in-language_%s_%s_%s%s_s*.csv" % (lang, model, tok, mk))))
        pm = sorted(glob.glob(os.path.join(
            SP, "runs_e4_char" if tok == "char" else "runs_pred_mt_mask",
            "pred_massive_mt-test_in-language_%s_%s_%s%s_s*.csv" % (lang, model, tok, mk))))
        if not pl or not pm:
            print("  %-9s 缺少 pred 文件 (loc %d / mt %d)" % (model, len(pl), len(pm)))
            continue
        rows = []
        for a, b in zip(pl, pm):
            da, db = pd.read_csv(a), pd.read_csv(b)
            if len(da) != len(lte) or len(db) != len(mte):
                print("  %-9s 长度不符: %d/%d vs %d/%d" % (model, len(da), len(lte), len(db), len(mte)))
                continue
            oka = (da["pred"].values == da["gold"].values)
            okb = (db["pred"].values == db["gold"].values)
            dl = pd.Series(dup_l).values[:len(oka)]
            dm = pd.Series(dup_m).values[:len(okb)]
            rows.append({
                "acc_loc_all": 100.0 * oka.mean(),
                "acc_mt_all": 100.0 * okb.mean(),
                "acc_loc_clean": 100.0 * oka[~dl].mean(),
                "acc_mt_clean": 100.0 * okb[~dm].mean(),
            })
        if not rows:
            continue
        T = pd.DataFrame(rows)
        print("  %-9s 全量:      本地化 %.2f -> 机翻 %.2f   Δ %+.2f"
              % (model, T.acc_loc_all.mean(), T.acc_mt_all.mean(),
                 (T.acc_mt_all - T.acc_loc_all).mean()))
        print("  %-9s 两侧去重后: 本地化 %.2f -> 机翻 %.2f   Δ %+.2f   <== 干净估计"
              % (model, T.acc_loc_clean.mean(), T.acc_mt_clean.mean(),
                 (T.acc_mt_clean - T.acc_loc_clean).mean()))
        print()
    print()


for lang in ["ko", "zh"]:
    analyze(lang, "char")
analyze("ko", "whitespace")

print("=" * 100)
print("百分比校正：OOV 成分占朴素空格估计的比例")
print("=" * 100)
for lang, model, ws, ch in [("ko", "TextCNN", 40.16, 22.45), ("ko", "BiLSTM", 42.36, 24.20)]:
    share = 100.0 * (ws - ch) / ws
    print("  %s %-9s  空格 %.2f -> 字符 %.2f   差 %.2f   = 空格估计的 %.1f%%"
          % (lang, model, ws, ch, ws - ch, share))

# -*- coding: utf-8 -*-
"""定稿数字：5 seeds + 固定评测子集 + 配对 bootstrap + 种子间范围。

数据来源一律经 runs_loader（按 run_id 去重）。预测文件按来源优先级取。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
from runs_loader import load_runs

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
LOC = {"ko": "ko-KR", "zh": "zh-CN"}
MARK = {"TextCNN": "", "BiLSTM": "_mask"}
rng = np.random.default_rng(20261004)

R, RERUN = load_runs()
print("结果记录: %d 行（去重后）" % len(R))
print()

DIRS = ["runs_e4_char", "runs_ws_extra", "runs_pred_loc", "runs_pred_mt",
        "runs_pred_loc_mask", "runs_pred_mt_mask"]


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def get_preds(lang, model, tok, var):
    """返回 {seed: (ok_array)}，按来源优先级取唯一一份。"""
    mk = MARK[model] if model == "BiLSTM" else ""
    found = {}
    for d in DIRS:
        for p in glob.glob(os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv"
                                        % (var, lang, model, tok, mk))):
            rid = os.path.basename(p)[5:-4]
            seed = int(rid.rsplit("_s", 1)[1])
            found.setdefault(seed, p)          # DIRS 顺序即优先级
    return found


D = {}
for l, loc in LOC.items():
    a, b = rd(loc, MASSIVE)
    c, d = rd(loc, MTDIR)
    D[l] = {"ltr": set(a), "lte": b, "mtr": set(c), "mte": d}
    D[l]["keep"] = np.array([(b[i] not in D[l]["ltr"]) and (d[i] not in D[l]["mtr"])
                             for i in range(len(b))])
    print("%-3s 固定子集 %d / %d (%.2f%%)" % (l, D[l]["keep"].sum(), len(b), 100.0 * D[l]["keep"].mean()))
print()

CELLS = ["localized", "mt-test", "mt", "mt-train"]
LABEL = {"localized": "L->L", "mt-test": "L->MT", "mt": "MT->MT", "mt-train": "MT->L"}


def collect(lang, tok, model):
    out = {}
    for var in CELLS:
        pr = get_preds(lang, model, tok, var)
        if not pr:
            out[var] = None
            continue
        keep = D[lang]["keep"]
        accs, oks = {}, {}
        for s, p in sorted(pr.items()):
            df = pd.read_csv(p)
            if len(df) != len(D[lang]["lte"]):
                continue
            ok = (df["pred"].values == df["gold"].values)
            oks[s] = ok
            accs[s] = 100.0 * ok[keep].mean()
        out[var] = (oks, accs)
    return out


print("=" * 108)
print("定稿：字符级 2x2（固定子集，5 seeds）")
print("=" * 108)
print("%-4s %-9s %8s %8s %8s %8s   %-28s" % (
    "语言", "模型", "L->L", "L->MT", "MT->MT", "MT->L", "各格 5-seed 范围（L->MT）"))
FINAL = {}
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        C = collect(lang, "char", model)
        if not all(C[v] for v in CELLS):
            print("%-4s %-9s 数据不全: %s" % (lang, model,
                  [v for v in CELLS if not C[v]]))
            continue
        m = {v: np.mean(list(C[v][1].values())) for v in CELLS}
        FINAL[(lang, "char", model)] = (C, m)
        rng5 = sorted(C["mt-test"][1].values())
        print("%-4s %-9s %8.2f %8.2f %8.2f %8.2f   [%.2f .. %.2f]"
              % (lang, model, m["localized"], m["mt-test"], m["mt"], m["mt-train"],
                 rng5[0], rng5[-1]))
print()

print("=" * 108)
print("膨胀（重新定义）：同一 MT 训练模型，MT 测试 vs 本地化测试")
print("=" * 108)
print("%-4s %-9s %10s %10s %10s %-26s" % ("语言", "模型", "MT->MT", "MT->L", "夸大", "逐 seed 夸大"))
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        C, m = FINAL[(lang, "char", model)]
        seeds = sorted(set(C["mt"][1]) & set(C["mt-train"][1]))
        per = [C["mt"][1][s] - C["mt-train"][1][s] for s in seeds]
        print("%-4s %-9s %10.2f %10.2f %+10.2f   [%s]" % (
            lang, model, m["mt"], m["mt-train"], m["mt"] - m["mt-train"],
            ", ".join("%+.2f" % x for x in per)))
print()

print("=" * 108)
print("配对 bootstrap（固定子集，按句重采样 2000 次；不含训练随机性）")
print("=" * 108)
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        C, m = FINAL[(lang, "char", model)]
        keep = D[lang]["keep"]
        for var in ["mt-test", "mt", "mt-train"]:
            seeds = sorted(set(C["localized"][0]) & set(C[var][0]))
            A = np.array([C["localized"][0][s][keep] for s in seeds])
            B = np.array([C[var][0][s][keep] for s in seeds])
            idx = np.arange(keep.sum())
            boot = []
            for _ in range(2000):
                bi = rng.choice(idx, size=len(idx), replace=True)
                boot.append((B[:, bi].mean() - A[:, bi].mean()) * 100)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            per = (B.mean(axis=1) - A.mean(axis=1)) * 100
            print("  %-3s %-9s %-6s  Δ %+7.2f   95%%CI [%+.2f, %+.2f]   种子范围 [%+.2f, %+.2f]"
                  % (lang, model, LABEL[var], per.mean(), lo, hi, per.min(), per.max()))
    print()

print("=" * 108)
print("伪影量级（固定子集，同口径）：空格 vs 字符")
print("=" * 108)
for lang in ["ko", "zh"]:
    for model in ["TextCNN", "BiLSTM"]:
        res = {}
        for tok in ["whitespace", "char"]:
            C = collect(lang, tok, model)
            if not (C["localized"] and C["mt-test"]):
                res[tok] = None
                continue
            seeds = sorted(set(C["localized"][1]) & set(C["mt-test"][1]))
            res[tok] = np.mean([C["mt-test"][1][s] - C["localized"][1][s] for s in seeds])
        if res["whitespace"] is None or res["char"] is None:
            print("  %-3s %-9s 数据不全" % (lang, model))
            continue
        w, c = res["whitespace"], res["char"]
        print("  %-3s %-9s  空格 %+7.2f   字符 %+7.2f   差 %+7.2f" % (lang, model, w, c, w - c))
        if w < 0 and c < 0:
            print("                    -> 空格%s %.2f 点（占空格的 %.1f%%）"
                  % ("夸大" if abs(w) > abs(c) else "缩小", abs(w) - abs(c),
                     100.0 * (abs(w) - abs(c)) / abs(w)))
print()

print("=" * 108)
print("重跑一致性（用于方法部分的复现性陈述）")
print("=" * 108)
if len(RERUN):
    print("  同一 run_id 的独立重跑: %d 组" % len(RERUN))
    print("  准确率最大差异 %.6f，中位差异 %.6f" % (RERUN.spread.max(), RERUN.spread.median()))

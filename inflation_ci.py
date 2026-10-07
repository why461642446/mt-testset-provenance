# -*- coding: utf-8 -*-
"""夸大量本身的 bootstrap 置信区间。

审稿意见：Table 5a 的区间都是相对 L->L 的，不是夸大量（MT->MT 减 MT->L）本身。
本脚本对固定评测子集重采样测试句，给出：
  * 每格准确率（L->L / L->MT / MT->MT / MT->L）
  * L->MT 相对 L->L 的差
  * MT->MT 相对 MT->L 的差（= 夸大量）
  * MT->L 相对 L->L 的差（= 训练来源代价）
以及各自的 95% bootstrap 区间（重采样**测试句**，与论文口径一致）。
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
NBOOT = 2000
RNG = np.random.default_rng(20240501)


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def load_preds(lang, variant, model, seeds, tokenizer="char"):
    """返回 [(seed, 正确性数组)]。掩码 BiLSTM 的文件名带 _mask。"""
    tag = model + ("_mask" if model == "BiLSTM" else "")
    for d in ("runs_e4_char", "runs_ws_extra", "runs_c2"):
        # 实际文件名形如 ..._BiLSTM_char_mask_s42.csv -> 模型_分词_掩码
        pat = os.path.join(SP, d,
                           "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv"
                           % (variant, lang, model, tokenizer,
                              "_mask" if model == "BiLSTM" else ""))
        fps = sorted(glob.glob(pat))
        if fps:
            break
    else:
        fps = []
    out = []
    for p in fps:
        s = int(p.rsplit("_s", 1)[1].split(".")[0])
        if s not in seeds:
            continue
        d = pd.read_csv(p)
        out.append((s, (d["pred"].values == d["gold"].values)))
    out.sort()
    return out


def boot_ci(correct_a, correct_b, mask, nboot=NBOOT):
    """配对 bootstrap：重采样测试句，返回差值的 95% 区间。"""
    idx = np.where(mask)[0]
    a = correct_a[:, idx].mean(axis=0)      # 每句在 seed 上的平均正确率
    b = correct_b[:, idx].mean(axis=0)
    d = (b - a) * 100.0
    n = len(idx)
    bs = np.empty(nboot)
    for i in range(nboot):
        s = RNG.integers(0, n, n)
        bs[i] = d[s].mean()
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


MODELS = [("TextCNN", False), ("BiLSTM", True)]
print("=" * 108)
print("固定评测子集上的 2x2，含**夸大量本身**的 95%% bootstrap 区间（重采样测试句, %d 次）" % NBOOT)
print("=" * 108)

rows = []
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    ltr, lte = rd(loc, MASSIVE)
    mtr, mte = rd(loc, MTDIR)
    n = len(lte)
    keep = np.array([(lte[i] not in set(ltr)) and (mte[i] not in set(mtr)) for i in range(n)])
    print()
    print("─" * 108)
    print("%s   固定子集 n = %d / %d" % (nm, keep.sum(), n))
    print("─" * 108)
    for model, masked in MODELS:
        # 读四格（BiLSTM 需要掩码版，pred 文件名里带 _mask）
        tag = model
        try:
            L = load_preds(lang, "localized", tag, {42, 43, 44, 45, 46})
            M = load_preds(lang, "mt-test", tag, {42, 43, 44, 45, 46})
            A = load_preds(lang, "mt", tag, {42, 43, 44, 45, 46})
            T = load_preds(lang, "mt-train", tag, {42, 43, 44, 45, 46})
        except Exception as e:
            print("   %s 读取失败: %s" % (model, e))
            continue
        if not (L and M and A and T):
            print("   %-10s 缺 pred（%d/%d/%d/%d）" % (model, len(L), len(M), len(A), len(T)))
            continue
        m = min([x[1].shape[0] for x in L + M + A + T] + [len(keep)])
        ok = keep[:m]
        cL = np.array([x[1][:m] for x in L])
        cM = np.array([x[1][:m] for x in M])
        cA = np.array([x[1][:m] for x in A])
        cT = np.array([x[1][:m] for x in T])
        seeds = sorted({x[0] for x in L} & {x[0] for x in M} & {x[0] for x in A} & {x[0] for x in T})

        def acc(c):
            return 100.0 * c[:, ok].mean()

        print()
        print("   %s (%s)  seeds %s n=%d" % (model, "masked" if masked else "-", seeds, ok.sum()))
        print("      L->L %6.2f   L->MT %6.2f   MT->MT %6.2f   MT->L %6.2f"
              % (acc(cL), acc(cM), acc(cA), acc(cT)))
        for name, a, b in [("L->MT  vs L->L (评测来源代价)", cL, cM),
                           ("MT->MT vs MT->L (**夸大量**)", cT, cA),
                           ("MT->L  vs L->L (训练来源代价)", cL, cT)]:
            d, lo, hi = boot_ci(a, b, ok)
            print("      %-32s %+7.2f  [%+7.2f, %+7.2f]" % (name, d, lo, hi))
            rows.append(dict(lang=lang, model=model, masked=masked, contrast=name,
                             delta=round(d, 2), lo=round(lo, 2), hi=round(hi, 2),
                             n_sent=int(ok.sum()), seeds=",".join(map(str, seeds))))

pd.DataFrame(rows).to_csv(os.path.join(SP, "table_inflation_ci.csv"), index=False, encoding="utf-8")
print()
print("已写出 table_inflation_ci.csv（%d 行）" % len(rows))

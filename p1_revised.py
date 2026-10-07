# -*- coding: utf-8 -*-
"""C1–C7：P1 分析重做。

要点（审稿意见）：
  * Table 11 的"81–95% 残留"是分层造成的，不是稳健结论 —— 样本 1/3 是按结果挑出的 flip
  * 补救：只用 random-100 算 A 类 vs 全体，给区间；或按总体比例加权
  * 主结论应改为：flip 的错误率是随机句的 1.35–1.52 倍
  * kappa 用 150 条（去重复项）重算
  * 删掉"意图变化约两倍"（韩语评判者2 方向相反）
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
RNG = np.random.default_rng(7)


def wilson(k, n):
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labs)}
    n = len(a)
    M = np.zeros((len(labs), len(labs)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / (n * n)
    if pe >= 1:
        return float("nan"), float("nan"), float("nan"), po
    k = (po - pe) / (1 - pe)
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return k, k - 1.96 * se, k + 1.96 * se, po


def test_order(loc):
    out = []
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "test":
            out.append(str(r["id"]))
    return out


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


print("=" * 100)
print("C1–C7  P1 分析重做")
print("=" * 100)

result = {}
for lang, loc, nm, enc in [("ko", "ko-KR", "Korean", "cp949"), ("zh", "zh-CN", "Chinese", "gbk")]:
    order = test_order(loc)
    idx = {i: k for k, i in enumerate(order)}
    L, M = get_preds(lang, "localized"), get_preds(lang, "mt-test")
    seeds = sorted(set(L) & set(M))
    OL = np.array([L[s] for s in seeds])   # shape (n_seeds, n_test)
    OM = np.array([M[s] for s in seeds])

    d = pd.read_csv(os.path.join(P1, "P1_%s_blind.csv" % lang), encoding=enc)
    key = pd.read_csv(os.path.join(P1, "P1_%s_key_PRIVATE.csv" % lang), encoding="utf-8")
    d["j1"] = d["judge1_category"].astype(str).str.strip().str.upper()
    d["j2"] = d["judge2_category"].astype(str).str.strip().str.upper()
    d = d.merge(key[["item_id", "group", "src_id"]], on="item_id", how="left")
    d["row"] = d["src_id"].astype(str).map(idx)
    d = d[d["row"].notna()].copy()
    d["row"] = d["row"].astype(int)
    nr = d[~d["group"].str.endswith("_repeat")].copy()

    print()
    print("═" * 100)
    print("%s" % nm)
    print("═" * 100)

    # ---------- C7: kappa 用 150 条 ----------
    print()
    print("【C7】Cohen's kappa —— 只用 150 条非重复项")
    for tag, sub in [("全部 162 条", d), ("150 条（去重复）", nr)]:
        for who in [("j1", "j2")]:
            k, lo, hi, po = kappa(list(sub["j1"]), list(sub["j2"]))
            print("   %-18s kappa %.3f [%.3f, %.3f]  观察一致 %d/%d = %.1f%%"
                  % (tag, k, lo, hi, int(round(po * len(sub))), len(sub), 100 * po))

    # ---------- C1: 只用 random-100 ----------
    print()
    print("【C1】只用 random-100 —— 全体 vs A 类")
    rand = nr[nr["group"] == "random"].copy()
    rr = rand["row"].values
    for who in ["j1", "j2"]:
        aL = OL[:, rr].mean() * 100
        aM = OM[:, rr].mean() * 100
        subA = rand[rand[who] == "A"]
        ra = subA["row"].values
        bL = OL[:, ra].mean() * 100
        bM = OM[:, ra].mean() * 100
        print("   %s  n=%d  L→L %.2f  L→MT %.2f  Δ %+.2f" % (who, len(rand), aL, aM, aM - aL))
        print("        只用 A 类 n=%d  L→L %.2f  L→MT %.2f  Δ %+.2f   -> 残留 %.0f%%"
              % (len(subA), bL, bM, bM - bL, 100 * (bM - bL) / (aM - aL) if aM != aL else float("nan")))

    # ---------- C1b: 按总体比例加权 ----------
    # flip 在固定子集里的真实占比：由预测直接算
    ltr = set()
    mtr = set()
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            ltr.add(r["utt"])
    for line in open(os.path.join(ROOT, "massive_mt", loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            mtr.add(r["utt"])
    lte = [json.loads(l)["utt"] for l in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8") if l.strip()]
    # 固定子集掩码（与前面分析一致）
    import itertools
    lte_list, mte_list = [], []
    for line in open(os.path.join(MASSIVE, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if line and json.loads(line)["partition"] == "test":
            lte_list.append(json.loads(line)["utt"])
    for line in open(os.path.join(ROOT, "massive_mt", loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if line and json.loads(line)["partition"] == "test":
            mte_list.append(json.loads(line)["utt"])
    n = min(len(lte_list), len(mte_list))
    keep = np.array([(lte_list[i] not in ltr) and (mte_list[i] not in mtr) for i in range(n)])
    okL = OL[:, :n].mean(axis=0) > 0.5
    okM = OM[:, :n].mean(axis=0) > 0.5
    flip_mask = keep & okL & ~okM
    p_flip = flip_mask.sum() / keep.sum()
    print()
    print("【C1b】分层加权  固定子集里 flip 的真实占比 p = %.4f  (%d / %d)"
          % (p_flip, flip_mask.sum(), keep.sum()))
    for who in ["j1", "j2"]:
        # 分层估计：A 类比例在两类里的加权
        parts = []
        for gname, gsub in [("flip", nr[nr["group"] == "flip"]), ("random", nr[nr["group"] == "random"])]:
            w = p_flip if gname == "flip" else (1 - p_flip)
            pc = (gsub[who].isin(["B", "C"])).mean()
            parts.append(w * pc)
        print("   %s  加权后 B+C 比例 = %.1f%%" % (who, 100 * sum(parts)))

    # ---------- C4/C5: flip 富集 ----------
    print()
    print("【C4】flip 相对随机的富集（按评判者，含 Wilson 区间）")
    for who in ["j1", "j2"]:
        for gname in ["random", "flip"]:
            sub = nr[nr["group"] == gname]
            bc = int(sub[who].isin(["B", "C"]).sum())
            lo, hi = wilson(bc, len(sub))
            print("   %s %-7s n=%3d  B+C %5.1f%% [%.1f, %.1f]"
                  % (who, gname, len(sub), 100 * bc / len(sub), lo, hi))
        r = nr[nr["group"] == "random"]; f = nr[nr["group"] == "flip"]
        ratio = (f[who].isin(["B", "C"]).mean()) / (r[who].isin(["B", "C"]).mean())
        print("        -> 富集倍数 %.2f" % ratio)

    result[lang] = dict(p_flip=round(float(p_flip), 4))

pd.DataFrame([result]).to_csv(os.path.join(SP, "table_p1_revised.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_p1_revised.csv")

# -*- coding: utf-8 -*-
"""导师 2.2 之 1/2：① 相似度分箱分析 ② 近重复敏感性（多阈值）。"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MT = os.path.join(ROOT, "massive_mt")
SEEDS = [42, 43, 44, 45, 46]
NB = 2000
RNG = np.random.default_rng(2026)


def load(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            tr.append(r)
        elif r["partition"] == "test":
            te.append(r)
    return tr, te


def preds(lang, variant, model="TextCNN"):
    out = {}
    mask = "_mask" if model == "BiLSTM" else ""
    for d in ["runs_e4_char", "runs_ws_extra", "runs_c2"]:
        for p in sorted(glob.glob(os.path.join(
                SP, d, "pred_massive_%s_in-language_%s_%s_char%s_s*.csv" % (variant, lang, model, mask)))):
            s = int(p.rsplit("_s", 1)[1].split(".")[0])
            if s in out or s not in SEEDS:
                continue
            df = pd.read_csv(p)
            out[s] = (df["pred"].values == df["gold"].values).astype(float)
    return out


def maxsim(A, B, step=512):
    """A 每行到 B 的最大余弦。"""
    m = np.empty(A.shape[0])
    for i in range(0, A.shape[0], step):
        m[i:i + step] = cosine_similarity(A[i:i + step], B).max(1)
    return m


REPORT = {}
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    print()
    print("=" * 108)
    print("  %s" % nm)
    print("=" * 108)
    Ltr, Lte = load(loc, MASSIVE)
    Mtr, Mte = load(loc, MT)
    En = load("en-US", MASSIVE)     # 英文源句
    EnTr = [r["utt"] for r in En[0]]
    id2en = {str(r["id"]): r["utt"] for r in En[1]}

    lset = set(r["utt"] for r in Ltr); mset = set(r["utt"] for r in Mtr)
    n = min(len(Lte), len(Mte))
    idx = np.where([(Lte[i]["utt"] not in lset) and (Mte[i]["utt"] not in mset) for i in range(n)])[0]

    # ---- 四格预测（TextCNN 为主）----
    C = {c: preds(lang, v) for c, v in [("LL", "localized"), ("LMT", "mt-test"),
                                        ("MTMT", "mt"), ("MTL", "mt-train")]}
    seeds = sorted(set.intersection(*[set(C[c]) for c in C]))
    ok = all(len(C[c][seeds[0]]) > int(idx.max()) for c in C)

    # ---- 相似度 ----
    v = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
    vEn = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
    Aen = vEn.fit_transform(EnTr)
    v.fit([r["utt"] for r in Ltr] + [r["utt"] for r in Mtr])
    AL = v.transform([r["utt"] for r in Ltr]); AM = v.transform([r["utt"] for r in Mtr])
    en_te = vEn.transform([id2en[str(Lte[i]["id"])] for i in idx])
    l_te = v.transform([Lte[i]["utt"] for i in idx])
    m_te = v.transform([Mte[i]["utt"] for i in idx])
    sim_en = maxsim(en_te, Aen)          # 英文源句 vs 英文训练集
    sim_LL = maxsim(l_te, AL); sim_LMT = maxsim(m_te, AL)
    sim_MT = maxsim(m_te, AM); sim_MTL = maxsim(l_te, AM)

    print("  最近邻相似度（中位数）：英文源 %.3f | L试验→L训练 %.3f | L试验→MT训练 %.3f | MT试验→MT训练 %.3f | MT试验→L训练 %.3f"
          % (np.median(sim_en), np.median(sim_LL), np.median(sim_MTL), np.median(sim_MT), np.median(sim_LMT)))
    for tag, s, thr in [("L试验→L训练", sim_LL, .8), ("MT试验→MT训练", sim_MT, .8), ("L试验→MT训练", sim_MTL, .8)]:
        print("     余弦≥%.1f 的比例：%-16s %.1f%%" % (thr, tag, 100 * (s >= thr).mean()))

    # ---- 按英文源相似度四分位 ----
    print()
    print("  === 按英文源句相似度四分位（TextCNN，字符，固定子集）===")
    qs = np.quantile(sim_en, [0.25, 0.5, 0.75])
    print("     分位边界: %.3f / %.3f / %.3f" % tuple(qs))
    bins = np.digitize(sim_en, qs)
    rows = []
    for k in range(4):
        m = (bins == k)
        if m.sum() < 20:
            continue
        acc = {c: 100 * np.mean([C[c][s][idx[m]].mean() for s in seeds]) for c in C}
        cost = acc["LMT"] - acc["LL"]
        infl = acc["MTMT"] - acc["MTL"]
        rows.append((k, int(m.sum()), cost, infl))
        print("     Q%d n=%4d  成本 %+6.2f   夸大量 %+6.2f" % (k + 1, m.sum(), cost, infl))

    # ---- 近重复敏感性 ----
    print()
    print("  === 近重复敏感性：剔除与各格训练集余弦 ≥ t 的试验句后重算 ===")
    sens = []
    for t in [1.01, 0.95, 0.9, 0.85, 0.8]:
        keep = (sim_LL < t) & (sim_LMT < t) & (sim_MT < t) & (sim_MTL < t)
        if keep.sum() < 100:
            continue
        acc = {c: 100 * np.mean([C[c][s][idx[keep]].mean() for s in seeds]) for c in C}
        cost = acc["LMT"] - acc["LL"]; infl = acc["MTMT"] - acc["MTL"]
        sens.append((t, int(keep.sum()), cost, infl))
        print("     t=%.2f  保留 n=%4d (%.1f%%)   成本 %+6.2f   夸大量 %+6.2f"
              % (t, keep.sum(), 100 * keep.mean(), cost, infl))
    REPORT[lang] = dict(bins=[[int(r[1]), round(r[2], 2), round(r[3], 2)] for r in rows],
                        sens=[[float(r[0]), int(r[1]), round(r[2], 2), round(r[3], 2)] for r in sens])

json.dump(REPORT, open(os.path.join(SP, "similarity_analysis.json"), "w"), indent=1, ensure_ascii=False)
print()
print("  已写出 similarity_analysis.json")

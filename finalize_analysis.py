# -*- coding: utf-8 -*-
"""两项收尾分析：
  (1) 空格条件下的 E4a，两侧去重 —— 使伪影与对照同口径
  (2) OOV 用【模型实际词表】重算（脚本上限 most_common(10000-2)）
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
LOC = {"ko": "ko-KR", "zh": "zh-CN"}
MARK = {"TextCNN": "", "BiLSTM": "_mask"}


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
    a, b = rd(loc, MASSIVE)
    c, d = rd(loc, MTDIR)
    D[l] = {"ltr": set(a), "lte": b, "mtr": set(c), "mte": d}

print("=" * 104)
print("(1) E4a 在【同一口径】下的对比：空格 vs 字符，均为两侧去重")
print("=" * 104)
SRC = {
    "localized": ["runs_pred_loc", "runs_pred_loc_mask", "runs_e4_char"],
    "mt-test":   ["runs_pred_mt", "runs_pred_mt_mask", "runs_e4_char"],
}


def find_pred(lang, model, tok, variant):
    mk = MARK[model] if model == "BiLSTM" else ""
    out = []
    for d in SRC[variant]:
        pat = os.path.join(SP, d, "pred_massive_%s_in-language_%s_%s_%s%s_s*.csv"
                           % (variant, lang, model, tok, mk))
        out += sorted(glob.glob(pat))
    # 去重：同一 run_id 只留一个（优先 runs_e4_char 的干净重跑）
    seen = {}
    for p in out:
        rid = os.path.basename(p)[5:-4]
        seen.setdefault(rid, p)
        if "runs_e4_char" in p:
            seen[rid] = p
    return [seen[k] for k in sorted(seen)]


rows = []
for lang in ["ko", "zh"]:
    for tok in ["whitespace", "char"]:
        for model in ["TextCNN", "BiLSTM"]:
            lp = find_pred(lang, model, tok, "localized")
            mp = find_pred(lang, model, tok, "mt-test")
            if len(lp) != 3 or len(mp) != 3:
                print("  %-3s %-11s %-9s pred 不足: loc %d / mt %d"
                      % (lang, tok, model, len(lp), len(mp)))
                continue
            lte, mte = D[lang]["lte"], D[lang]["mte"]
            Ltr, Mtr = D[lang]["ltr"], D[lang]["mtr"]
            dl = np.array([t in Ltr for t in lte])
            dm = np.array([t in Mtr for t in mte])
            A_all, A_cln, B_all, B_cln = [], [], [], []
            for a, b in zip(lp, mp):
                da, db = pd.read_csv(a), pd.read_csv(b)
                oa = (da["pred"].values == da["gold"].values)
                ob = (db["pred"].values == db["gold"].values)
                A_all.append(oa.mean()); A_cln.append(oa[~dl].mean())
                B_all.append(ob.mean()); B_cln.append(ob[~dm].mean())
            rows.append(dict(lang=lang, tok=tok, model=model,
                             loc_all=np.mean(A_all) * 100, mt_all=np.mean(B_all) * 100,
                             loc_cln=np.mean(A_cln) * 100, mt_cln=np.mean(B_cln) * 100,
                             d_all=(np.mean(B_all) - np.mean(A_all)) * 100,
                             d_cln=(np.mean(B_cln) - np.mean(A_cln)) * 100))
T = pd.DataFrame(rows)
print("%-4s %-11s %-9s %10s %10s %10s %10s %9s %9s" % (
    "语言", "分词", "模型", "本地化全量", "机翻全量", "Δ全量", "Δ去重", "本地化去重", "机翻去重"))
for _, r in T.iterrows():
    print("%-4s %-11s %-9s %9.2f%% %9.2f%% %+9.2f %+9.2f %9.2f%% %9.2f%%" % (
        r.lang, r.tok, r.model, r.loc_all, r.mt_all, r.d_all, r.d_cln, r.loc_cln, r.mt_cln))
print()
print("  【同口径伪影量级】去重口径下，空格比字符【夸大】了多少：")
for lang in ["ko", "zh"]:
    s = T[(T.lang == lang)]
    for model in ["TextCNN", "BiLSTM"]:
        w = s[(s.tok == "whitespace") & (s.model == model)]
        c = s[(s.tok == "char") & (s.model == model)]
        if len(w) and len(c):
            wd, cd = float(w.d_cln.iloc[0]), float(c.d_cln.iloc[0])
            print("    %-3s %-9s  空格 %.2f  ->  字符 %.2f   差 %.2f 点（占空格的 %.1f%%）"
                  % (lang, model, wd, cd, wd - cd, 100.0 * (wd - cd) / abs(wd)))
print()

# ---------------------------------------------------------------- (2) OOV
print("=" * 104)
print("(2) OOV 用【模型实际词表】重算（脚本：most_common(10000-2) + PAD/UNK）")
print("=" * 104)


def tk(text, kind):
    return list(text) if kind == "char" else text.split()


def oov(train, test, kind, cap=None):
    c = Counter()
    for t in train:
        c.update(w.lower() for w in tk(t, kind))
    if cap:
        vocab = {w for w, _ in c.most_common(cap)}
    else:
        vocab = set(c)
    nt = ov = nu = fu = 0
    for t in test:
        ws = tk(t, kind)
        if not ws:
            continue
        nu += 1
        hit = 0
        for w in ws:
            nt += 1
            if w.lower() in vocab:
                hit += 1
            else:
                ov += 1
        if hit == 0:
            fu += 1
    return len(vocab), 100.0 * ov / max(1, nt), 100.0 * fu / max(1, nu)


print("%-4s %-11s %-13s %-11s %11s %11s %11s %11s" % (
    "语言", "训练词表", "测试集", "分词", "完整词表", "OOV(完整)", "实际词表", "OOV(实际)"))
for lang in ["ko", "zh"]:
    for trname, trk in [("本地化", "ltr"), ("机翻", "mtr")]:
        for tename, tek in [("本地化测试", "lte"), ("机翻测试", "mte")]:
            if trname == "机翻" and tename == "本地化测试":
                pass
            for kind in ["whitespace", "char"]:
                v1, o1, f1 = oov(D[lang][trk], D[lang][tek], kind, cap=None)
                v2, o2, f2 = oov(D[lang][trk], D[lang][tek], kind, cap=9998)
                flag = "  <== 截断生效" if v2 < v1 else ""
                print("%-4s %-11s %-13s %-11s %11d %10.2f%% %11d %10.2f%%%s" % (
                    lang, trname, tename, kind, v1, o1, v2, o2, flag))
    print()

T.to_csv(os.path.join(SP, "table_e4a_same_convention.csv"), index=False, encoding="utf-8")
print("已写出 table_e4a_same_convention.csv")

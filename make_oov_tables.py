# -*- coding: utf-8 -*-
"""构建 OOV 率、跨划分重复句率、去重后准确率三张表。

映射依据（experiment_v2.py load_massive）：
  Xte 按 jsonl 中 partition=='test' 的出现顺序排列，
  pred_*.csv 的 idx 即为该顺序中的下标。
"""
import glob
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
LOC = {"en": "en-US", "ko": "ko-KR", "zh": "zh-CN"}


def read_split(loc, root):
    """返回 (train_list, test_list)，均按文件顺序。"""
    tr, te = [], []
    with open(os.path.join(root, loc + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["partition"] == "train":
                tr.append(r["utt"])
            elif r["partition"] == "test":
                te.append(r["utt"])
    return tr, te


def toks(text, kind):
    if kind == "char":
        return list(text)
    if kind == "whitespace":
        return text.split()
    raise ValueError(kind)


def oov_stats(train, test, kind):
    V = Counter()
    for t in train:
        V.update(w.lower() for w in toks(t, kind))
    ntok = oov = nutt = full = 0
    for t in test:
        ws = toks(t, kind)
        if not ws:
            continue
        nutt += 1
        hit = 0
        for w in ws:
            ntok += 1
            if w.lower() in V:
                hit += 1
            else:
                oov += 1
        if hit == 0:
            full += 1
    return {"token_oov_pct": 100.0 * oov / max(1, ntok),
            "utt_full_oov_pct": 100.0 * full / max(1, nutt),
            "vocab": len(V), "n_test": nutt}


# ---------------------------------------------------------------- 数据准备
L = {}
for lang, loc in LOC.items():
    ltr, lte = read_split(loc, MASSIVE)
    L[lang] = {"loc_train": ltr, "loc_test": lte}
    mp = os.path.join(MTDIR, loc + ".jsonl")
    if os.path.exists(mp):
        mtr, mte = read_split(loc, MTDIR)
        L[lang]["mt_train"] = mtr
        L[lang]["mt_test"] = mte
        L[lang]["mt_ids_match"] = (len(mte) == len(lte))

print("=" * 104)
print("表 A. 测试集相对训练词表的 OOV 率（按评测条件与分词方案）")
print("=" * 104)
print("%-4s %-16s %-11s %-11s %10s %10s %11s %8s" % (
    "语言", "训练词表来源", "测试集", "分词", "词表规模", "token OOV", "整句全OOV", "n"))
rowsA = []
for lang in ["en", "ko", "zh"]:
    for cond, trk, tek, tname in [
            ("localized", "loc_train", "loc_test", "本地化测试"),
            ("mt-test",   "loc_train", "mt_test",  "机翻测试"),
            ("mt",        "mt_train",  "mt_test",  "机翻测试")]:
        if tek not in L[lang] or trk not in L[lang]:
            continue
        if cond == "mt" and lang == "en":
            continue
        if cond == "mt-test" and lang == "en":
            continue
        for kind in ["whitespace", "char"]:
            s = oov_stats(L[lang][trk], L[lang][tek], kind)
            tag = {"loc_train": "本地化训练", "mt_train": "机翻训练"}[trk]
            print("%-4s %-16s %-11s %10s %10d %9.2f%% %9.2f%% %8d" % (
                lang, tag, tname, kind, s["vocab"], s["token_oov_pct"],
                s["utt_full_oov_pct"], s["n_test"]))
            rowsA.append(dict(lang=lang, condition=cond, train_vocab_from=trk,
                              test_set=tname, tokenizer=kind, **s))

print()
print("=" * 104)
print("表 B. 训练集与测试集的完全重复句（去重后）")
print("=" * 104)
print("%-4s %-8s %-8s %-10s %-10s %10s %10s" % (
    "语言", "训练唯一", "测试唯一", "重复句数", "占测试%", "去重后测试", "去重后占比"))
rowsB = []
for lang in ["en", "ko", "zh"]:
    ltr, lte = L[lang]["loc_train"], L[lang]["loc_test"]
    str_, ste = set(ltr), set(lte)
    inter = str_ & ste
    print("%-4s %-8d %-8d %-10d %9.2f%% %10d %10s" % (
        lang, len(str_), len(ste), len(inter), 100.0 * len(inter) / len(ste),
        len(ste) - len(inter), ""))
    rowsB.append(dict(lang=lang, n_train_uniq=len(str_), n_test_uniq=len(ste),
                      n_dup=len(inter), dup_pct=100.0 * len(inter) / len(ste)))
    if "mt_test" in L[lang]:
        mte = L[lang]["mt_test"]
        mset = set(mte)
        mi = set(L[lang]["mt_train"]) & mset
        print("     %-50s 机翻: 重复 %d 句 (%.2f%%)" % ("", len(mi), 100.0 * len(mi) / len(mset)))

# ---------------------------------------------------------------- 去重后准确率
print()
print("=" * 104)
print("表 C. 去重后的准确率（本地化条件；重复句 = 与训练集完全相同的测试句）")
print("=" * 104)
preds = glob.glob(os.path.join(SP, "runs_pred_loc", "pred_massive_localized_*.csv")) + \
        glob.glob(os.path.join(SP, "runs_e4_char", "pred_massive_localized_*.csv"))
print("  可用 pred 文件: %d" % len(preds))
print("%-4s %-8s %-11s %6s %10s %10s %10s" % (
    "语言", "模型", "分词", "seed", "全测试集", "去重后", "重复句上"))
rowsC = []
for p in sorted(preds):
    base = os.path.basename(p)[5:-4]          # massive_localized_in-language_ko_TextCNN_whitespace_s42
    f = base.split("_")
    # f = [dataset, variant, mode, lang, model, tokenizer, sNN]
    try:
        lang = f[3]; model = f[4]; tok = f[5]; seed = f[6][1:]
    except Exception:
        print("  !! 文件名无法解析: " + base)
        continue
    if lang not in LOC:
        continue
    df = pd.read_csv(p)
    lte = L[lang]["loc_test"]
    trset = set(L[lang]["loc_train"])
    if len(df) != len(lte):
        print("  !! 长度不符 %s: pred %d vs test %d" % (base, len(df), len(lte)))
        continue
    ok = (df["pred"].values == df["gold"].values)
    dup = [(i < len(lte) and lte[i] in trset) for i in range(len(lte))]
    dup = pd.Series(dup).values
    a_all = ok.mean() * 100
    a_non = ok[~dup].mean() * 100 if (~dup).sum() else float("nan")
    a_dup = ok[dup].mean() * 100 if dup.sum() else float("nan")
    print("%-4s %-8s %-11s %6s %9.2f%% %9.2f%% %9s   (重复 %d 句)" % (
        lang, model, tok, seed, a_all, a_non,
        ("%.2f%%" % a_dup) if dup.sum() else "—", int(dup.sum())))
    rowsC.append(dict(lang=lang, model=model, tokenizer=tok, seed=seed,
                      acc_all=a_all, acc_nodup=a_non, acc_dup=a_dup, n_dup=int(dup.sum())))

if rowsC:
    C = pd.DataFrame(rowsC)
    print()
    print("  按 语言×模型×分词 汇总（3 seed 均值）:")
    g = C.groupby(["lang", "model", "tokenizer"]).agg(
        acc_all=("acc_all", "mean"), acc_nodup=("acc_nodup", "mean"),
        acc_dup=("acc_dup", "mean"), n_dup=("n_dup", "first")).round(2)
    g["差(全-去重)"] = (g.acc_all - g.acc_nodup).round(2)
    print(g.to_string())

pd.DataFrame(rowsA).to_csv(os.path.join(SP, "table_oov.csv"), index=False, encoding="utf-8")
pd.DataFrame(rowsB).to_csv(os.path.join(SP, "table_duplicates.csv"), index=False, encoding="utf-8")
pd.DataFrame(rowsC).to_csv(os.path.join(SP, "table_dedup_accuracy.csv"), index=False, encoding="utf-8")
print()
print("已写出: table_oov.csv / table_duplicates.csv / table_dedup_accuracy.csv")

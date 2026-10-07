# -*- coding: utf-8 -*-
"""任务4 全套结果：字符级 E4 对照 + 掩码 morph 判定 + 去重后准确率。"""
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
LOC = {"en": "en-US", "ko": "ko-KR", "zh": "zh-CN"}

fr = []
for p in glob.glob(os.path.join(SP, "runs_*", "results_v2.csv")):
    x = pd.read_csv(p)
    x["__src"] = os.path.basename(os.path.dirname(p))
    fr.append(x)
R = pd.concat(fr, ignore_index=True)
R["bilstm_mask"] = R["bilstm_mask"].fillna(False).astype(bool)
R = R[(R.dataset == "massive") & (R["mode"] == "in-language")]


def M(lang, tok, model, mask, variant="localized"):
    s = R[(R.test_lang == lang) & (R.tokenizer == tok) & (R.model == model)
          & (R.bilstm_mask == mask) & (R.data_variant == variant)]
    if not len(s):
        return None
    return s.accuracy.mean() * 100, s.accuracy.std() * 100, len(s)


def f(t):
    return "—" if t is None else "%6.2f±%-5.2f" % (t[0], t[1])


print("=" * 104)
print("【判定 M3】分词梯度：掩码 vs 未掩码（本地化条件，TextCNN 不受影响，BiLSTM 是关键）")
print("=" * 104)
print("%-4s %-9s %-11s %18s %18s" % ("语言", "模型", "读出端", "whitespace", "morph"))
for lang in ["ko", "zh"]:
    print("%-4s %-9s %-11s %18s %18s" % (
        lang, "BiLSTM", "no mask", f(M(lang, "whitespace", "BiLSTM", False)),
        f(M(lang, "morph", "BiLSTM", False))))
    print("%-4s %-9s %-11s %18s %18s" % (
        lang, "BiLSTM", "+mask", f(M(lang, "whitespace", "BiLSTM", True)),
        f(M(lang, "morph", "BiLSTM", True))))
print()

print("=" * 104)
print("【判定 M1】字符级 E4 对照（OOV≈0）：韩语下降是词表问题还是分布差异？")
print("=" * 104)
print("%-4s %-6s %-9s %-8s %20s %20s %20s %10s" % (
    "语言", "分词", "模型", "读出端", "本地化→本地化", "本地化→机翻", "机翻→机翻", "空格Δ"))
for lang in ["ko", "zh"]:
    for tok in ["whitespace", "char"]:
        for model, mask in [("TextCNN", False), ("BiLSTM", True)]:
            a = M(lang, tok, model, mask, "localized")
            b = M(lang, tok, model, mask, "mt-test")
            c = M(lang, tok, model, mask, "mt")
            d = (b[0] - a[0]) if (a and b) else None
            print("%-4s %-6s %-9s %-8s %20s %20s %20s %10s" % (
                lang, tok, model, "+mask" if mask else "-",
                f(a), f(b), f(c), ("%+.2f" % d) if d is not None else "—"))
    print()

print("=" * 104)
print("【汇总】E4a（本地化→机翻）与 E4b（机翻→机翻）的效应量，按分词")
print("=" * 104)
print("%-4s %-11s %-9s %12s %12s %12s" % ("语言", "分词", "模型", "本地化", "E4a机翻", "E4b机翻"))
for lang in ["ko", "zh"]:
    for tok in ["whitespace", "char"]:
        for model, mask in [("TextCNN", False), ("BiLSTM", True)]:
            a = M(lang, tok, model, mask, "localized")
            b = M(lang, tok, model, mask, "mt-test")
            c = M(lang, tok, model, mask, "mt")
            if not a:
                continue
            da = ("%+.2f" % (b[0] - a[0])) if b else "—"
            db = ("%+.2f" % (c[0] - a[0])) if c else "—"
            print("%-4s %-11s %-9s %12.2f %12s %12s" % (lang, tok, model, a[0], da, db))
    print()

# ---------------- 去重后的字符级准确率 ----------------
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


print("=" * 104)
print("【去重】字符级条件下，扣除与训练集完全相同的测试句后的准确率")
print("=" * 104)
L = {}
for l, loc in LOC.items():
    L[l] = {"tr": rd(loc, MASSIVE)[0], "te": rd(loc, MASSIVE)[1]}
D = {}
for l, loc in LOC.items():
    p = os.path.join(MTDIR, loc + ".jsonl")
    if os.path.exists(p):
        D[l] = rd(loc, MTDIR)

print("%-4s %-11s %-9s %-16s %10s %10s %10s %8s" % (
    "语言", "分词", "模型", "条件", "全量", "去重后", "重复句", "n_dup"))
for lang in ["ko", "zh"]:
    trset = set(L[lang]["tr"])
    for tok, tag in [("whitespace", "本地化"), ("char", "本地化")]:
        for model, mask in [("TextCNN", False), ("BiLSTM", True)]:
            pats = glob.glob(os.path.join(
                SP, "runs_pred_loc" if tok == "whitespace" else "runs_e4_char",
                "pred_massive_localized_in-language_%s_%s_%s%s_s*.csv" % (
                    lang, model, tok, "_mask" if mask else "")))
            if not pats:
                continue
            all_ok = non_ok = dup_ok = n_all = n_non = n_dup = 0
            for p in pats:
                df = pd.read_csv(p)
                te = L[lang]["te"]
                if len(df) != len(te):
                    continue
                ok = (df["pred"].values == df["gold"].values)
                dup = pd.Series([t in trset for t in te]).values
                all_ok += ok.sum(); non_ok += ok[~dup].sum(); dup_ok += ok[dup].sum()
                n_all += len(ok); n_non += (~dup).sum(); n_dup += dup.sum()
            if n_all:
                print("%-4s %-11s %-9s %-16s %9.2f%% %9.2f%% %9.2f%% %8d" % (
                    lang, tok, model, tag, 100.0 * all_ok / n_all,
                    100.0 * non_ok / max(1, n_non), 100.0 * dup_ok / max(1, n_dup), n_dup))
    print()

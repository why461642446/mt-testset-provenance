# -*- coding: utf-8 -*-
"""算出 P0 第 1、2、4、5、6 条所需的全部正确值。"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
from runs_loader import load_runs

SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
R, _ = load_runs()

print("=" * 100)
print("P0-1 你点名的每一处，给出正确值")
print("=" * 100)

D = R[(R["model"].isin(["TextCNN", "BiLSTM"])) & (R["mode"] == "in-language")]
D = D[(D["model"] != "BiLSTM") | (D.bilstm_mask == True)]

def acc(tok, var, lang, model):
    d = D[(D.tokenizer == tok) & (D.data_variant == var) & (D.test_lang == lang) & (D["model"] == model)]
    a = d.accuracy.dropna().values * 100
    return a.mean() if len(a) else float("nan")

# --- ① 空格 L->MT 代价范围（摘要里的 38.9-41.2）---
print("\n① 【空格】L→MT 代价（固定子集，需用 pred；这里先给全量）")
for lang in ["ko", "zh"]:
    for m in ["TextCNN", "BiLSTM"]:
        g = acc("whitespace", "mt-test", lang, m) - acc("whitespace", "localized", lang, m)
        print("     %-3s %-8s %+7.2f" % (lang, m, g))

# --- ② 词表贡献占比（43-45%）---
print("\n② 词表敏感性占空格估计的比例")
for lang in ["ko", "zh"]:
    for m in ["TextCNN", "BiLSTM"]:
        ws = acc("whitespace", "mt-test", lang, m) - acc("whitespace", "localized", lang, m)
        ch = acc("char", "mt-test", lang, m) - acc("char", "localized", lang, m)
        if ws != 0:
            print("     %-3s %-8s  空格 %+7.2f  字符 %+7.2f  词表贡献 %5.1f%%"
                  % (lang, m, ws, ch, 100.0 * (abs(ws) - abs(ch)) / abs(ws)))

# --- ③ 三语 BERT-TextCNN 差（4.1 的 8.33 / 7.89 / 5.75）---
print("\n③ 三语 BERT−TextCNN 差（全量，TextCNN 用各语言可用分词）")
BERT = R[(R["model"] == "BERT") & (R["mode"] == "in-language") & (R.data_variant == "localized")]
for lang, tok in [("en", "whitespace"), ("ko", "whitespace"), ("zh", "char")]:
    b = BERT[BERT.test_lang == lang].accuracy
    t = D[(D["model"] == "TextCNN") & (D.test_lang == lang) & (D.tokenizer == tok) &
          (D.data_variant == "localized")].accuracy
    if len(b) and len(t):
        print("     %-3s  BERT %6.2f − TextCNN(%s) %6.2f = %+6.2f"
              % (lang, b.mean() * 100, tok, t.mean() * 100, (b.mean() - t.mean()) * 100))

# --- ④ 三语跨度 ---
print("\n④ 三语跨度")
ws = [acc("whitespace", "localized", lg, "TextCNN") for lg in ["en", "ko", "zh"]]
ch = [acc("char", "localized", lg, "TextCNN") for lg in ["en", "ko", "zh"]]
print("     空格 %s  跨度 %.2f" % ([round(x, 2) for x in ws], max(ws) - min(ws)))
print("     字符 %s  跨度 %.2f" % ([round(x, 2) for x in ch], max(ch) - min(ch)))

# --- ⑤ §4.5 的架构对比 ---
print("\n⑤ §4.5 TextCNN vs BiLSTM（字符级，全量，掩码）")
for lang in ["en", "ko", "zh"]:
    t = D[(D["model"] == "TextCNN") & (D.test_lang == lang) & (D.tokenizer == "char") &
          (D.data_variant == "localized")]
    b = D[(D["model"] == "BiLSTM") & (D.test_lang == lang) & (D.tokenizer == "char") &
          (D.data_variant == "localized")]
    if len(t) and len(b):
        print("     %-3s TextCNN %.2f±%.2f (n=%d)  BiLSTM %.2f±%.2f (n=%d)  差 %+.2f"
              % (lang, t.accuracy.mean() * 100, t.accuracy.std(ddof=1) * 100, len(t),
                 b.accuracy.mean() * 100, b.accuracy.std(ddof=1) * 100, len(b),
                 (t.accuracy.mean() - b.accuracy.mean()) * 100))

# --- ⑥ 掩码消融的 sd ---
print("\n⑥ 掩码消融（韩语 BiLSTM whitespace）：未掩码 vs 掩码")
for mask in [False, True]:
    d = D[(D["model"] == "BiLSTM") & (D.test_lang == "ko") & (D.tokenizer == "whitespace") &
          (D.bilstm_mask == mask) & (D.data_variant == "localized")]
    print("     mask=%-5s n=%d  %.2f ± %.2f  seeds=%s"
          % (str(mask), len(d), d.accuracy.mean() * 100, d.accuracy.std(ddof=1) * 100,
             sorted(set(d.seed))))

# --- ⑦ §4.3 morph 增益 ---
print("\n⑦ §4.3 morph 与 char 相对 whitespace 的增益（全量，掩码）")
for lang in ["ko", "zh"]:
    for m in ["TextCNN", "BiLSTM"]:
        w = acc("whitespace", "localized", lang, m)
        mo = acc("morph", "localized", lang, m)
        c = acc("char", "localized", lang, m)
        print("     %-3s %-8s  ws %.2f  morph %+.2f  char %+.2f" % (lang, m, w, mo - w, c - w))

# --- ⑧ 校验和：附录 A 与 3.6 的说法 ---
print()
print("=" * 100)
print("P0-6 校验和：实际记录了什么")
print("=" * 100)
for d in ["runs_e2b", "runs_eff", "runs_v4a", "runs_e4_char"]:
    p = os.path.join(SP, d, "manifest_v2.json")
    if os.path.exists(p):
        m = json.load(open(p, encoding="utf-8"))
        print("  %-14s 顶层键: %s" % (d, list(m.keys())))
rr = os.path.join(SP, "results_v2.csv")
if os.path.exists(rr):
    c = pd.read_csv(rr).columns
    print("\n  results_v2.csv 列: %s" % list(c))
    print("  含 checksum/md5 列: %s" % any("check" in x.lower() or "md5" in x.lower() for x in c))

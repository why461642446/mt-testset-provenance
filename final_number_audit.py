# -*- coding: utf-8 -*-
"""全文数字终审：把稿子里所有关键数字与数据文件逐个核对。"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\yanjiubaogaoxiangmu\sci_paper")
from runs_loader import load_runs

SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
R, _ = load_runs()
T = open(os.path.join(SP, "manuscript_draft.md"), encoding="utf-8").read()

D = R[(R["model"].isin(["TextCNN", "BiLSTM"])) & (R["mode"] == "in-language")]
Dm = D[(D["model"] != "BiLSTM") | (D["bilstm_mask"] == True)]


def cell(tok, var, lang, model, mask=None):
    d = Dm[(Dm.tokenizer == tok) & (Dm.data_variant == var) & (Dm.test_lang == lang) & (Dm["model"] == model)]
    a = d.accuracy.dropna().values * 100
    return (round(a.mean(), 2), round(a.std(ddof=1), 2) if len(a) > 1 else 0.0, len(a))


print("=" * 100)
print("全文数字终审")
print("=" * 100)
ok = bad = 0


def chk(label, expect_str, present=True):
    """expect_str 应出现在稿中；present=False 表示应不出现"""
    global ok, bad
    got = expect_str in T
    good = (got == present)
    if good:
        ok += 1
    else:
        bad += 1
        print("  !! %-52s '%s' %s" % (label, expect_str, "缺失" if present else "残留"))
    return good


# ---------- Table 6 全量值 ----------
print()
print("【Table 6】全量测试集，n=5")
for lang, m, tok, mask in [("en", "TextCNN", "whitespace", None), ("en", "BiLSTM", "whitespace", True),
                           ("en", "BiLSTM", "whitespace", False), ("en", "BiLSTM", "char", True),
                           ("en", "TextCNN", "char", None),
                           ("ko", "TextCNN", "whitespace", None), ("ko", "TextCNN", "morph", None),
                           ("ko", "TextCNN", "char", None), ("ko", "BiLSTM", "whitespace", True),
                           ("ko", "BiLSTM", "whitespace", False), ("ko", "BiLSTM", "morph", True),
                           ("ko", "BiLSTM", "char", True),
                           ("zh", "TextCNN", "whitespace", None), ("zh", "TextCNN", "morph", None),
                           ("zh", "TextCNN", "char", None), ("zh", "BiLSTM", "whitespace", True),
                           ("zh", "BiLSTM", "whitespace", False), ("zh", "BiLSTM", "morph", True),
                           ("zh", "BiLSTM", "char", True)]:
    mean, sd, n = cell(tok, "localized", lang, m, mask)
    tag = "%s %s %s%s" % (lang, m, tok, "" if mask is None else ("(mask)" if mask else "(unmask)"))
    chk(tag, "%.2f" % mean)
    print("  %-34s %.2f ± %.2f  (n=%d)" % (tag, mean, sd, n))

# ---------- 关键论断数字 ----------
print()
print("【关键数字】")
K = [
    ("v4 mBERT ko 代价", "−17.58"), ("v4 mBERT zh 代价", "−9.14"),
    ("v4 XLM-R ko 代价", "−11.29"), ("v4 XLM-R zh 代价", "−7.39"),
    ("v4 mBERT ko 夸大量", "+6.86"), ("v4 XLM-R zh 夸大量", "−0.20"),
    ("审计 en", "82.97"), ("审计 ko", "6.32"), ("审计 zh", "65.78"),
    ("审计 mBERT ko", "29.67"), ("审计 mBERT zh", "51.66"),
    ("MD5 en", "135c741954f86f23d37784dba247c78a"),
    ("MD5 ko", "18f2b556dde5b8ac77ace58d0f2a6c1d"),
    ("MD5 zh", "2d2a3fb725b3a37793e2dcd16442b1e0"),
    ("P1 kappa 韩", "0.646"), ("P1 kappa 中", "0.524"),
    ("P1 A 类控制 韩", "−36.97"), ("P1 A 类控制 中", "−30.29"),
    ("chrF 译文长度 韩", "+1.88"), ("chrF 长度匹配 韩", "−20.24"),
    ("chrF 长度匹配 中", "−11.68"),
    ("sd 缩减 29.1", "29.1"), ("sd 缩减 16.2", "16.2"),
    ("词表贡献", "42–45%"), ("预设检验", "2.1 to 4.9"),
    ("17.43", "17.43"), ("三语 BERT−TC ko", "7.90"), ("三语 BERT−TC zh", "5.49"),
    ("三语跨度 空格", "70.29"), ("三语跨度 字符", "4.38"),
    ("OOV ko 字符", "3.16%"), ("OOV zh 字符", "4.75%"),
    ("重复句 7.5%", "7.5%"), ("去重后 4.48%", "4.48%"),
    ("空格夸大量 韩", "27.57"), ("空格夸大量 韩 B", "27.04"),
]
for lab, s in K:
    chk(lab, s)

# ---------- 应已清除 ----------
print()
print("【应已清除】")
for lab, s in [("旧 sd 27.49", "27.49"), ("旧 sd 10.24", "10.24"), ("旧 sd 9.16", "9.16"),
               ("旧值 38.52", "38.52"), ("旧值 80.11(非Distil)", "80.11"), ("旧值 49.38(阶梯)",
                "49.38 → 57.80 → 67.61"), ("旧值 80.36", "80.36"), ("旧值 66.07", "66.07"),
               ("修订史 an earlier version", "an earlier version"),
               ("修订史 between drafts", "between drafts"),
               ("修订史 we first wrote", "we first wrote"),
               ("草稿路径 figures/", "figures/fig"),
               ("假校验和 dataset file MD5", "dataset file MD5"),
               ("sum of parts", "sum of parts"),
               ("Table3 旧题注", "Every row is a condition we specified")]:
    if s in ("80.11", "49.38 → 57.80 → 67.61"):
        print("  (跳过易误判项) %s" % lab)
        continue
    chk(lab, s, present=False)

print()
print("=" * 100)
print("  通过 %d 项，问题 %d 项" % (ok, bad))
print("  篇幅 %d 词 / %d 表" % (len(T.split()), sum(1 for l in T.split("\n") if re.match(r"^\|\s*---", l))))

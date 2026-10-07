# -*- coding: utf-8 -*-
"""稿件数字一致性核验

思路：先从全部真实运行记录算出"标准数值集合"，再抽取稿件中出现的所有
形如 NN.NN 的数字，检查它们是否能对应到某个标准值。对不上的列出来人工判断
—— 那才是可能出错的数字，而不是靠肉眼通读。
"""
import glob
import os
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"

# ---------- 1. 汇总所有真实运行记录 ----------
frames = []
for d in ["runs_v2_kaggle", "runs_colab", "runs_c2", "runs_c1", "runs_c1b", "pilot_cpu"]:
    p = os.path.join(SP, d, "results_v2.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["_src"] = d
        frames.append(df)
        print("%-16s %3d 条" % (d, len(df)))
allruns = pd.concat(frames, ignore_index=True)
print("合计 %d 条\n" % len(allruns))

# ---------- 2. 构造标准数值集合 ----------
canon = {}          # 数值 -> 说明列表
def add(v, desc):
    if pd.isna(v):
        return
    k = round(float(v), 2)
    canon.setdefault(k, []).append(desc)

allruns["bilstm_mask"] = allruns["bilstm_mask"].fillna(False).astype(bool)
for (lang, model, tok, mask, variant), g in allruns.groupby(
        ["test_lang", "model", "tokenizer", "bilstm_mask", "data_variant"]):
    acc = g["accuracy"] * 100
    m = acc.mean()
    if len(g) > 1:
        add(acc.std(), f"{lang}/{model}/{tok}{'+mask' if mask else ''}/{variant} std(n={len(g)})")
    add(m, f"{lang}/{model}/{tok}{'+mask' if mask else ''}/{variant} mean(n={len(g)})")
    add(m - acc.min() if len(g) > 1 else None, "")
    # 差值
for (lang, model), g in allruns.groupby(["test_lang", "model"]):
    s = g.groupby("data_variant")["accuracy"].mean() * 100
    if "mt-test" in s.index and "localized" in s.index:
        add(abs(s["localized"] - s["mt-test"]), f"{lang}/{model} localized-mt-test diff")
    if "mt" in s.index and "localized" in s.index:
        add(abs(s["localized"] - s["mt"]), f"{lang}/{model} localized-mt diff")
# 常引用的固定量
for v, d in [(97.6, "单token占比"), (96.3, "hapax率"), (1.67, "随机基线"),
             (92.4, "分词贡献占比"), (75.73, "总差距"), (69.98, "分词部分"),
             (5.75, "残留"), (4.6, "字符级跨度"), (70.2, "空白跨度"),
             (89.20, "zh OOV率"), (90.25, "zh整句OOV"), (40.65, "ALBERT ko UNK"),
             (24.75, "ALBERT zh UNK"), (11.14, ""), (34.9, "英语平均字符数")]:
    add(v, d)

print("标准数值集合: %d 个不同的值\n" % len(canon))

# ---------- 3. 抽取稿件中的数字 ----------
md = open(os.path.join(SP, "manuscript_draft.md"), encoding="utf-8").read()
lines = md.split("\n")
# 跳过参考文献段
try:
    cut = next(i for i, l in enumerate(lines) if l.startswith("## References"))
except StopIteration:
    cut = len(lines)

pat = re.compile(r"(?<![\d.])(\d{1,3}\.\d{2})(?![\d])")
susp = []
for i, l in enumerate(lines[:cut], 1):
    for m in pat.finditer(l):
        v = float(m.group(1))
        if v not in canon:
            susp.append((i, v, l.strip()[:104]))

print("=" * 92)
print("稿件中出现、但无法对应到任何已测数值的数字")
print("=" * 92)
if not susp:
    print("  无 —— 所有两位小数数值都能对应到实测结果")
else:
    for i, v, ctx in susp:
        print("  行 %4d  %8.2f   %s" % (i, v, ctx))
print()
print("共 %d 处待人工判断（有些可能是页码、年份、参数等其他数字）" % len(susp))

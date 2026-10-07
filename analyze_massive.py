# -*- coding: utf-8 -*-
"""
analyze_massive.py —— 剖析 MASSIVE 的 en-US / ko-KR / zh-CN 三个子集

产出（用于论文的数据集章节与实验设计）：
  1. 各语言 train/dev/test 规模
  2. 意图类别数、场景数
  3. 18 个 scenario 的分布（重点看 iot 智能家居）
  4. 空白分词下的 token 数分布（这是"中文崩溃"的直接量化证据）
  5. 字符数分布
  6. 三语是否严格平行（按 id 对齐）
  7. 词汇量 / 未登录词
  8. 样例
"""

import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\yanjiubaogaoxiangmu\massive\1.1\data"
LOCALES = ["en-US", "ko-KR", "zh-CN"]


def load(locale):
    recs = []
    with open(os.path.join(ROOT, locale + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


data = {loc: load(loc) for loc in LOCALES}

print("=" * 78)
print("1) 各语言划分规模")
print("=" * 78)
print("%-8s %-8s %8s %8s %8s" % ("locale", "file", "train", "dev", "test"))
for loc in LOCALES:
    parts = Counter(r["partition"] for r in data[loc])
    print("%-8s %-8s %8d %8d %8d   (合计 %d)"
          % (loc, loc + ".jsonl", parts.get("train", 0), parts.get("dev", 0),
             parts.get("test", 0), len(data[loc])))

print()
print("=" * 78)
print("2) 意图 / 场景 规模")
print("=" * 78)
for loc in LOCALES:
    intents = {r["intent"] for r in data[loc]}
    scen = {r["scenario"] for r in data[loc]}
    print("  %-8s 意图 %d 类, 场景 %d 类" % (loc, len(intents), len(scen)))

print()
print("=" * 78)
print("3) 18 个 scenario 分布（train+dev+test）")
print("=" * 78)
scen_tab = {}
for loc in LOCALES:
    scen_tab[loc] = Counter(r["scenario"] for r in data[loc])
allscen = sorted(set().union(*[set(t) for t in scen_tab.values()]))
print("%-18s %8s %8s %8s" % ("scenario", *LOCALES))
for s in allscen:
    mark = "  <== 智能家居" if s == "iot" else ""
    print("%-18s %8d %8d %8d%s"
          % (s, scen_tab["en-US"][s], scen_tab["ko-KR"][s], scen_tab["zh-CN"][s], mark))

print()
print("=" * 78)
print("4) 空白(whitespace)分词下的 token 数 —— 中文崩溃的直接证据")
print("=" * 78)
tok_stats = {}
for loc in LOCALES:
    toks = [len(r["utt"].split()) for r in data[loc]]
    chars = [len(r["utt"]) for r in data[loc]]
    tok_stats[loc] = toks
    print("  %-8s  token: 均值 %6.2f  中位 %5.1f  最大 %4d | "
          "字符: 均值 %6.2f" % (loc, st.mean(toks), st.median(toks), max(toks),
                                st.mean(chars)))
print()
print("  → 空白分词后 token 数为 1 的句子占比：")
for loc in LOCALES:
    n1 = sum(1 for t in tok_stats[loc] if t == 1)
    print("     %-8s %5d / %5d = %5.1f%%"
          % (loc, n1, len(tok_stats[loc]), 100.0 * n1 / len(tok_stats[loc])))

print()
print("=" * 78)
print("5) 三语是否严格平行（按 id 对齐）")
print("=" * 78)
ids = {loc: set(r["id"] for r in data[loc]) for loc in LOCALES}
base = ids["en-US"]
for loc in LOCALES:
    print("  %-8s id 数 %6d | 与 en-US 交集 %6d | 完全相同: %s"
          % (loc, len(ids[loc]), len(ids[loc] & base), ids[loc] == base))
# 对齐后检查同一 id 的 intent 是否一致
en_by_id = {r["id"]: r for r in data["en-US"]}
for loc in ["ko-KR", "zh-CN"]:
    d = {r["id"]: r for r in data[loc]}
    common = set(en_by_id) & set(d)
    same = sum(1 for i in common if en_by_id[i]["intent"] == d[i]["intent"])
    print("  %-8s 共有 id %d, 其中 intent 一致 %d (%.2f%%)"
          % (loc, len(common), same, 100.0 * same / max(1, len(common))))

print()
print("=" * 78)
print("6) 样例（iot 场景，每个语言 3 条）")
print("=" * 78)
for loc in LOCALES:
    print("  --- %s ---" % loc)
    n = 0
    for r in data[loc]:
        if r["scenario"] == "iot":
            print("     [%s] %s" % (r["intent"], r["utt"][:96]))
            n += 1
            if n >= 3:
                break

print()
print("=" * 78)
print("7) 词汇量（train 集，小写后按空白切分）")
print("=" * 78)
for loc in LOCALES:
    vocab = Counter()
    for r in data[loc]:
        if r["partition"] == "train":
            vocab.update(r["utt"].lower().split())
    print("  %-8s 词表大小 %6d | 出现 1 次的词 %6d (%.1f%%)"
          % (loc, len(vocab), sum(1 for v in vocab.values() if v == 1),
             100.0 * sum(1 for v in vocab.values() if v == 1) / len(vocab)))

print()
print("=" * 78)
print("8) 与现有 Snips 三语数据的可比性")
print("=" * 78)
print("  Snips(英语, 7 intents)   : train 11,587 / test 2,897  (合并后 80/20 重切)")
for loc in LOCALES:
    parts = Counter(r["partition"] for r in data[loc])
    print("  MASSIVE %-8s(60 intents): train %6d / dev %5d / test %5d"
          % (loc, parts["train"], parts["dev"], parts["test"]))

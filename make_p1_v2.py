# -*- coding: utf-8 -*-
"""P1 盲表（修订版）：把 50 句翻转句混入同一张随机表，再各自统计。

同时输出：
  - P1_<lang>_blind.csv   给评判者（不含分组信息、不含本地化参考、不含模型预测）
  - P1_<lang>_key.csv     PRIVATE，含分组与预测对错，用于统计
  - 中文空格固定子集上的基线（多数类占比、均匀随机）
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
OUT = os.path.join(SP, "p1_judging")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261004)
N_RAND, N_FLIP, N_REP = 100, 50, 12


def rd(loc, root):
    rows = []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append((r["id"], r["utt"], r["partition"], r["intent"]))
    return rows


en_test = {i: u for i, u, p, it in rd("en-US", MASSIVE) if p == "test"}

for lang, loc, name in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    lrows, mrows = rd(loc, MASSIVE), rd(loc, MTDIR)
    lte = [(i, u, it) for i, u, p, it in lrows if p == "test"]
    mte = {i: u for i, u, p, it in mrows if p == "test"}
    ltr = set(u for i, u, p, it in lrows if p == "train")
    mtr = set(u for i, u, p, it in mrows if p == "train")
    keep = [(i, u, it) for (i, u, it) in lte if u not in ltr and mte.get(i, "") not in mtr]
    idx_of = {i: k for k, (i, u, it) in enumerate(lte)}

    lp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_localized_in-language_%s_TextCNN_char_s*.csv" % lang)))
    mp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_mt-test_in-language_%s_TextCNN_char_s*.csv" % lang)))
    ok_l = (pd.read_csv(lp[0])["pred"].values == pd.read_csv(lp[0])["gold"].values)
    ok_m = (pd.read_csv(mp[0])["pred"].values == pd.read_csv(mp[0])["gold"].values)

    # 随机 100（按意图分层）
    by_int = {}
    for (i, u, it) in keep:
        by_int.setdefault(it, []).append((i, u, it))
    intents = sorted(by_int)
    tot = sum(len(by_int[k]) for k in intents)
    quota = {k: max(1, int(round(N_RAND * len(by_int[k]) / tot))) for k in intents}
    order = sorted(intents, key=lambda k: -len(by_int[k]))
    j = 0
    while sum(quota.values()) < N_RAND and j < 100000:
        k = order[j % len(order)]
        if quota[k] < len(by_int[k]):
            quota[k] += 1
        j += 1
    while sum(quota.values()) > N_RAND:
        k = max(quota, key=lambda x: quota[x])
        if quota[k] > 1:
            quota[k] -= 1
        else:
            break
    rand = []
    for k in intents:
        pool = by_int[k]
        sel = rng.choice(len(pool), size=min(quota[k], len(pool)), replace=False)
        rand += [pool[x] for x in sel]
    rand = rand[:N_RAND]

    # 翻转句
    flips = [(i, u, it) for (i, u, it) in keep
             if ok_l[idx_of[i]] and not ok_m[idx_of[i]]]
    flips = [flips[x] for x in rng.choice(len(flips), size=min(N_FLIP, len(flips)), replace=False)]

    items = []
    for n, (i, u, it) in enumerate(rand):
        items.append(dict(group="random", src_id=i, english_source=en_test.get(i, ""),
                          target_text=mte.get(i, ""), intent_label=it))
    for n, (i, u, it) in enumerate(flips):
        items.append(dict(group="flip", src_id=i, english_source=en_test.get(i, ""),
                          target_text=mte.get(i, ""), intent_label=it))
    # 重复项（从全部条目里抽，混入后以新 item_id 出现）
    reps = [dict(items[x]) for x in rng.choice(len(items), size=min(N_REP, len(items)), replace=False)]
    for r in reps:
        r["group"] = r["group"] + "_repeat"
    items += reps

    rng.shuffle(items)
    for n, c in enumerate(items):
        c["order"] = n + 1
        c["item_id"] = "%s-%03d" % (lang, n + 1)
    key = pd.DataFrame(items)[["order", "item_id", "group", "src_id",
                               "intent_label", "english_source", "target_text"]]
    blind = key[["order", "item_id", "english_source", "target_text", "intent_label"]].copy()
    blind["judge1_category"] = ""
    blind["judge2_category"] = ""
    blind["judge1_note"] = ""
    blind["judge2_note"] = ""
    blind.to_csv(os.path.join(OUT, "P1_%s_blind.csv" % lang), index=False, encoding="utf-8-sig")
    key.to_csv(os.path.join(OUT, "P1_%s_key_PRIVATE.csv" % lang), index=False, encoding="utf-8-sig")
    print("%-6s 随机 %3d + 翻转 %3d + 重复 %2d = %d 行  -> P1_%s_blind.csv"
          % (name, len(rand), len(flips), len(reps), len(blind), lang))
    print("       翻转句池 %d（L 对 / MT 错，字符级 TextCNN seed42）" % len(
        [(i, u, it) for (i, u, it) in keep if ok_l[idx_of[i]] and not ok_m[idx_of[i]]]))

# ---------------- 中文空格固定子集基线 ----------------
print()
print("=" * 96)
print("中文空格条件下的基线数字（固定子集）")
print("=" * 96)
for lang, loc, nm in [("ko", "ko-KR", "韩语"), ("zh", "zh-CN", "中文")]:
    lrows, mrows = rd(loc, MASSIVE), rd(loc, MTDIR)
    lte = [(i, u, it) for i, u, p, it in lrows if p == "test"]
    mte = {i: u for i, u, p, it in mrows if p == "test"}
    ltr = set(u for i, u, p, it in lrows if p == "train")
    mtr = set(u for i, u, p, it in mrows if p == "train")
    keep = [k for k in range(len(lte)) if lte[k][1] not in ltr and mte.get(lte[k][0], "") not in mtr]
    labels = [lte[k][2] for k in keep]
    vc = pd.Series(labels).value_counts()
    maj = vc.iloc[0] / len(labels) * 100
    print("  %s 固定子集 %d 句，%d 个意图" % (nm, len(keep), vc.size))
    print("     多数类 = %s，占 %.2f%%    均匀随机 = 1.67%%" % (vc.index[0], maj))
    # 实际预测的类别分布（空格 TextCNN seed42）
    p = os.path.join(SP, "runs_pred_loc",
                     "pred_massive_localized_in-language_%s_TextCNN_whitespace_s42.csv" % lang)
    if os.path.exists(p):
        df = pd.read_csv(p)
        gold = df["gold"].values[keep]
        pred = df["pred"].values[keep]
        vp = pd.Series(pred).value_counts()
        print("     模型预测的众数类占比 = %.2f%%；预测不同类别数 = %d / %d"
              % (vp.iloc[0] / len(pred) * 100, vp.size, vc.size))
        print("     （若预测几乎集中于一类，说明模型未在读文本）")

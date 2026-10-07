# -*- coding: utf-8 -*-
"""P1 判定盲表生成。

评判者看到的：英文源句、MT 译文、意图标签名称。
不提供：本地化参考译文、模型预测（避免锚定）。

两个样本：
  A. 随机 100 句（按意图分层）—— 估计三类比例
  B. 50 句「L 训练模型在本地化测试答对、在 MT 测试答错」—— 直接检验翻转归因
两个表各混入约 10 句重复项（以不同 item_id 出现），用于检验评判者自身一致性。
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

# 意图标签中文名（MASSIVE 无官方中文标签，此处只给英文标签名，避免自造）
CFG = {"ko": ("ko-KR", "Korean", "TextCNN"), "zh": ("zh-CN", "Chinese", "TextCNN")}


def rd_ids(loc, root):
    """返回 {id: utt} 与按顺序的 [(id, utt, partition, intent)]"""
    rows = []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append((r["id"], r["utt"], r["partition"], r["intent"]))
    return rows


en_rows = rd_ids("en-US", MASSIVE)
en_test = {i: (u, it) for i, u, p, it in en_rows if p == "test"}
print("英文测试句: %d" % len(en_test))

for lang, (loc, name, model) in CFG.items():
    lrows = rd_ids(loc, MASSIVE)
    mrows = rd_ids(loc, MTDIR)
    lte = [(i, u, it) for i, u, p, it in lrows if p == "test"]
    mte = {i: u for i, u, p, it in mrows if p == "test"}
    print("%s 本地化测试 %d  MT 测试 %d" % (name, len(lte), len(mte)))

    # 固定评测子集（与 2x2 一致）
    ltr = set(u for i, u, p, it in lrows if p == "train")
    mtr = set(u for i, u, p, it in mrows if p == "train")
    keep = [(i, u, it) for (i, u, it) in lte
            if (u not in ltr) and (mte.get(i, "") not in mtr)]
    print("  固定子集: %d" % len(keep))

    # 取 L 训练模型（字符级 TextCNN）的逐句对错
    lp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_localized_in-language_%s_%s_char_s*.csv" % (lang, model))))
    mp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_mt-test_in-language_%s_%s_char_s*.csv" % (lang, model))))
    ok_l = ok_m = None
    if len(lp) == 3 and len(mp) == 3:
        dp = [pd.read_csv(p) for p in lp]
        dm = [pd.read_csv(p) for p in mp]
        ok_l = (dp[0]["pred"].values == dp[0]["gold"].values)
        ok_m = (dm[0]["pred"].values == dm[0]["gold"].values)
        print("  用 seed42 的字符级 %s 判定翻转" % model)
    else:
        print("  !! 缺 pred 文件，无法定位翻转句")

    # 索引映射：keep 里的条目在原测试列表中的下标
    idx_of = {i: k for k, (i, u, it) in enumerate(lte)}
    # ---- 样本 A：随机 100，按意图分层 ----
    by_intent = {}
    for (i, u, it) in keep:
        by_intent.setdefault(it, []).append((i, u, it))
    intents = sorted(by_intent)
    n_a = 100
    # 按意图规模比例分配，每个意图至少 1 句，再补齐到 100
    total = sum(len(by_intent[it]) for it in intents)
    quota = {it: max(1, int(round(n_a * len(by_intent[it]) / total))) for it in intents}
    while sum(quota.values()) > n_a:
        it = max(quota, key=lambda k: quota[k])
        if quota[it] > 1:
            quota[it] -= 1
        else:
            break
    order = sorted(intents, key=lambda k: -len(by_intent[k]))
    j = 0
    while sum(quota.values()) < n_a and j < 10000:
        it = order[j % len(order)]
        if quota[it] < len(by_intent[it]):
            quota[it] += 1
        j += 1
    picked = []
    for it in intents:
        pool = by_intent[it]
        take = min(quota[it], len(pool))
        sel = rng.choice(len(pool), size=take, replace=False)
        picked += [pool[jj] for jj in sel]
    if len(picked) > n_a:
        picked = [picked[jj] for jj in rng.choice(len(picked), size=n_a, replace=False)]
    print("  样本 A: %d 句，覆盖 %d 个意图" % (len(picked), len(set(p[2] for p in picked))))

    # ---- 样本 B：L 对 / MT 错 ----
    flips = []
    if ok_l is not None:
        for (i, u, it) in keep:
            k = idx_of[i]
            if k < len(ok_l) and ok_l[k] and k < len(ok_m) and not ok_m[k]:
                flips.append((i, u, it))
    print("  样本 B: %d 句（L 对 / MT 错）" % len(flips))
    if len(flips) > 50:
        flips = [flips[j] for j in rng.choice(len(flips), size=50, replace=False)]

    # ---- 组装并混入重复项 ----
    def build(rows, tag, n_rep=10):
        items = []
        for n, (i, u, it) in enumerate(rows):
            items.append(dict(
                item_id="%s-%s-%03d" % (lang, tag, n + 1),
                english_source=en_test.get(i, ("<missing>", ""))[0],
                target_text=mte.get(i, "<missing>"),
                intent_label=it))
        if rows:
            reps = rng.choice(len(items), size=min(n_rep, len(items)), replace=False)
            for n, r in enumerate(reps):
                c = dict(items[r])
                c["item_id"] = "%s-%s-R%02d" % (lang, tag, n + 1)
                items.append(c)
        rng.shuffle(items)
        for n, c in enumerate(items):
            c["order"] = n + 1
            c["judge1_category"] = ""
            c["judge2_category"] = ""
            c["judge1_note"] = ""
            c["judge2_note"] = ""
        cols = ["order", "item_id", "english_source", "target_text", "intent_label",
                "judge1_category", "judge2_category", "judge1_note", "judge2_note"]
        return pd.DataFrame(items)[cols]

    A = build(picked, "A")
    B = build(flips, "B")
    A.to_csv(os.path.join(OUT, "P1_%s_random100.csv" % lang), index=False, encoding="utf-8-sig")
    B.to_csv(os.path.join(OUT, "P1_%s_flips50.csv" % lang), index=False, encoding="utf-8-sig")
    print("  已写出 P1_%s_random100.csv (%d 行) / P1_%s_flips50.csv (%d 行)"
          % (lang, len(A), lang, len(B)))
    print()

# 说明文件
with open(os.path.join(OUT, "README_judging.md"), "w", encoding="utf-8") as f:
    f.write("""# P1 判定说明（MT 标签保真度抽检）

## 评判者能看到什么
- `english_source`：英文源句
- `target_text`：机翻译文（韩语或中文）
- `intent_label`：该句的意图标签名（英文）

**不要提供**：本地化参考译文、任何模型预测 —— 避免锚定。

## 判定三类（填在 judge1_category / judge2_category）
| 代码 | 含义 |
|---|---|
| **A** | 意图与语义均保持 |
| **B** | 意图保持，但内容有出入（漏译、加译、错词、语体异常） |
| **C** | 意图已变 |

`judge1_note` / `judge2_note` 记录理由，C 类必须写。

## 两个样本
- `P1_<lang>_random100.csv`：按意图分层随机抽 100 句 → 估计三类比例
- `P1_<lang>_flips50.csv`：50 句「L 训练模型在本地化测试答对、在机翻测试答错」→ 检验这些翻转有多少能归因于 MT 错误

## 一致性
各表**混入约 10 句重复项**（item_id 以 `-R` 结尾，内容与某一行完全相同但顺序不同）。
判完后请核对重复项的两次判定是否一致；不一致比例即为判定噪声的上界。

## 独立性要求
- **至少两位评判者**，各自独立判定后再比对（不要商量后统一）
- **韩语必须由熟悉韩语的人判定**；中文可由作者直接判，再请他人复核 20–30 句
- 作者不应是唯一判定者

## 报告
- 三类比例 + 置信区间（n=100、比例约 5% 时约 ±4 点）
- 重复项一致率
- **单列 C 类（意图已变）句子在 E4a 中的预测准确率**，看它们是否构成残余下降的主要部分
""")
print("已写出 README_judging.md")

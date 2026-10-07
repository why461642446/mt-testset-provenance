# -*- coding: utf-8 -*-
"""P1 人工抽检：完整统计。

盲表被 Excel 改存过 -> 韩语 CP949、中文 GBK。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
P1 = os.path.join(SP, "p1_judging")
ENCD = {"ko": "cp949", "zh": "gbk"}
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labs)}
    n = len(a)
    M = np.zeros((len(labs), len(labs)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / (n * n)
    if pe >= 1:
        return float("nan"), float("nan"), float("nan"), po
    k = (po - pe) / (1 - pe)
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return k, k - 1.96 * se, k + 1.96 * se, po


def rd(loc, root):
    te = []
    ids = []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "test":
            te.append(r["utt"])
            ids.append(r["id"])
    return te, ids


print("=" * 104)
print("P1 人工抽检结果")
print("=" * 104)

DATA = {}
for lang, enc in ENCD.items():
    f = "P1_%s_blind.csv" % lang
    d = pd.read_csv(os.path.join(P1, f), encoding=enc)
    key = pd.read_csv(os.path.join(P1, "P1_%s_key_PRIVATE.csv" % lang), encoding="utf-8")
    d["j1"] = d["judge1_category"].astype(str).str.strip().str.upper()
    d["j2"] = d["judge2_category"].astype(str).str.strip().str.upper()
    d = d.merge(key[["item_id", "group", "src_id"]], on="item_id", how="left")
    DATA[lang] = d

for lang, d in DATA.items():
    nm = "Korean" if lang == "ko" else "Chinese"
    print()
    print("═" * 104)
    print("%s  (n = %d)" % (nm, len(d)))
    print("═" * 104)

    # ---------- 1. 三类比例（只用 random 100）----------
    r = d[d["group"] == "random"]
    print()
    print("① 随机 100 句的三类比例")
    for who, col in [("评判者1", "j1"), ("评判者2", "j2")]:
        vc = r[col].value_counts()
        parts = []
        for c in ["A", "B", "C"]:
            k = int(vc.get(c, 0))
            lo, hi = wilson(k, len(r))
            parts.append("%s %.1f%% [%.1f, %.1f]" % (c, 100 * k / len(r), lo, hi))
        bc = int(vc.get("B", 0)) + int(vc.get("C", 0))
        lo, hi = wilson(bc, len(r))
        print("   %s   %s   |  有误(B+C) %.1f%% [%.1f, %.1f]"
              % (who, "  ".join(parts), 100 * bc / len(r), lo, hi))

    # ---------- 2. 翻转 vs 随机 ----------
    print()
    print("② 翻转句 vs 随机句 —— 这是关键检验")
    print("   %-10s %-10s %6s %8s %8s %8s" % ("评判者", "组", "n", "B+C率", "C率", "A率"))
    for who, col in [("评判者1", "j1"), ("评判者2", "j2")]:
        for g, gl in [("random", "随机"), ("flip", "翻转")]:
            s = d[d["group"] == g]
            bc = int((s[col].isin(["B", "C"])).sum())
            cc = int((s[col] == "C").sum())
            aa = int((s[col] == "A").sum())
            lob, hib = wilson(bc, len(s))
            loc_, hic = wilson(cc, len(s))
            print("   %-10s %-10s %6d %7.1f%% %7.1f%% %7.1f%%    B+C 95%%CI [%.1f, %.1f]  C 95%%CI [%.1f, %.1f]"
                  % (who, gl, len(s), 100 * bc / len(s), 100 * cc / len(s), 100 * aa / len(s),
                     lob, hib, loc_, hic))

    # ---------- 3. kappa ----------
    k, lo, hi, po = kappa(list(d["j1"]), list(d["j2"]))
    print()
    print("③ 评判者间一致性（全部 162 行）")
    print("   Cohen's kappa = %.3f  [%.3f, %.3f]   观察一致率 = %.1f%%"
          % (k, lo, hi, 100 * po))
    rk, rlo, rhi, rpo = kappa(list(r["j1"]), list(r["j2"]))
    print("   仅随机 100 句: kappa = %.3f [%.3f, %.3f]   一致率 %.1f%%"
          % (rk, rlo, rhi, 100 * rpo))

    # ---------- 4. 重复项一致率 ----------
    print()
    print("④ 重复项一致率（同一句两次判定是否相同）")
    rep = d[d["group"].isin(["random_repeat", "flip_repeat"])]
    base = d[d["group"].isin(["random", "flip"])]
    for who, col in [("评判者1", "j1"), ("评判者2", "j2")]:
        same = 0
        tot = 0
        for _, row in rep.iterrows():
            mate = base[(base["src_id"] == row["src_id"])]
            if len(mate) == 0:
                continue
            tot += 1
            if str(mate.iloc[0][col]).strip().upper() == str(row[col]).strip().upper():
                same += 1
        if tot:
            print("   %s  一致 %d / %d = %.1f%%   (不一致率 = 判定噪声上界 %.1f%%)"
                  % (who, same, tot, 100 * same / tot, 100 * (tot - same) / tot))
        else:
            print("   %s  无法配对（%d 条重复）" % (who, len(rep)))

pd.concat([v.assign(lang=k) for k, v in DATA.items()]).to_csv(
    os.path.join(SP, "table_p1_judged.csv"), index=False, encoding="utf-8")
print()
print("  已写出 table_p1_judged.csv")

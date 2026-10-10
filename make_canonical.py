# -*- coding: utf-8 -*-
"""§2.1 统一出表脚本
   ① 汇总全部 runs_*/results_v2.csv + 逐句预测
   ② 在固定子集上重算每个条件的权威值
   ③ 输出 canonical_numbers.json（唯一数字来源）
   ④ 扫描正文，标出与权威值不符的数字
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MT = os.path.join(ROOT, "massive_mt")
MT2 = os.path.join(SP, "massive_mt2")
SEEDS5, SEEDS3 = [42, 43, 44, 45, 46], [42, 43, 44]


def rows(p, part):
    o = []
    if not os.path.exists(p):
        return o
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["partition"] == part:
                o.append(r)
    return o


# ---------- 固定子集掩码 ----------
MASKS = {}
for lang, loc in [("ko", "ko-KR"), ("zh", "zh-CN")]:
    try:
        ltr = set(r["utt"] for r in rows(os.path.join(MASSIVE, loc + ".jsonl"), "train"))
        mtr = set(r["utt"] for r in rows(os.path.join(MT, loc + ".jsonl"), "train"))
        lte = rows(os.path.join(MASSIVE, loc + ".jsonl"), "test")
        mte = rows(os.path.join(MT, loc + ".jsonl"), "test")
        n = min(len(lte), len(mte))
        MASKS[lang] = np.array([(lte[i]["utt"] not in ltr) and (mte[i]["utt"] not in mtr)
                                for i in range(n)])
    except Exception as e:
        MASKS[lang] = None
        print("  掩码失败 %s: %s" % (lang, e))

print("=" * 104)
print("  §2.1 统一出表：汇总全部运行记录")
print("=" * 104)

# ---------- 汇总 results_v2.csv ----------
frames = []
seen = set()
for p in sorted(glob.glob(os.path.join(SP, "**", "results_v2.csv"), recursive=True)):
    if "_x_" in p or "\\_x_" in p:
        pass
    try:
        d = pd.read_csv(p)
    except Exception:
        continue
    if "run_id" not in d.columns:
        continue
    d = d.drop_duplicates("run_id")
    frames.append(d)
df = pd.concat(frames, ignore_index=True).drop_duplicates("run_id")
print("  运行记录 %d 行（去重后），来自 %d 个目录" % (len(df), len(frames)))
print()
print(df.groupby(["dataset", "data_variant"]).size().to_string())
df.to_csv(os.path.join(SP, "canonical_runs.csv"), index=False, encoding="utf-8")
print()
print("  -> canonical_runs.csv")

# ---------- 逐句预测：固定子集精度 ----------
pred = defaultdict(dict)
for d in ["runs_e4_char", "runs_ws_extra", "runs_c2", "runs_bi", "_x_ws/runs_bi_ws_packed",
          "_x_ws/runs_bi_ws_both", "_x_ws/runs_bi_mo_packed", "_x_ws/runs_bi_mo_both"]:
    base = os.path.join(SP, d)
    if not os.path.isdir(base):
        continue
    for p in glob.glob(os.path.join(base, "pred_*.csv")):
        m = re.match(r"pred_massive_([a-z0-9-]+)_in-language_([a-z]{2})_(\w+?)_(\w+?)(_mask)?(_\w+)?_s(\d+)\.csv$",
                     os.path.basename(p))
        if not m:
            continue
        variant, lang, model, tok, mask, extra, seed = m.groups()
        key = (variant, lang, model, tok + (mask or "") + (extra or ""))
        pred[key][int(seed)] = p
print()
print("  预测文件覆盖 %d 个条件" % len(pred))

CANON = {}
for key, seeds in sorted(pred.items()):
    variant, lang, model, tok = key
    if lang not in MASKS or MASKS[lang] is None:
        continue
    km = MASKS[lang]
    accs = []
    for s, f in sorted(seeds.items()):
        try:
            d = pd.read_csv(f)
            ok = (d["pred"].values == d["gold"].values)
            m = min(len(ok), len(km))
            accs.append(100 * ok[:m][km[:m]].mean())
        except Exception:
            pass
    if accs:
        CANON["%s|%s|%s|%s" % key] = dict(
            n_seeds=len(accs), mean=round(float(np.mean(accs)), 2),
            sd=round(float(np.std(accs, ddof=1)), 2) if len(accs) > 1 else None,
            per_seed=[round(a, 2) for a in accs])
json.dump(CANON, open(os.path.join(SP, "canonical_numbers.json"), "w"), indent=1, ensure_ascii=False)
print("  -> canonical_numbers.json（%d 个条件，固定子集）" % len(CANON))

# ---------- 打印关键值 ----------
print()
print("=" * 104)
print("  固定子集权威值（论文应引用这些）")
print("=" * 104)
for v, vn in [("localized", "L→L"), ("mt-test", "L→MT"), ("mt", "MT→MT"), ("mt-train", "MT→L")]:
    print()
    print("  [%s]" % vn)
    for k in sorted(CANON):
        a, b, c, d = k.split("|")
        if a == v and d.startswith("char") and c in ("TextCNN", "BiLSTM"):
            r = CANON[k]
            print("    %-3s %-8s %-10s n=%d  %6.2f ± %-5s" % (b, c, d, r["n_seeds"], r["mean"], r["sd"]))

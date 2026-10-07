# -*- coding: utf-8 -*-
"""清理 runs_e4_char 的污染：
   1) 备份 results_v2.csv
   2) 删掉 12 个 localized run_id 的全部行（它们的 pred/cm/history 文件也已被覆盖）
   3) 删掉对应的 pred_/cm_/history_ 文件，使脚本重新运行这些格子
   mt-test 与 mt 的 24 行不动。
"""
import glob
import os
import shutil
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
D = os.path.join(SP, "runs_e4_char")
CSV = os.path.join(D, "results_v2.csv")

assert os.path.abspath(D).startswith(os.path.abspath(SP)), "路径越界"

shutil.copy2(CSV, CSV + ".polluted.bak")
print("已备份 -> results_v2.csv.polluted.bak")

d = pd.read_csv(CSV)
loc_ids = sorted(d[d.data_variant == "localized"].run_id.unique())
print("待清理的 localized run_id: %d 个" % len(loc_ids))
for r in loc_ids:
    print("   " + r)

# 环境变量确认：这 12 个 run_id 的 pred 文件是否已被覆盖
print()
print("检查这些 run_id 的 pred 文件是否存在：")
n_pred = 0
for rid in loc_ids:
    for p in glob.glob(os.path.join(D, "pred_%s.csv" % rid)):
        n_pred += 1
print("   找到 %d 个 pred 文件（这些文件的内容已被后续运行覆盖，需一并删除）" % n_pred)

# 删除行
before = len(d)
d2 = d[~d.run_id.isin(loc_ids)].copy()
d2.to_csv(CSV, index=False, encoding="utf-8")
print()
print("results_v2.csv: %d 行 -> %d 行（删掉 %d 行）" % (before, len(d2), before - len(d2)))
print("剩余 data_variant 分布:")
print(d2.data_variant.value_counts().to_string())

# 删除文件
removed = 0
for rid in loc_ids:
    for pat in ("pred_%s.csv", "history_%s.csv", "cm_%s.csv", "cm_%s.json", "cm_%s.png"):
        for p in glob.glob(os.path.join(D, pat % rid)):
            ap = os.path.abspath(p)
            if ap.startswith(os.path.abspath(D)):
                os.remove(ap)
                removed += 1
print()
print("已删除关联文件: %d 个" % removed)

# -*- coding: utf-8 -*-
"""用正确编码读取被 Excel 改存的盲表，检查填写情况。"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper\p1_judging"
ENC = {"P1_ko_blind.csv": "cp949", "P1_zh_blind.csv": "gbk"}

for f, enc in ENC.items():
    d = pd.read_csv(SP + "\\" + f, encoding=enc)
    print("=" * 100)
    print("%s   %d 行   (编码 %s)" % (f, len(d), enc))
    print("=" * 100)
    print("  列:", list(d.columns))
    print()
    for c in ["judge1_category", "judge2_category", "judge1_note", "judge2_note"]:
        if c in d.columns:
            s = d[c].dropna().astype(str).str.strip()
            s = s[s != ""]
            vc = s.value_counts().to_dict()
            print("  %-18s 已填 %3d / %d   取值: %s" % (c, len(s), len(d), vc))
    print()
    cols = [c for c in ["order", "english_source", "target_text", "intent_label",
                        "judge1_category", "judge2_category", "judge1_note", "judge2_note"]
            if c in d.columns]
    print("  前 5 行:")
    print(d[cols].head(5).to_string(index=False, max_colwidth=38))
    print()

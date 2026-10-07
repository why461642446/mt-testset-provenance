# -*- coding: utf-8 -*-
"""探测并读取被 Excel 改存过的 CSV 编码。"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper\p1_judging"

CANDS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "gbk", "gb18030", "big5",
         "cp1252", "latin-1", "utf-16"]

for f in ["P1_ko_blind.csv", "P1_zh_blind.csv"]:
    p = SP + "\\" + f
    raw = open(p, "rb").read()
    print("=" * 96)
    print("%s   %d 字节" % (f, len(raw)))
    print("=" * 96)
    print("  前 3 字节 (BOM?):", " ".join("%02X" % b for b in raw[:3]))
    ok = []
    for enc in CANDS:
        try:
            t = raw.decode(enc)
            # 判定是否"看起来对"：应含英文原句里的常见词
            good = ("turn" in t.lower() or "play" in t.lower() or "what" in t.lower()
                    or "alexa" in t.lower() or "please" in t.lower())
            ok.append((enc, good))
            print("     %-12s 可解码   含英文原句: %s" % (enc, good))
        except Exception as e:
            print("     %-12s 失败 (%s)" % (enc, type(e).__name__))
    print()
    good_encs = [e for e, g in ok if g]
    print("  可用编码:", good_encs)
    print()

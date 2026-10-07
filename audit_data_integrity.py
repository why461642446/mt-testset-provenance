# -*- coding: utf-8 -*-
"""D3-①：en-US 训练文件里的非 ASCII / CJK 字符比例；与 zh-CN 的 ID 重叠核对。"""
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
M = r"D:\yanjiubaogaoxiangmu\massive\1.1\data"


def rd(loc):
    rows = []
    for line in open(os.path.join(M, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append((r["id"], r["utt"], r["partition"], r["intent"]))
    return rows


CJK = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uac00-\ud7af]")   # 中日韩 + 韩文音节
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")

en = rd("en-US")
zh = rd("zh-CN")
ko = rd("ko-KR")

print("=" * 96)
print("D3-① 数据审计：en-US 文本里有没有混入中文/韩文")
print("=" * 96)

for part in ["train", "dev", "test"]:
    rows = [r for r in en if r[2] == part]
    n = len(rows)
    n_cjk = sum(1 for r in rows if CJK.search(r[1]))
    n_ascii = sum(1 for r in rows if any(ord(c) > 127 for c in r[1]))
    chars = sum(len(r[1]) for r in rows)
    cjk_chars = sum(len(CJK.findall(r[1])) for r in rows)
    print()
    print("  en-US / %s   n = %d" % (part, n))
    print("     含 CJK/韩文字符的句子     : %d  (%.3f%%)" % (n_cjk, 100 * n_cjk / n))
    print("     含任意非 ASCII 字符的句子 : %d  (%.3f%%)" % (n_ascii, 100 * n_ascii / n))
    print("     CJK/韩文字符占全部字符     : %d / %d  (%.4f%%)" % (cjk_chars, chars, 100 * cjk_chars / chars))

# 例示
ex = [r[1] for r in en if r[2] == "train" and CJK.search(r[1])][:5]
print()
print("  en-US 训练集里含 CJK 的句子示例：")
for e in ex:
    print("     %s" % e[:88])
if not ex:
    print("     （无）")

# ID 重叠
print()
print("=" * 96)
print("  划分 ID 重叠核对")
print("=" * 96)
for a, an, b, bn in [(en, "en-US", zh, "zh-CN"), (en, "en-US", ko, "ko-KR"), (zh, "zh-CN", ko, "ko-KR")]:
    for pa in ["train", "dev", "test"]:
        sa = {r[0] for r in a if r[2] == pa}
        for pb in ["train", "dev", "test"]:
            sb = {r[0] for r in b if r[2] == pb}
            ov = sa & sb
            if ov:
                print("  !! %s/%s 与 %s/%s 重叠 %d 个 ID" % (an, pa, bn, pb, len(ov)))
print("  （未打印的行表示重叠为 0）")

# 全局 ID 唯一性
print()
allsets = {}
for rows, nm in [(en, "en-US"), (zh, "zh-CN"), (ko, "ko-KR")]:
    ids = [r[0] for r in rows]
    allsets[nm] = set(ids)
    print("  %-8s 总记录 %5d   唯一 ID %5d" % (nm, len(ids), len(set(ids))))

# 量化：如果 en-US 训练集被"替换成"中文，会怎样
print()
print("=" * 96)
print("  如果训练文件被误解析成 zh-CN，会观察到什么")
print("=" * 96)
print("   zh-CN 训练集 CJK 字符占比（对照）:")
for part in ["train"]:
    rows = [r for r in zh if r[2] == part]
    chars = sum(len(r[1]) for r in rows)
    cj = sum(len(CJK.findall(r[1])) for r in rows)
    print("     zh-CN / %s : %d / %d = %.2f%%" % (part, cj, chars, 100 * cj / chars))

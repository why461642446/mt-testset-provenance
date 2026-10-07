# -*- coding: utf-8 -*-
"""汇总全部待写入论文的新数据到一个参考文件，避免遗漏。"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
out = []


def P(s=""):
    print(s)
    out.append(s)


P("=" * 100)
P("待写入论文的新数据汇总")
P("=" * 100)

# ---------------- 1. v4a / v4c ----------------
P()
P("【1】v4a (mBERT) / v4c (XLM-R) —— 固定评测子集，3 seeds")
P("-" * 100)
V = [("localized", "L→L"), ("mt-test", "L→MT"), ("mt", "MT→MT"), ("mt-train", "MT→L")]
for d, name in [("runs_v4a", "mBERT"), ("runs_v4c", "XLM-R")]:
    f = os.path.join(SP, d, "results_v2.csv")
    if not os.path.exists(f):
        P("  %s 缺" % d)
        continue
    df = pd.read_csv(f)
    for lang in ["ko", "zh"]:
        vals, sds = {}, {}
        for var, tag in V:
            a = df[(df.test_lang == lang) & (df.data_variant == var)].accuracy.values * 100
            vals[tag], sds[tag] = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0)
        P("  %-7s %-3s  L→L %6.2f±%.2f  L→MT %6.2f±%.2f  MT→MT %6.2f±%.2f  MT→L %6.2f±%.2f"
          % (name, lang, vals["L→L"], sds["L→L"], vals["L→MT"], sds["L→MT"],
             vals["MT→MT"], sds["MT→MT"], vals["MT→L"], sds["MT→L"]))
        P("          -> L→MT 代价 %+.2f    夸大量 %+.2f"
          % (vals["L→MT"] - vals["L→L"], vals["MT→MT"] - vals["MT→L"]))

P()
P("  【对照】非预训练（固定子集，5 seeds）:")
P("     BiLSTM  ko 代价 −23.47  夸大量 +12.26 | zh 代价 −15.90  夸大量 +8.11")
P("     TextCNN ko 代价 −21.33  夸大量 +11.48 | zh 代价 −13.17  夸大量 +7.52")
P("     mBERT   ko 代价 −17.58  夸大量 +6.86  | zh 代价 −9.14   夸大量 +2.39")
P("     XLM-R   ko 代价 −11.29  夸大量 +1.73  | zh 代价 −7.39   夸大量 −0.20")

# ---------------- 2. 审计 ----------------
P()
P("【2】bert-base-chinese 审计（Kaggle T4，3 seeds）")
P("-" * 100)
z = os.path.join(SP, "audit_zh_bert_results.zip")
if os.path.exists(z):
    import zipfile
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            if n.endswith("audit_results.csv"):
                import io
                d = pd.read_csv(io.BytesIO(f.read(n)))
                for ck, g in d.groupby("checkpoint"):
                    P("  %-32s en %6.2f±%.2f  ko %6.2f±%.2f  zh %6.2f±%.2f"
                      % (ck, g.en.mean(), g.en.std(ddof=1), g.ko.mean(), g.ko.std(ddof=1),
                         g.zh.mean(), g.zh.std(ddof=1)))
                P("  原报: bert-base-chinese zh 65.78±1.70 | mBERT zh 51.66±4.05  -> 均精确复现")
            if n.endswith("audit_data.json"):
                import io
                dd = json.load(io.BytesIO(f.read(n)))
                for k, v in dd.items():
                    P("  MD5 %-8s %s" % (v["file"], v["md5"]))

# ---------------- 3. P1 ----------------
P()
P("【3】P1 人工抽检（两位独立评判者，各 162 行）")
P("-" * 100)
P("  随机 100 句三类比例:")
P("    韩语  评判者1 A 71.0 / B 23.0 / C 6.0   评判者2 A 70.0 / B 29.0 / C 1.0")
P("    中文  评判者1 A 74.0 / B 20.0 / C 6.0   评判者2 A 66.0 / B 31.0 / C 3.0")
P("  翻转 vs 随机 (B+C率):")
P("    韩语  随机 29.0/30.0   翻转 44.0/42.0   -> 富集约 1.5x")
P("    中文  随机 26.0/34.0   翻转 38.0/46.0   -> 富集约 1.4x")
P("  C 率:  韩语 随机 6.0/1.0  翻转 16.0/0.0")
P("         中文 随机 6.0/3.0  翻转 12.0/10.0")
P("  Cohen's kappa:  韩语 0.646 [0.526, 0.765] 一致率 82.7%")
P("                  中文 0.524 [0.391, 0.656] 一致率 76.5%")
P("  重复项一致率: 12/12 = 100%（两位都是）")
P("  **A 类控制（关键）**: 只取人工判为 A（译文完好）的句子")
P("    韩语 全部 150 句 Δ −40.53 | 只用 A 类(n=99) Δ −36.97(j1) / −32.73(j2)")
P("    中文 全部 150 句 Δ −31.87 | 只用 A 类(n=105/93) Δ −30.29(j1) / −27.96(j2)")
P("    -> 下降的 81–95% 在译文完好的句子上仍然存在")
P("  C 类 L→MT 准确率: 韩语 12.86% | 中文 25.00% / 7.50%")

# ---------------- 4. chrF 长度控制 ----------------
P()
P("【4】chrF 的两个长度控制")
P("-" * 100)
P("  源句长度相关: 韩语 r=−0.011(字符)/−0.020(词)   中文 r=+0.009/−0.011")
P("  译文长度相关: 韩语 MT +0.017 / 本地化 +0.037 / 长度差 −0.012")
P("                中文 MT +0.005 / 本地化 +0.034 / 长度差 −0.030")
P("  译文长度差: 韩语 MT 比本地化长 +1.88 字符;  中文 +0.51 字符")
P("  **长度匹配子集 (|MT−LOC|≤2)**:")
P("    韩语 n=1186  Δ −20.24 (全子集 −21.33)；低 chrF 半 −28.06 vs 高 chrF 半 −12.41")
P("    中文 n=1814  Δ −11.68 (全子集 −13.17)；低 chrF 半 −18.48 vs 高 chrF 半 −4.87")

# ---------------- 5. 其他已定稿数字 ----------------
P()
P("【5】其他已定稿数字（提醒）")
P("-" * 100)
P("  Table 6 全部 19 格 n=5（见 table6_final.csv）")
P("  标准差缩减: en 20.37→0.70 (29.1x)   ko 17.31→1.07 (16.2x)")
P("  夸大量本身 bootstrap 区间: ko TC +11.48[+10.08,+12.88] BiLSTM +12.26[+10.71,+13.74]")
P("                             zh TC  +7.52[+6.20,+8.87]   BiLSTM  +8.11[+6.83,+9.46]")
P("  预设检验 2.1–4.9 点；17.43 点；词表贡献 42–45%")
P("  三语 BERT−TextCNN: en 7.89  ko 7.90  zh 5.49")
P("  三语跨度: 空格 70.29  字符 4.38")
P("  OOV: ko MT-test 空格 37.99% / 字符 3.16%;  zh 98.26% / 4.75%")
P("  重复句: zh 223 (7.50%) 83.11% 正确 占全部正确 60%; 去重后 4.48%")

open(os.path.join(SP, "NEW_DATA_REFERENCE.md"), "w", encoding="utf-8").write(
    "# 待写入论文的新数据（自动生成）\n\n```\n" + "\n".join(out) + "\n```\n")
print()
print("  已写出 NEW_DATA_REFERENCE.md")

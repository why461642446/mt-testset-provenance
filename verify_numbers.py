# -*- coding: utf-8 -*-
"""核对并修正教授指出的 5 处数值不一致。"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
sys.path.insert(0, SP)
from runs_loader import load_runs

R, _ = load_runs()
D = R[R["mode"] == "in-language"]
Dm = D[(D["model"] != "BiLSTM") | (D["bilstm_mask"] == True)]     # 掩码版（全量）
Dun = D[(D["model"] == "BiLSTM") & (D["bilstm_mask"] == False)]   # 未掩码


def cell(tok, var, lang, model, mask=None):
    src = Dm if not (model == "BiLSTM" and mask is False) else Dun
    d = src[(src.tokenizer == tok) & (src.data_variant == var) &
            (src.test_lang == lang) & (src["model"] == model)]
    a = d.accuracy.dropna().values * 100
    return round(a.mean(), 2), round(a.std(ddof=1), 2) if len(a) > 1 else 0.0, len(a)


print("=" * 88)
print("从数据核对教授指出的数值")
print("=" * 88)
CHECKS = [
    ("① 17.43 vs 17.6：韩 TextCNN MT→MT − L→MT",
     lambda: (cell("char", "mt", "ko", "TextCNN")[0] - cell("char", "mt-test", "ko", "TextCNN")[0], None)),
    ("② −4.1 vs '4.3 points'：英语字符级变化（空格→字符）",
     lambda: (cell("char", "localized", "en", "TextCNN")[0] - cell("whitespace", "localized", "en", "TextCNN")[0], None)),
    ("③ 73.36 vs 73.07：韩 BiLSTM 空格 掩码",
     lambda: (cell("whitespace", "localized", "ko", "BiLSTM", True)[0], None)),
    ("④ 10.31 vs 10.38：中 TextCNN 空格",
     lambda: (cell("whitespace", "localized", "zh", "TextCNN")[0], None)),
    ("⑤ 10.63 vs 10.65：中 BiLSTM 空格 掩码",
     lambda: (cell("whitespace", "localized", "zh", "BiLSTM", True)[0], None)),
]
got = {}
for name, fn in CHECKS:
    v, _ = fn()
    got[name] = round(v, 2)
    print("  %-52s = %+.2f" % (name[:52], v))

print()
print("=" * 88)
print("判定")
print("=" * 88)
VERDICT = {
    "①": ("17.43", "17.6", got["① 17.43 vs 17.6：韩 TextCNN MT→MT − L→MT"]),
    "②": ("−4.1", "4.3", got["② −4.1 vs '4.3 points'：英语字符级变化（空格→字符）"]),
    "③": ("73.36", "73.07", got["③ 73.36 vs 73.07：韩 BiLSTM 空格 掩码"]),
    "④": ("10.31", "10.38", got["④ 10.31 vs 10.38：中 TextCNN 空格"]),
    "⑤": ("10.63", "10.65", got["⑤ 10.63 vs 10.65：中 BiLSTM 空格 掩码"]),
}
for k, (a, b, v) in VERDICT.items():
    which = a if abs(abs(v) - abs(float(a.replace("−", "-")))) < abs(abs(v) - abs(float(b))) else b
    print("  %s  数据 = %+.2f  ->  正确值是 %s（稿中另一值 %s 需改）" % (k, v, which, b if which == a else a))

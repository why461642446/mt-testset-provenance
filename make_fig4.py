# -*- coding: utf-8 -*-
"""重绘 Figure 4：两个 panel，只用可用的干净配置。

(a) E4a — 本地化训练，本地化测试 vs 机翻测试
(b) E4b — 本地化全流程 vs 机翻全流程

两个 panel 都用 TextCNN 与 padding-masked BiLSTM；
unmasked BiLSTM 不进入图（已判定不可用作证据）。
所有数值来自 runs_*/results_v2.csv，不插值不估计。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
FIGDIR = os.path.join(SP, "figures")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "axes.axisbelow": True})

R = {}
for k, d in [("loc", "runs_c2"), ("mttest", "runs_c1"),
             ("mt", "runs_c1b"), ("locmask", "runs_pred_loc_mask"),
             ("mtmask", "runs_pred_mt_mask")]:
    p = os.path.join(SP, d, "results_v2.csv")
    if os.path.exists(p):
        x = pd.read_csv(p)
        x["bilstm_mask"] = x["bilstm_mask"].fillna(False).astype(bool)
        R[k] = x

# 本地化训练 + 机翻测试：TextCNN/未掩码在 runs_c1，掩码 BiLSTM 在 runs_pred_mt_mask。
# 注意：**不能**把 runs_pred_loc_mask 并进来 —— 那是本地化测试的结果，
# 混进来会把「机翻测试」的数值和「本地化测试」平均，得出错误的中间值。
R["mttest_all"] = pd.concat(
    [R[k] for k in ("mttest", "mtmask") if k in R], ignore_index=True)
R["mttest_all"]["bilstm_mask"] = R["mttest_all"]["bilstm_mask"].fillna(False).astype(bool)

# 断言：机翻侧每个条件只能有一条来源（避免再次把两个条件混在一起）
for _lang in ("ko", "zh"):
    for _m, _k in (("TextCNN", False), ("BiLSTM", True)):
        _s = R["mttest_all"][(R["mttest_all"].test_lang == _lang)
                             & (R["mttest_all"].model == _m)
                             & (R["mttest_all"].bilstm_mask == _k)]
        if len(_s):
            assert _s.data_variant.nunique() == 1, (
                "机翻侧 %s/%s 混入了多个 data_variant: %s" % (
                    _lang, _m, sorted(_s.data_variant.unique())))


def cell(tag, lang, model, mask, variant=None):
    d = R[tag]
    s = d[(d.test_lang == lang) & (d.model == model) & (d.bilstm_mask == mask)]
    if variant and "data_variant" in d:
        s = s[s.data_variant == variant]
    if "tokenizer" in d:
        s = s[s.tokenizer == "whitespace"]
    if not len(s):
        return None
    return s.accuracy.mean() * 100, s.accuracy.std() * 100


LANGS = [("ko", "Korean"), ("zh", "Chinese")]
CFG = [("TextCNN", False, "TextCNN"), ("BiLSTM", True, "BiLSTM (padding-masked)")]

# ---- 每个面板的四组柱：lang × cfg ----
panel_a = []
for lang, lname in LANGS:
    for model, mask, cname in CFG:
        x = cell("loc", lang, model, mask, "localized")
        y = cell("mttest_all", lang, model, mask)
        if x and y:
            panel_a.append((lname, cname, x, y))

panel_b = []
for lang, lname in LANGS:
    for model, mask, cname in CFG:
        x = cell("loc", lang, model, mask, "localized")
        y = cell("mt", lang, model, mask)
        if x and y:
            panel_b.append((lname, cname, x, y))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
C1, C2 = "#4C72B0", "#C44E52"

for ax, data, title, l1, l2, tag in [
    (axes[0], panel_a, "(a) E4a — localized-trained model", "Localized test", "MT test", "E4a"),
    (axes[1], panel_b, "(b) E4b — in-distribution pipeline", "Localized pipeline", "MT pipeline", "E4b"),
]:
    n = len(data)
    x = np.arange(n)
    w = 0.36
    v1 = [d[2][0] for d in data]
    e1 = [d[2][1] for d in data]
    v2 = [d[3][0] for d in data]
    e2 = [d[3][1] for d in data]
    b1 = ax.bar(x - w / 2, v1, w, yerr=e1, capsize=4, label=l1, color=C1,
                error_kw=dict(elinewidth=1.2, ecolor="#333333"))
    b2 = ax.bar(x + w / 2, v2, w, yerr=e2, capsize=4, label=l2, color=C2,
                error_kw=dict(elinewidth=1.2, ecolor="#333333"))
    for bars, vals in [(b1, v1), (b2, v2)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.6, "%.1f" % val,
                    ha="center", va="bottom", fontsize=9.5)
    # 差值标注：白底文本框，避免与柱顶数值标签重叠
    for i, d in enumerate(data):
        delta = d[3][0] - d[2][0]
        ax.annotate("", xy=(i + w / 2, d[3][0]), xytext=(i + w / 2, d[2][0]),
                    arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.1))
        ax.text(i + w / 2 + 0.22, (d[2][0] + d[3][0]) / 2, "%+.1f" % delta,
                fontsize=9.5, fontweight="bold", va="center", ha="left", color="#111111",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92))
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (d[0], d[1]) for d in data], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9.5)

fig.suptitle("Machine-translated data does not inflate measured accuracy: "
             "it understates it (MASSIVE, 60 intents, chance = 1.67%)",
             fontsize=12.5, y=1.005)
fig.tight_layout()
out = os.path.join(FIGDIR, "fig4_inflation.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print("已写出", out)
print()
print("图内数值（全部来自 results_v2.csv）:")
print("  (a) E4a")
for d in panel_a:
    print("      %-8s %-24s %6.2f -> %6.2f   Δ %+6.2f" % (
        d[0], d[1], d[2][0], d[3][0], d[3][0] - d[2][0]))
print("  (b) E4b")
for d in panel_b:
    print("      %-8s %-24s %6.2f -> %6.2f   Δ %+6.2f" % (
        d[0], d[1], d[2][0], d[3][0], d[3][0] - d[2][0]))

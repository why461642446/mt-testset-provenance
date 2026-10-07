#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py —— 生成论文图

两类：
  A. 数据集图（现在就能出，只依赖 MASSIVE 原始数据，不需要任何实验结果）
       fig1_tokenization.png   空白分词下的 token 数分布  ← C2 的量化证据
       fig2_vocabulary.png     空白"词表"规模与 hapax 比例
  B. 结果图（跑完实验后才有）
       fig3_accuracy.png       三语 × 模型 准确率
       fig4_inflation.png      本地化 vs 机翻（C1）

所有图用英文标注 —— Colab / 服务器默认没有中日韩字体，中文标注会变方块。
图内数值全部来自真实数据，不做任何估计。

用法：
    python make_figures.py                      # 只出数据集图
    python make_figures.py --outdir runs_v2     # 顺带出结果图
"""

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
LOCALES = [("en-US", "English"), ("ko-KR", "Korean"), ("zh-CN", "Chinese")]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})


def load_stats():
    """重新从原始 jsonl 统计，保证图里的数字可追溯"""
    out = []
    for loc, name in LOCALES:
        path = os.path.join(MASSIVE, loc + ".jsonl")
        if not os.path.exists(path):
            print(f"  [跳过] 找不到 {path}")
            continue
        toks, chars, vocab = [], [], Counter()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                u = r["utt"]
                t = u.split()
                toks.append(len(t))
                chars.append(len(u))
                if r["partition"] == "train":
                    vocab.update(w.lower() for w in t)
        toks = np.array(toks)
        hapax = sum(1 for v in vocab.values() if v == 1)
        out.append({"locale": loc, "name": name, "n": len(toks),
                    "mean_tokens": toks.mean(), "median_tokens": float(np.median(toks)),
                    "single_pct": 100.0 * (toks == 1).mean(),
                    "mean_chars": float(np.mean(chars)),
                    "vocab": len(vocab), "hapax_pct": 100.0 * hapax / max(1, len(vocab))})
    return pd.DataFrame(out)


def fig_tokenization(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(df))
    ax = axes[0]
    b = ax.bar(x, df["mean_tokens"], color=COLORS[:len(df)])
    ax.set_xticks(x); ax.set_xticklabels(df["name"])
    ax.set_ylabel("Mean whitespace tokens per utterance")
    ax.set_title("(a) Whitespace tokenization yields near-single-token Chinese")
    ax.set_ylim(0, max(df["mean_tokens"]) * 1.25)
    for rect, v in zip(b, df["mean_tokens"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.15,
                f"{v:.2f}", ha="center", fontsize=10)

    ax = axes[1]
    b = ax.bar(x, df["single_pct"], color=COLORS[:len(df)])
    ax.set_xticks(x); ax.set_xticklabels(df["name"])
    ax.set_ylabel("Utterances reduced to a single token (%)")
    ax.set_title("(b) Share of utterances collapsed to one token")
    ax.set_ylim(0, 108)
    for rect, v in zip(b, df["single_pct"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                f"{v:.1f}%", ha="center", fontsize=10)
    fig.suptitle("MASSIVE v1.1 — effect of whitespace tokenization by locale",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_tokenization.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print("  已写出", p)


def fig_vocabulary(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(df))
    ax = axes[0]
    b = ax.bar(x, df["vocab"], color=COLORS[:len(df)])
    ax.set_xticks(x); ax.set_xticklabels(df["name"])
    ax.set_ylabel("Whitespace vocabulary size (train)")
    ax.set_title("(a) Vocabulary induced by whitespace splitting")
    for rect, v in zip(b, df["vocab"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 120,
                f"{v:,}", ha="center", fontsize=10)

    ax = axes[1]
    b = ax.bar(x, df["hapax_pct"], color=COLORS[:len(df)])
    ax.set_xticks(x); ax.set_xticklabels(df["name"])
    ax.set_ylabel("Tokens occurring exactly once (%)")
    ax.set_title("(b) Hapax ratio — Chinese 'words' are whole sentences")
    ax.set_ylim(0, 108)
    for rect, v in zip(b, df["hapax_pct"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2,
                f"{v:.1f}%", ha="center", fontsize=10)
    fig.suptitle("MASSIVE v1.1 — whitespace vocabulary statistics (train split)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_vocabulary.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print("  已写出", p)


def fig_results(outdirs, figdir):
    frames = []
    for d in outdirs:
        p = os.path.join(d, "results_v2.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print("  [跳过] 结果图需要 results_v2.csv（跑完实验后再来）")
        return
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["run_id"], keep="last")
            .reset_index(drop=True))
    loc = df[(df["dataset"] == "massive") & (df["data_variant"] == "localized")
             & (df["mode"] == "in-language")]
    if not len(loc):
        print("  [跳过] 没有 localized + in-language 的记录")
        return

    # 这张图是对比模型，不是对比分词方案（分词消融另有 fig6）。
    # 因此固定到默认条件：whitespace 分词。
    # BiLSTM 的未掩码/加掩码两种读出端单列成组，否则会把消融条件混在一起平均，
    # 得出既不是默认条件也不是修正条件的中间值（曾经出现过 en=61 这种混合数）。
    loc = loc[loc["tokenizer"] == "whitespace"].copy()
    loc["bilstm_mask"] = loc["bilstm_mask"].fillna(False).astype(bool)
    loc["group"] = np.where(loc["model"] == "BiLSTM",
                            np.where(loc["bilstm_mask"], "BiLSTM (+mask)", "BiLSTM"),
                            loc["model"])

    agg = loc.groupby(["test_lang", "group"])["accuracy"].agg(["mean", "std"]).reset_index()
    langs = [l for l in ["en", "ko", "zh"] if l in set(agg["test_lang"])]
    order = ["TextCNN", "BiLSTM", "BiLSTM (+mask)", "BERT", "DistilBERT", "ALBERT"]
    models = [m for m in order if m in set(agg["group"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w = 0.8 / max(1, len(langs))
    x = np.arange(len(models))
    for i, lang in enumerate(langs):
        vals, errs = [], []
        for m in models:
            r = agg[(agg["test_lang"] == lang) & (agg["group"] == m)]
            vals.append(r["mean"].iloc[0] * 100 if len(r) else np.nan)
            errs.append((r["std"].iloc[0] * 100) if len(r) and not pd.isna(r["std"].iloc[0]) else 0)
        ax.bar(x + i * w, vals, w, yerr=errs, capsize=3,
               label=lang, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x + w * (len(langs) - 1) / 2)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(100 / 60, ls="--", c="gray", lw=1)
    ax.text(0.02, 100 / 60 + 1.4, "chance (1/60)", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_title("MASSIVE in-language accuracy, whitespace tokenization "
                 "(mean ± std over seeds)")
    ax.legend(title="Test locale")
    fig.tight_layout()
    p = os.path.join(figdir, "fig3_accuracy.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print("  已写出", p)

    # C1 翻译膨胀
    mt = df[(df["dataset"] == "massive") & (df["data_variant"].isin(["mt", "mt-test"]))
            & (df["mode"] == "in-language")]
    if len(mt):
        a = loc.groupby(["test_lang", "model"])["accuracy"].mean() * 100
        b = mt.groupby(["test_lang", "model"])["accuracy"].mean() * 100
        common = a.index.intersection(b.index)
        if len(common):
            labels = [f"{l}\n{m}" for l, m in common]
            x = np.arange(len(common))
            fig, ax = plt.subplots(figsize=(max(7, len(common) * 1.1), 4.4))
            ax.bar(x - 0.2, [a[k] for k in common], 0.4, label="Localized test",
                   color="#4C72B0")
            ax.bar(x + 0.2, [b[k] for k in common], 0.4, label="MT test",
                   color="#C44E52")
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("C1 — effect of machine-translated test data")
            ax.legend()
            fig.tight_layout()
            p = os.path.join(figdir, "fig4_inflation.png")
            fig.savefig(p, dpi=300); plt.close(fig)
            print("  已写出", p)
    else:
        print("  [跳过] C1 图需要 --data mt 或 mt-test 的结果")


def fig_dissociation(df, figdir):
    """核心图：填充缺陷（BiLSTM ± 掩码）与分词问题在语言间的双向分离。

    直觉：若掩码把某语言的 BiLSTM 拉回 TextCNN 水平，说明该语言卡在填充；
    若拉不动，说明卡在别处（对中文而言就是分词）。
    """
    base = df[(df["dataset"] == "massive") & (df["data_variant"] == "localized")
              & (df["mode"] == "in-language") & (df["tokenizer"] == "whitespace")]
    if not len(base):
        print("  [跳过] double-dissociation 图需要 whitespace 条件下的结果")
        return
    bl = base[base["model"] == "BiLSTM"].copy()
    tc = base[base["model"] == "TextCNN"]
    if "bilstm_mask" not in bl.columns or not bl["bilstm_mask"].fillna(False).any():
        print("  [跳过] double-dissociation 图需要 --bilstm-mask 的结果")
        return
    # 该列混有 None/NaN，dtype 是 object，必须先转成真正的 bool 才能做 ~ 运算
    bl["bilstm_mask"] = bl["bilstm_mask"].fillna(False).astype(bool)

    langs = [l for l in ["en", "ko", "zh"] if l in set(bl["test_lang"])]
    names = {"en": "English", "ko": "Korean", "zh": "Chinese"}
    x = np.arange(len(langs))
    w = 0.34
    um, um_s, mk, mk_s, tc_ref = [], [], [], [], []
    for l in langs:
        a = bl[(bl["test_lang"] == l) & (~bl["bilstm_mask"].fillna(False))]
        b = bl[(bl["test_lang"] == l) & (bl["bilstm_mask"].fillna(False))]
        c = tc[tc["test_lang"] == l]
        um.append(a["accuracy"].mean() * 100 if len(a) else np.nan)
        um_s.append(a["accuracy"] * 100)
        mk.append(b["accuracy"].mean() * 100 if len(b) else np.nan)
        mk_s.append(b["accuracy"] * 100)
        tc_ref.append(c["accuracy"].mean() * 100 if len(c) else np.nan)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(x - w / 2, um, w, label="BiLSTM, last timestep (as commonly written)",
                color="#C44E52")
    b2 = ax.bar(x + w / 2, mk, w, label="BiLSTM, last non-padding timestep (E3b)",
                color="#55A868")
    # 每个种子单独画点，展示方差
    for i, pts in enumerate(um_s):
        ax.scatter(np.full(len(pts), x[i] - w / 2), pts, s=26, color="black",
                   zorder=5, linewidths=0.6)
    for i, pts in enumerate(mk_s):
        ax.scatter(np.full(len(pts), x[i] + w / 2), pts, s=26, color="black",
                   zorder=5, linewidths=0.6)
    for i, v in enumerate(tc_ref):
        ax.hlines(v, x[i] - 0.5, x[i] + 0.5, colors="#4C72B0", linestyles="--",
                  linewidth=2, zorder=4)
    ax.hlines([], [], [], colors="#4C72B0", linestyles="--",
              label="TextCNN reference (padding-insensitive)")

    ax.set_xticks(x)
    ax.set_xticklabels([names[l] for l in langs])
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(100 / 60, ls=":", c="gray", lw=1)
    ax.text(len(langs) - 0.48, 100 / 60 + 1.5, "chance (1/60)", fontsize=8, color="gray")
    ax.set_title("Double dissociation: correcting padding recovers English and Korean,\n"
                 "but leaves Chinese near chance (dots = individual seeds)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    p = os.path.join(figdir, "fig5_dissociation.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print("  已写出", p)


def fig_tokenization_effect(df, figdir):
    """C2 的核心图：分词方案 × 语言 对下游准确率的影响。

    要传达的两件事：
      (a) 中文在空白分词下崩到随机水平，换成 jieba / 字符级后回升到与英韩同档；
      (b) 字符级并非总是更好 —— 英语反而下降，即这是语言×分词的交互效应。
    """
    base = df[(df["dataset"] == "massive") & (df["data_variant"] == "localized")
              & (df["mode"] == "in-language") & (df["model"].isin(["TextCNN", "BiLSTM"]))]
    if not len(base):
        print("  [跳过] 分词效应图需要非预训练模型的结果")
        return
    base = base.copy()
    base["bilstm_mask"] = base.get("bilstm_mask", False)
    base["bilstm_mask"] = base["bilstm_mask"].fillna(False).astype(bool)
    # 只画未掩码的 BiLSTM，避免与掩码条件混淆
    base = base[~base["bilstm_mask"]]

    toks = [t for t in ["whitespace", "morph", "char"] if t in set(base["tokenizer"])]
    langs = [l for l in ["en", "ko", "zh"] if l in set(base["test_lang"])]
    names = {"en": "English", "ko": "Korean", "zh": "Chinese"}
    tcolors = {"whitespace": "#999999", "morph": "#4C72B0", "char": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, ["TextCNN", "BiLSTM"]):
        sub = base[base["model"] == model]
        x = np.arange(len(langs))
        w = 0.8 / max(1, len(toks))
        for i, t in enumerate(toks):
            vals, errs = [], []
            for l in langs:
                r = sub[(sub["test_lang"] == l) & (sub["tokenizer"] == t)]["accuracy"]
                vals.append(r.mean() * 100 if len(r) else np.nan)
                errs.append(r.std() * 100 if len(r) > 1 else 0)
            ax.bar(x + i * w, vals, w, yerr=errs, capsize=3,
                   label=t, color=tcolors[t])
        ax.set_xticks(x + w * (len(toks) - 1) / 2)
        ax.set_xticklabels([names[l] for l in langs])
        ax.axhline(100 / 60, ls=":", c="gray", lw=1)
        ax.set_title(f"({chr(97 + (0 if model == 'TextCNN' else 1))}) {model}")
        ax.set_ylim(0, 95)
    axes[0].set_ylabel("Accuracy (%)")
    # 标注放进坐标区内，避免被画布边缘裁掉
    axes[0].annotate("chance (1/60)", xy=(0.015, 100 / 60 + 1.5),
                     xycoords=("axes fraction", "data"), fontsize=8, color="gray")
    # 图例放到面板下方，否则会压住中文那组柱子
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, title="Tokenizer", fontsize=9, title_fontsize=9,
               loc="lower center", ncol=len(toks), frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Effect of tokenization on 60-class intent classification "
                 "(mean ± std over seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    p = os.path.join(figdir, "fig6_tokenization_effect.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print("  已写出", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", nargs="+", default=["runs_v2"],
                    help="结果目录，可给多个（本地 runs_c2 与 Colab runs_v2 会自动合并）")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    figdir = args.figdir if os.path.isabs(args.figdir) else os.path.join(HERE, args.figdir)
    os.makedirs(figdir, exist_ok=True)

    print("=" * 70)
    print("make_figures.py")
    print("=" * 70)
    print("[A] 数据集图（真实数据）")
    df = load_stats()
    print(df.to_string(index=False))
    df.to_csv(os.path.join(figdir, "dataset_stats_for_figures.csv"), index=False)
    fig_tokenization(df, figdir)
    fig_vocabulary(df, figdir)
    print()
    print("[B] 结果图")
    frames = []
    for d in args.outdir:
        p = os.path.join(d, "results_v2.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if frames:
        merged = (pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset=["run_id"], keep="last")
                    .reset_index(drop=True))
        fig_dissociation(merged, figdir)
        fig_tokenization_effect(merged, figdir)
    else:
        print("  [跳过] 尚未找到 results_v2.csv")
    fig_results(args.outdir, figdir)
    print()
    print("输出目录:", figdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_results.py —— 从 results_v2.csv 生成论文表格与统计检验

用法：
    python analyze_results.py --outdir runs_v2
    python analyze_results.py --outdir runs_v2 --format latex

产出（写到 outdir/tables/）：
    table4_inlanguage.md      单语性能（model × language）
    table5_inflation.md       翻译膨胀（localized vs MT）+ Wilcoxon
    table6_tokenization.md    分词消融
    table7_zeroshot.md        零样本跨语言迁移
    table8_efficiency.md      参数量 / 体积 / 延迟 / 显存
    summary.md                全部表格合并
    stats.csv                 机器可读的汇总

设计原则：
  * 没有数据时不报错，而是明确告诉你还缺哪一组实验。
  * 不做任何估计或填充 —— 缺的数据就写 "—"。
  * 只有 n>=2 才报 std；只有成对样本齐了才做 Wilcoxon。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

MODEL_ORDER = ["TextCNN", "BiLSTM", "BERT", "DistilBERT", "ALBERT", "mBERT", "XLMR"]
TOK_ORDER = ["whitespace", "whitespace+mask", "morph", "char", "subword"]
LANG_ORDER = ["en", "ko", "zh"]


def fmt(mean, std, n):
    if pd.isna(mean):
        return "—"
    if n < 2 or pd.isna(std):
        return f"{mean*100:.2f}"
    return f"{mean*100:.2f} ± {std*100:.2f}"


def md_table(df, index=True):
    """自带的 Markdown 表格渲染，不依赖 tabulate（Colab/本地都可能没装）"""
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            cells.append("—" if (v is None or (isinstance(v, float) and pd.isna(v))
                                 or v is pd.NaT) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def wilcoxon_paired(df, key_a, key_b, group_cols, val="accuracy"):
    """在多个种子上做配对 Wilcoxon。返回 (p, n_pairs, diff)"""
    if not HAVE_SCIPY:
        return None, 0, None
    a = df[key_a].groupby(group_cols)[val].apply(list)
    b = df[key_b].groupby(group_cols)[val].apply(list)
    diffs, xs, ys = [], [], []
    for k in a.index:
        if k in b.index and len(a[k]) == len(b[k]) and len(a[k]) >= 5:
            d = np.array(a[k]) - np.array(b[k])
            diffs.append(d.mean())
            xs.extend(a[k])
            ys.extend(b[k])
    if len(xs) < 6 or np.allclose(xs, ys):
        return None, len(xs), (float(np.mean(diffs)) if diffs else None)
    try:
        p = wilcoxon(xs, ys).pvalue
    except Exception:
        p = None
    return p, len(xs), (float(np.mean(diffs)) if diffs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", nargs="+", default=["runs_v2"],
                    help="结果目录，可给多个（例如本地 runs_c2 与 Colab runs_v2），"
                         "会自动合并并按 run_id 去重")
    ap.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    args = ap.parse_args()

    frames, found, missing = [], [], []
    for d in args.outdir:
        p = os.path.join(d, "results_v2.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
            found.append(p)
        else:
            missing.append(p)
    if not frames:
        raise SystemExit(
            "[!] 一个结果文件都没找到：\n    " + "\n    ".join(missing) + "\n"
            "    请先在 Colab / 本地跑实验，例如：\n"
            "    python experiment_v2.py --dataset massive --langs en "
            "--seeds 42,43,44 --outdir runs_v2")

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["run_id"], keep="last")
            .reset_index(drop=True))
    tdir = os.path.join(args.outdir[0], "tables")
    os.makedirs(tdir, exist_ok=True)

    print("=" * 78)
    print(f"合并 {len(found)} 个结果文件，去重后 {len(df)} 条运行记录")
    for p in found:
        print("   + " + p)
    if missing:
        print("   (未找到，已跳过):")
        for p in missing:
            print("   - " + p)
    # ⚠️ 掩码消融必须与未掩码结果分开统计。
    # BiLSTM 的 tokenizer 都是 whitespace，仅靠 tokenizer 无法区分是否加了掩码，
    # 若不处理会把 {7.03, 50.84, 57.70} 与 {81.07, 80.06, 79.19} 混成 59.31±28.66。
    if "bilstm_mask" not in df.columns:
        df["bilstm_mask"] = False
    df["bilstm_mask"] = df["bilstm_mask"].fillna(False).astype(bool)
    df["cond"] = df["tokenizer"] + np.where(df["bilstm_mask"], "+mask", "")

    print("=" * 78)
    print("配置分布：")
    for col in ["dataset", "data_variant", "mode", "test_lang", "cond"]:
        if col in df.columns:
            print(f"  {col:14s}: {dict(df[col].value_counts())}")
    n_seeds = df["seed"].nunique()
    print(f"  种子数        : {n_seeds}")
    if n_seeds < 2:
        print("  [提示] 只有 1 个种子 —— 无法报告 std，也无法做 Wilcoxon。")
        print("         论文要求多种子，请加 --seeds 42,43,44 重跑。")
    if not HAVE_SCIPY:
        print("  [提示] 未安装 scipy，无法做显著性检验：pip install scipy")

    g = df.groupby(["dataset", "data_variant", "mode", "test_lang", "model",
                    "cond"])["accuracy"]
    agg = g.agg(n="count", mean="mean", std="std").reset_index()
    agg["display"] = [fmt(r["mean"], r["std"], r["n"]) for _, r in agg.iterrows()]
    agg.to_csv(os.path.join(tdir, "stats.csv"), index=False)

    tables = {}

    # ---- Table 4: 单语性能（localized, in-language）----
    sub = agg[(agg["dataset"] == "massive") & (agg["data_variant"] == "localized")
              & (agg["mode"] == "in-language")]
    if len(sub):
        piv = sub.pivot_table(index=["model", "cond"], columns="test_lang",
                              values="display", aggfunc="first")
        piv = piv.reindex(columns=[c for c in LANG_ORDER if c in piv.columns])
        tables["table4_inlanguage"] = ("表 4. MASSIVE 单语意图分类（60 类，mean ± std，%）\n"
                                       "随机基线 1.67%\n\n" + md_table(piv))
    else:
        tables["table4_inlanguage"] = ("表 4. 尚未生成。\n"
                                       "需要：--dataset massive --mode in-language "
                                       "--langs en,ko,zh --seeds 42,43,44")

    # ---- Table 5: 翻译膨胀（C1）----
    loc = agg[(agg["dataset"] == "massive") & (agg["data_variant"] == "localized")
              & (agg["mode"] == "in-language")]
    mt = agg[(agg["dataset"] == "massive") & (agg["data_variant"].isin(["mt", "mt-test"]))
             & (agg["mode"] == "in-language")]
    if len(loc) and len(mt):
        rows = []
        for lang in [l for l in ["ko", "zh"] if l in set(mt["test_lang"])]:
            for model in MODEL_ORDER:
                a = loc[(loc["test_lang"] == lang) & (loc["model"] == model)]
                b = mt[(mt["test_lang"] == lang) & (mt["model"] == model)]
                if not len(a) or not len(b):
                    continue
                a, b = a.iloc[0], b.iloc[0]
                rows.append({"语言": lang, "模型": model,
                             f"人工本地化": a["display"], f"机翻": b["display"],
                             "Δ(机翻−本地化)": f"{(b['mean']-a['mean'])*100:+.2f}"})
        if rows:
            t5 = md_table(pd.DataFrame(rows), index=False)
            extra = []
            for lang in set(mt["test_lang"]):
                d = df[(df["dataset"] == "massive") & (df["data_variant"].isin(["mt", "mt-test"]))
                       & (df["test_lang"] == lang) & (df["mode"] == "in-language")]
                d2 = df[(df["dataset"] == "massive") & (df["data_variant"] == "localized")
                        & (df["test_lang"] == lang) & (df["mode"] == "in-language")]
                m = d.merge(d2, on=["model", "tokenizer", "seed"], suffixes=("_mt", "_loc"))
                if len(m):
                    diffs = m["accuracy_mt"] - m["accuracy_loc"]
                    p = None
                    if HAVE_SCIPY and len(diffs) >= 6:
                        try:
                            p = wilcoxon(m["accuracy_mt"], m["accuracy_loc"]).pvalue
                        except Exception:
                            p = None
                    extra.append(f"- {lang}: 平均 Δ = {diffs.mean()*100:+.2f} pp "
                                 f"(n={len(diffs)} 对), Wilcoxon p = "
                                 f"{('%.4g' % p) if p is not None else '样本不足'}")
            tables["table5_inflation"] = ("表 5. 翻译膨胀：同一批英文原句，两条翻译管线（C1 核心）\n\n"
                                          + t5 + "\n\n" + "\n".join(extra))
    else:
        tables["table5_inflation"] = (
            "表 5. 尚未生成。需要两组：\n"
            "  (a) --dataset massive --data localized （默认，已有 massive/1.1/data）\n"
            "  (b) --dataset massive --data mt        （需先用机翻生成 massive_mt/*.jsonl）\n"
            "两组必须使用相同的 --langs / --models / --seeds，才能配对比较。")

    # ---- Table 6: 分词消融（C2）----
    sub = agg[(agg["dataset"] == "massive") & (agg["data_variant"] == "localized")
              & (agg["mode"] == "in-language") & (agg["model"].isin(["TextCNN", "BiLSTM"]))]
    if len(sub):
        piv = sub.pivot_table(index=["test_lang", "model"], columns="cond",
                              values="display", aggfunc="first")
        piv = piv.reindex(columns=[c for c in TOK_ORDER if c in piv.columns])
        tables["table6_tokenization"] = ("表 6. 分词方案消融（仅非预训练模型；C2 核心）\n\n"
                                         + md_table(piv))
    else:
        tables["table6_tokenization"] = "表 6. 尚未生成。"

    # ---- Table 7: 零样本 ----
    zs = agg[(agg["dataset"] == "massive") & (agg["mode"] == "zero-shot")]
    if len(zs):
        piv = zs.pivot_table(index="model", columns="test_lang", values="display",
                             aggfunc="first")
        tables["table7_zeroshot"] = ("表 7. 零样本跨语言迁移（英文训练 → 目标语言测试）\n\n"
                                     + md_table(piv))
    else:
        tables["table7_zeroshot"] = "表 7. 尚未生成。需要 --mode zero-shot --langs ko,zh"

    # ---- Table 8: 效率 ----
    if "params" in df.columns:
        eff = (df.dropna(subset=["params"])
                 .groupby("model")
                 .agg(**{"参数量(M)": ("params", lambda s: round(s.iloc[0] / 1e6, 2)),
                         "体积(MB)": ("size_mb", "first"),
                         "推理延迟(ms/sample)": ("latency_ms_per_sample", "mean"),
                         "显存峰值(MB)": ("peak_gpu_mb", "max")})
                 .reset_index())
        tables["table8_efficiency"] = ("表 8. 效率实测（推理延迟为全测试集平均）\n\n"
                                       + md_table(eff, index=False))
    else:
        tables["table8_efficiency"] = "表 8. 尚未生成。"

    # ---- 落盘 ----
    for name, content in tables.items():
        with open(os.path.join(tdir, name + ".md"), "w", encoding="utf-8") as f:
            f.write(content + "\n")

    summary = "\n\n---\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items())
    with open(os.path.join(tdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# 论文表格汇总（由 analyze_results.py 自动生成）\n\n"
                "# 所有数值均来自真实运行记录，未做任何估计或填充。\n\n" + summary + "\n")

    print()
    print(summary)
    print()
    print(f"已写出：{os.path.join(tdir, 'summary.md')} 及单表 md、stats.csv")


if __name__ == "__main__":
    main()

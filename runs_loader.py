# -*- coding: utf-8 -*-
"""规范的结果加载器（所有分析都应使用它）。

背景：同一 run_id 可能出现在多个 runs_* 目录中，原因有二：
  (a) 为保存逐条预测而把同一条件重跑到另一个目录（数值完全相同）
  (b) 同一配置在不同会话各跑一次（数值略有差异，如 en BERT 0.887000 vs 0.887021）

规则：
  1. 按 run_id 去重
  2. 保留"非空字段最多"的那一行（这样 runs_eff 的完整效率列不会被 colab 的空值覆盖）
  3. 同分时按来源优先级
  4. 额外返回一份"重跑一致性"报告：同一 run_id 出现多次且数值不同的情况
"""
import glob
import os
import sys

import pandas as pd

SP = r"D:\yanjiubaogaoxiangmu\sci_paper"

# 来源优先级：越靠前越权威（预测与效率更完整者优先）
PRIORITY = ["runs_eff", "runs_e2b", "runs_e4_char", "runs_ws_extra",
            "runs_unmask",
            "runs_v2_kaggle", "runs_colab", "runs_c2", "runs_c1b", "runs_c1",
            "runs_pred_loc", "runs_pred_mt", "runs_pred_loc_mask", "runs_pred_mt_mask",
            "pilot_cpu"]


def load_runs(verbose=False):
    fr = []
    for p in sorted(glob.glob(os.path.join(SP, "runs_*", "results_v2.csv"))):
        x = pd.read_csv(p)
        x["__src"] = os.path.basename(os.path.dirname(p))
        fr.append(x)
    R = pd.concat(fr, ignore_index=True)
    for c in ["bilstm_mask"]:
        if c in R:
            R[c] = R[c].fillna(False).astype(bool)

    data_cols = [c for c in ["accuracy", "f1_weighted", "train_seconds", "params",
                             "size_mb", "latency_ms_per_sample", "peak_gpu_mb"] if c in R]
    R["__nn"] = R[data_cols].notna().sum(axis=1)
    R["__pri"] = R["__src"].map(lambda s: PRIORITY.index(s) if s in PRIORITY else 99)

    # 重跑一致性报告（去重前）
    rep = []
    for rid, g in R.groupby("run_id"):
        if len(g) > 1:
            accs = sorted(set(round(float(a), 6) for a in g.accuracy.dropna()))
            if len(accs) > 1:
                rep.append(dict(run_id=rid, n=len(g), accs=accs,
                                spread=max(accs) - min(accs)))
    R = R.sort_values(["run_id", "__nn", "__pri"],
                      ascending=[True, False, True]).drop_duplicates("run_id", keep="first")
    R = R.drop(columns=["__nn", "__pri"]).reset_index(drop=True)
    if verbose:
        sys.stdout.reconfigure(encoding="utf-8")
        print("  加载 %d 行（去重后），来源分布:" % len(R))
        print(R.__src.value_counts().to_string())
        if rep:
            print()
            print("  ⚠ 同一 run_id 的独立重跑（数值不同）: %d 组" % len(rep))
            r = pd.DataFrame(rep).sort_values("spread", ascending=False)
            print("     最大差异 %.6f（%s）" % (r.spread.max(), r.iloc[0].run_id))
            print("     中位差异 %.6f" % r.spread.median())
    return R, pd.DataFrame(rep)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    R, rep = load_runs(verbose=True)
    print()
    print("=== 去重后各条件计数 ===")
    print(R.groupby(["dataset", "mode", "data_variant"]).size().to_string())
    print()
    print("=== 效率数据的覆盖（非空计数）===")
    for c in ["params", "size_mb", "latency_ms_per_sample", "peak_gpu_mb", "train_seconds"]:
        if c in R:
            print("  %-24s 非空 %d / %d" % (c, R[c].notna().sum(), len(R)))

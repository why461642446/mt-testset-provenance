#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_zh_bert.py —— 中文零样本 BERT 结果的完整审计（v2）

背景
----
4.4 节报告：`bert-base-chinese`（纯中文词表）在**英文**上微调后，
在**中文**测试集上零样本得到 65.78 ± 1.70，比有跨语言对齐的 mBERT（51.66）高 14 点。
这非常反常，审稿人认为最可能是**流程问题**（数据被误解析到别的 locale）。

v2 相比 v1 新增四项检查
-----------------------
 ① **记录每次数据加载实际打开的路径与 MD5**（v1 只记录命令行参数，
    无法证明"程序读的确实是那个文件"）。
 ② **随机初始化对照**：同结构但**不加载预训练权重**的模型，走完全相同的流程。
    若它也能在中文上远高于随机，说明是数据/标签泄漏而非迁移。
 ③ **逐意图拆解** 65.78% 的来源，看是否集中在含数字/拉丁字符的意图上。
 ④ **剥离数字与拉丁字符后重测**：若去掉这些字符后中文准确率崩塌，
    说明模型主要在利用这些共享符号，而非真正的中文语义理解。

同时保存**逐句预测**，供事后任何分层分析使用。

用法
----
  python audit_zh_bert.py --models bert-base-chinese --seeds 42,43,44 \
      --random-init --outdir audit_zh_bert_v2
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASSIVE_ROOT = os.path.join(ROOT, "massive", "1.1", "data")

LOCALES = {"en": "en-US", "ko": "ko-KR", "zh": "zh-CN"}
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
LR = 2e-5
CJK = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uac00-\ud7af]")
LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")

LOAD_LOG = []          # ① 每次加载的实际路径与 MD5


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_locale(loc):
    """加载一个 locale，并把**实际打开的路径与 MD5**记进 LOAD_LOG。"""
    path = os.path.join(MASSIVE_ROOT, loc + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit("[FAIL] 找不到 %s\n       请先下载并解压 MASSIVE v1.1" % path)
    m = md5_of(path)
    parts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            parts.setdefault(r["partition"], []).append(r)
    LOAD_LOG.append({
        "locale": loc,
        "abs_path": os.path.abspath(path),
        "md5": m,
        "n_total": sum(len(v) for v in parts.values()),
        "n_train": len(parts.get("train", [])),
        "n_test": len(parts.get("test", [])),
        "first_train_utt": parts.get("train", [{}])[0].get("utt", "")[:80],
        "cjk_frac": round(float(np.mean([bool(CJK.search(r["utt"]))
                                         for r in parts.get("train", [])])) if parts.get("train") else 0.0, 4),
    })
    return parts


class EncDataset(Dataset):
    def __init__(self, ids, mask, y):
        self.ids, self.mask, self.y = ids, mask, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i],
                "label": torch.tensor(self.y[i], dtype=torch.long)}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for b in loader:
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device))
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        golds.extend(b["label"].tolist())
    return np.array(preds), np.array(golds)


def build_loader(tok, texts, ys, seed, shuffle):
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=MAX_LEN, return_tensors="pt")
    ds = EncDataset(enc["input_ids"], enc["attention_mask"], ys)
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=BATCH if shuffle else 64, shuffle=shuffle, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bert-base-chinese")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--outdir", default="audit_zh_bert_v2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--random-init", action="store_true",
                    help="额外跑一个**不加载预训练权重**的同结构对照")
    ap.add_argument("--strip", action="store_true", default=True,
                    help="做数字/拉丁字符剥离消融（默认开）")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 92)
    print("audit_zh_bert.py v2 —— 中文零样本 BERT 完整审计")
    print("=" * 92)
    print("  设备    : %s%s" % (device, "  " + torch.cuda.get_device_name(0) if device == "cuda" else ""))
    print("  模型    : %s" % models)
    print("  种子    : %s" % seeds)
    print("  随机初始化对照: %s" % args.random_init)
    print("  字符剥离消融  : %s" % args.strip)
    print()

    # ---------------- ① 数据加载 + 路径/MD5 记录 ----------------
    print("-" * 92)
    print("① 数据加载 —— 记录**实际打开的路径与 MD5**")
    print("-" * 92)
    data = {}
    for lg, loc in LOCALES.items():
        data[lg] = read_locale(loc)
    for rec in LOAD_LOG:
        print("  %-4s  %s" % (rec["locale"], rec["abs_path"]))
        print("        md5=%s  train=%d test=%d  CJK句占比=%.3f"
              % (rec["md5"], rec["n_train"], rec["n_test"], rec["cjk_frac"]))
    print()
    print("  **训练集首句**（人工确认加载到的是哪一种语言）:")
    for rec in LOAD_LOG:
        print("     [%s] %s" % (rec["locale"], rec["first_train_utt"]))
    print()

    classes = sorted({r["intent"] for r in data["en"]["train"]})
    l2i = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    Xtr = [r["utt"] for r in data["en"]["train"]]
    ytr = [l2i[r["intent"]] for r in data["en"]["train"]]

    TESTS = {}
    for lg in ("en", "ko", "zh"):
        TESTS[lg] = ([r["utt"] for r in data[lg]["test"]],
                     [l2i[r["intent"]] for r in data[lg]["test"]])
    print("  平凡基线: 随机 %.2f%%  多数类 %.2f%%"
          % (100.0 / num_classes, 100.0 * max(np.bincount(np.array(TESTS["zh"][1]), minlength=num_classes)) / len(TESTS["zh"][1])))
    print()

    # ---------------- ④ 剥离数字/拉丁字符的中文测试集 ----------------
    zh_txt, zh_y = TESTS["zh"]
    has_ld = np.array([bool(LATIN_DIGIT.search(t)) for t in zh_txt])
    zh_stripped = [LATIN_DIGIT.sub(" ", t).strip() or "空" for t in zh_txt]
    print("  中文测试集里含数字/拉丁字符的句子: %d / %d = %.1f%%"
          % (has_ld.sum(), len(zh_txt), 100 * has_ld.mean()))
    print()

    rows = []
    preds_out = []
    t_start = time.time()

    JOBS = [(ck, False) for ck in models]
    if args.random_init:
        JOBS += [(ck, True) for ck in models]

    for checkpoint, rand_init in JOBS:
        tag = checkpoint + ("  [RANDOM-INIT]" if rand_init else "")
        print("=" * 92)
        print("%s" % tag)
        print("=" * 92)
        tok = AutoTokenizer.from_pretrained(checkpoint)

        # 所有测试集（含剥离版）一次性编码
        enc_te = {}
        for lg in ("en", "ko", "zh"):
            enc_te[lg] = tok(list(TESTS[lg][0]), truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")
        enc_strip = tok(zh_stripped, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt") if args.strip else None

        for seed in seeds:
            set_seed(seed)
            tr_loader = build_loader(tok, Xtr, ytr, seed, True)

            if rand_init:
                cfg = AutoConfig.from_pretrained(checkpoint, num_labels=num_classes)
                model = AutoModelForSequenceClassification.from_config(cfg).to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, num_labels=num_classes).to(device)
            opt = optim.AdamW(model.parameters(), lr=LR)

            t0 = time.time()
            for ep in range(1, args.epochs + 1):
                model.train()
                tot = 0.0
                for b in tr_loader:
                    opt.zero_grad()
                    loss = model(input_ids=b["input_ids"].to(device),
                                 attention_mask=b["attention_mask"].to(device),
                                 labels=b["label"].to(device)).loss
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                print("    %s seed=%d epoch %d/%d  loss=%.4f"
                      % (tag, seed, ep, args.epochs, tot / len(tr_loader)), flush=True)
            secs = round(time.time() - t0, 1)

            res = {"model": checkpoint, "random_init": rand_init, "seed": seed,
                   "train_seconds": secs}
            # 三个 locale
            for lg in ("en", "ko", "zh"):
                ds = EncDataset(enc_te[lg]["input_ids"], enc_te[lg]["attention_mask"], TESTS[lg][1])
                p, g = predict(model, DataLoader(ds, batch_size=64), device)
                res["%s_acc" % lg] = round(100 * accuracy_score(g, p), 4)
                res["%s_f1" % lg] = round(100 * f1_score(g, p, average="weighted"), 4)
                if lg == "zh":
                    zh_pred = p
                    # 逐句预测落盘
                    for i in range(len(g)):
                        preds_out.append({"model": checkpoint, "random_init": rand_init,
                                          "seed": seed, "row": i,
                                          "text": zh_txt[i], "gold": int(g[i]),
                                          "pred": int(p[i]), "correct": int(p[i] == g[i]),
                                          "has_latin_digit": int(has_ld[i])})

            # ③ 逐意图
            per_intent = {}
            for ci, cname in enumerate(classes):
                m = (np.array(TESTS["zh"][1]) == ci)
                if m.sum():
                    per_intent[cname] = round(100 * float((zh_pred[m] == ci).mean()), 2)
            res["zh_per_intent"] = per_intent

            # ④ 剥离消融
            if args.strip and enc_strip is not None:
                ds = EncDataset(enc_strip["input_ids"], enc_strip["attention_mask"], zh_y)
                ps, gs = predict(model, DataLoader(ds, batch_size=64), device)
                res["zh_stripped_acc"] = round(100 * accuracy_score(gs, ps), 4)
                # 分「含/不含数字拉丁」两半
                m1 = has_ld
                m0 = ~has_ld
                res["zh_acc_with_ld"] = round(100 * float((zh_pred[m1] == np.array(zh_y)[m1]).mean()), 2) if m1.sum() else None
                res["zh_acc_without_ld"] = round(100 * float((zh_pred[m0] == np.array(zh_y)[m0]).mean()), 2) if m0.sum() else None

            print("    -> en %.2f | ko %.2f | zh %.2f | zh剥离 %.2f   (%.0f s)"
                  % (res["en_acc"], res["ko_acc"], res["zh_acc"],
                     res.get("zh_stripped_acc", float("nan")), secs), flush=True)

            rows.append(res)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, "audit_results.csv"),
                                      index=False, encoding="utf-8")
            pd.DataFrame(preds_out).to_csv(os.path.join(outdir, "zh_predictions.csv"),
                                           index=False, encoding="utf-8")
            with open(os.path.join(outdir, "data_load_log.json"), "w", encoding="utf-8") as f:
                json.dump(LOAD_LOG, f, ensure_ascii=False, indent=2)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------------- 汇总 ----------------
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ["zh_per_intent"] if c in df.columns]).to_csv(
        os.path.join(outdir, "audit_results.csv"), index=False, encoding="utf-8")
    print()
    print("=" * 92)
    print("汇总")
    print("=" * 92)
    cols = ["en_acc", "ko_acc", "zh_acc", "zh_stripped_acc",
            "zh_acc_with_ld", "zh_acc_without_ld"]
    for (ck, ri), g in df.groupby(["model", "random_init"]):
        print("  %s%s" % (ck, "  [RANDOM-INIT]" if ri else ""))
        for c in cols:
            if c in g.columns and g[c].notna().any():
                v = g[c].dropna().values
                print("      %-22s %.2f ± %.2f" % (c, v.mean(), v.std(ddof=1) if len(v) > 1 else 0))
    print()
    print("  判读:")
    print("    * en 应约 85%；随机初始化对照应远低于此")
    print("    * 随机初始化若在 zh 上仍远高于随机(1.67%) -> 数据/标签泄漏")
    print("    * zh 剥离数字/拉丁后若崩塌 -> 模型主要在利用共享符号")
    print()
    print("  总耗时 %.1f 分钟" % ((time.time() - t_start) / 60))
    print("  已写出: audit_results.csv / zh_predictions.csv / data_load_log.json")
    print("AUDIT-V2-DONE")


if __name__ == "__main__":
    main()

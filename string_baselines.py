# -*- coding: utf-8 -*-
"""导师 2.2：字符串基线（零训练权重）。

默认跑 ko/zh（复现论文 Table 5k）；加 --langs de,vi,ja 与 --mt-root 可跑确认实验。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MT_DEFAULT = os.path.join(ROOT, "massive_mt")
NB = 2000
RNG = np.random.default_rng(2026)

# 语种代码 -> MASSIVE locale 目录名 / 显示名
LOCALE = {
    "en": ("en-US", "English"), "ko": ("ko-KR", "Korean"), "zh": ("zh-CN", "Chinese"),
    "de": ("de-DE", "German"), "vi": ("vi-VN", "Vietnamese"), "ja": ("ja-JP", "Japanese"),
}


def load(loc, root):
    tr, te = [], []
    path = os.path.join(root, loc + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit("[FAIL] 找不到语料文件: %s" % path)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "train":
            tr.append(r)
        elif r["partition"] == "test":
            te.append(r)
    return tr, te


def vec_fit(texts):
    v = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
    return v, v.fit_transform(texts)


def nn_predict(Xtr, ytr, Xte):
    out = np.empty(Xte.shape[0], dtype=ytr.dtype)
    for i in range(0, Xte.shape[0], 256):
        S = cosine_similarity(Xte[i:i + 256], Xtr)
        out[i:i + 256] = ytr[S.argmax(1)]
    return out


def mci(d, nboot=NB):
    d = np.asarray(d, dtype=float)
    n = len(d)
    bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(nboot)])
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def paired(a, b):
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * 100
    n = len(d)
    bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(NB)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ko,zh", help="逗号分隔，如 ko,zh 或 de,vi,ja")
    ap.add_argument("--mt-root", default="", help="机翻语料目录；留空则用 %s" % MT_DEFAULT)
    ap.add_argument("--out", default="string_baseline.json")
    args = ap.parse_args()
    MT = args.mt_root or MT_DEFAULT
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    print("  本地化语料: %s" % MASSIVE)
    print("  机翻语料  : %s" % MT)
    print("  语种      : %s" % langs)
    print()

    RESULTS = {}
    for lang in langs:
        loc, nm = LOCALE[lang]
        print("=" * 106)
        print("  %s  (%s)" % (nm, loc))
        print("=" * 106)
        Ltr, Lte = load(loc, MASSIVE)
        Mtr, Mte = load(loc, MT)
        print("  本地化 train/test %d/%d   机翻 train/test %d/%d"
              % (len(Ltr), len(Lte), len(Mtr), len(Mte)))
        lset = set(r["utt"] for r in Ltr)
        mset = set(r["utt"] for r in Mtr)
        n = min(len(Lte), len(Mte))
        idx = np.where([(Lte[i]["utt"] not in lset) and (Mte[i]["utt"] not in mset)
                        for i in range(n)])[0]
        print("  固定子集 n=%d" % len(idx))

        labels = sorted(set(r["intent"] for r in Ltr))
        lab2i = {c: i for i, c in enumerate(labels)}
        cells = {}
        for cell, tr, te in [("LL", Ltr, Lte), ("LMT", Ltr, Mte),
                             ("MTMT", Mtr, Mte), ("MTL", Mtr, Lte)]:
            ytr = np.array([lab2i[r["intent"]] for r in tr])
            yte = np.array([lab2i[te[i]["intent"]] for i in idx])
            v, Xtr = vec_fit([r["utt"] for r in tr])
            Xte = v.transform([te[i]["utt"] for i in idx])
            t0 = time.time()
            c_nn = (nn_predict(Xtr, ytr, Xte) == yte)
            lr = LogisticRegression(C=10, max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
            c_lr = (lr.predict(Xte) == yte)
            cells[cell] = dict(nn=c_nn, lr=c_lr)
            print("   %-5s  最近邻 %6.2f%%   字符n-gram LR %6.2f%%   (%.0fs)"
                  % (cell, 100 * c_nn.mean(), 100 * c_lr.mean(), time.time() - t0))

        for tag, key in [("最近邻查找", "nn"), ("字符 n-gram LR", "lr")]:
            vals = {c: mci(cells[c][key]) for c in ["LL", "LMT", "MTMT", "MTL"]}
            c1 = paired(cells["LMT"][key], cells["LL"][key])
            c2 = paired(cells["MTMT"][key], cells["MTL"][key])
            print("   %-16s LL %6.2f  LMT %6.2f  MTMT %6.2f  MTL %6.2f | "
                  "成本 %+6.2f [%+6.2f,%+6.2f] 夸大量 %+6.2f [%+6.2f,%+6.2f]"
                  % (tag, vals["LL"][0], vals["LMT"][0], vals["MTMT"][0], vals["MTL"][0],
                     c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]))
            RESULTS.setdefault(lang, {})[key] = dict(
                n=int(len(idx)), LL=round(vals["LL"][0], 2), LMT=round(vals["LMT"][0], 2),
                MTMT=round(vals["MTMT"][0], 2), MTL=round(vals["MTL"][0], 2),
                cost=[round(x, 2) for x in c1], infl=[round(x, 2) for x in c2])
        print()

    out = args.out if os.path.isabs(args.out) else os.path.join(SP, args.out)
    json.dump(RESULTS, open(out, "w"), indent=1, ensure_ascii=False)
    print("  === 与论文 TextCNN 对照 ===")
    print("    韩 成本 -21.33 夸大量 +11.48 | 中 成本 -13.17 夸大量 +7.52")
    print("  已写出 %s" % out)


if __name__ == "__main__":
    main()

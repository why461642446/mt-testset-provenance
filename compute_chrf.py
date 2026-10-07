# -*- coding: utf-8 -*-
"""chrF（Popović 2015）—— 自己实现，字符 n-gram 1..6，β=2，语料级聚合。

不做 word n-gram（那是 chrF++），因为中韩需要额外分词器。

⚠ 解释上的限制（必须写进论文）：
   MASSIVE 的本地化译文本身就是意译与省略，单一参考下 chrF 偏低
   并不等于 MT 质量差。chrF 只作辅助指标。
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\yanjiubaogaoxiangmu"
MASSIVE = os.path.join(ROOT, "massive", "1.1", "data")
MTDIR = os.path.join(ROOT, "massive_mt")
BETA, ORDER = 2.0, 6


def ngrams(s, n, strip_space=True):
    if strip_space:
        s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_corpus(hyps, refs, strip_space=True):
    """语料级 chrF。"""
    stats = {}
    for n in range(1, ORDER + 1):
        m = h = r = 0
        for hy, re_ in zip(hyps, refs):
            ch, cr = ngrams(hy, n, strip_space), ngrams(re_, n, strip_space)
            h += sum(ch.values())
            r += sum(cr.values())
            m += sum((ch & cr).values())
        stats[n] = (m, h, r)
    precs = [stats[n][0] / stats[n][1] if stats[n][1] else 0.0 for n in stats]
    recs = [stats[n][0] / stats[n][2] if stats[n][2] else 0.0 for n in stats]
    p, r = float(np.mean(precs)), float(np.mean(recs))
    if p + r == 0:
        return 0.0
    return (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r) * 100.0


def chrf_sent(hy, re_, strip_space=True):
    stats = {}
    for n in range(1, ORDER + 1):
        ch, cr = ngrams(hy, n, strip_space), ngrams(re_, n, strip_space)
        stats[n] = (sum((ch & cr).values()), sum(ch.values()), sum(cr.values()))
    precs = [stats[n][0] / stats[n][1] if stats[n][1] else 0.0 for n in stats]
    recs = [stats[n][0] / stats[n][2] if stats[n][2] else 0.0 for n in stats]
    p, r = float(np.mean(precs)), float(np.mean(recs))
    if p + r == 0:
        return 0.0
    return (1 + BETA ** 2) * p * r / (BETA ** 2 * p + r) * 100.0


def rd(loc, root):
    tr, te = [], []
    for line in open(os.path.join(root, loc + ".jsonl"), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


print("自检：chrF(x, x) 应为 100 ->", round(chrf_corpus(["안녕하세요 세계"], ["안녕하세요 세계"]), 4))
print("自检：完全不相干应接近 0 ->", round(chrf_corpus(["abc"], ["xyz"]), 4))
print()

rows = []
per_sent = {}
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    _ltr, lte = rd(loc, MASSIVE)
    _mtr, mte = rd(loc, MTDIR)
    n = min(len(lte), len(mte))
    lte, mte = lte[:n], mte[:n]

    c_corpus = chrf_corpus(mte, lte, strip_space=True)
    c_corpus_sp = chrf_corpus(mte, lte, strip_space=False)
    ss = np.array([chrf_sent(a, b) for a, b in zip(mte, lte)])
    per_sent[lang] = ss
    print("%-8s n=%d" % (nm, n))
    print("   语料级 chrF（去空格）  = %.2f" % c_corpus)
    print("   语料级 chrF（保留空格）= %.2f" % c_corpus_sp)
    print("   句级 chrF 均值 = %.2f ± %.2f   中位 %.2f   四分位 [%.2f, %.2f]"
          % (ss.mean(), ss.std(), np.median(ss),
             np.percentile(ss, 25), np.percentile(ss, 75)))
    print("   chrF < 30 的句子占比 = %.1f%%" % (100.0 * (ss < 30).mean()))
    rows.append(dict(lang=lang, n=n, chrf_corpus=round(c_corpus, 2),
                     chrf_corpus_keepspace=round(c_corpus_sp, 2),
                     chrf_sent_mean=round(ss.mean(), 2), chrf_sent_std=round(ss.std(), 2),
                     chrf_sent_median=round(float(np.median(ss)), 2),
                     pct_below_30=round(100.0 * (ss < 30).mean(), 1)))
    print()

print("=" * 96)
print("chrF 能否预测翻转？（L 训练模型：本地化答对 / 机翻答错）")
print("=" * 96)
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
import glob
for lang, loc, nm in [("ko", "ko-KR", "Korean"), ("zh", "zh-CN", "Chinese")]:
    lp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_localized_in-language_%s_TextCNN_char_s*.csv" % lang)))
    mp = sorted(glob.glob(os.path.join(
        SP, "runs_e4_char", "pred_massive_mt-test_in-language_%s_TextCNN_char_s*.csv" % lang)))
    if not lp or not mp:
        print("  %s 缺 pred" % nm)
        continue
    d = pd.read_csv(lp[0]); om = pd.read_csv(mp[0])
    ok_l = (d["pred"].values == d["gold"].values)
    ok_m = (om["pred"].values == om["gold"].values)
    ss = per_sent[lang]
    m = min(len(ss), len(ok_l), len(ok_m))
    ss, ok_l, ok_m = ss[:m], ok_l[:m], ok_m[:m]
    grp = {
        "翻转（L对/MT错）": ok_l & ~ok_m,
        "两者都对": ok_l & ok_m,
        "两者都错": ~ok_l & ~ok_m,
        "反向（L错/MT对）": ~ok_l & ok_m,
    }
    print("  %s" % nm)
    for k, mask in grp.items():
        if mask.sum():
            print("    %-18s n=%4d   chrF 均值 %5.2f ± %4.2f"
                  % (k, mask.sum(), ss[mask].mean(), ss[mask].std()))
    print("    %-18s        chrF 整体 %5.2f" % ("全部", ss.mean()))
    print()

pd.DataFrame(rows).to_csv(os.path.join(SP, "table_chrf.csv"), index=False, encoding="utf-8")
print("已写出 table_chrf.csv")

# -*- coding: utf-8 -*-
"""两项机制验证（纯 CPU，无需 GPU）：

(A) 测试集 OOV 率 —— 词表只由训练集构建，统计测试集中未登录 token 的比例。
    这是"为什么模型几乎没有 signal"的直接证据。

(B) ALBERT 词表覆盖 —— albert-base-v2 是英语 SentencePiece 词表，
    统计它对韩语/中文的 [UNK] 率与 token 数，用以区分
    "单语预训练" 与 "英语词表覆盖不足" 这两个混在一起的原因。
"""
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

MASSIVE = r"D:\yanjiubaogaoxiangmu\massive\1.1\data"
LOC = {"en": "en-US", "ko": "ko-KR", "zh": "zh-CN"}


def load(lang):
    p = os.path.join(MASSIVE, LOC[lang] + ".jsonl")
    tr, te = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (tr if r["partition"] == "train" else te if r["partition"] == "test" else []).append(r["utt"])
    return tr, te


def oov_analysis(lang, tokenizer):
    tr, te = load(lang)

    def tok(s):
        if tokenizer == "whitespace":
            return s.split()
        if tokenizer == "char":
            return list(s.replace(" ", ""))
        raise ValueError(tokenizer)

    vocab = Counter()
    for s in tr:
        vocab.update(tok(s))

    n_tok = 0
    n_oov = 0
    n_utt_all_oov = 0
    for s in te:
        ts = tok(s)
        o = [t for t in ts if vocab.get(t, 0) == 0]
        n_tok += len(ts)
        n_oov += len(o)
        if ts and len(o) == len(ts):
            n_utt_all_oov += 1
    return {
        "lang": lang, "tok": tokenizer, "vocab": len(vocab),
        "n_test_tok": n_tok,
        "oov_token_rate": 100.0 * n_oov / max(1, n_tok),
        "utt_all_oov_rate": 100.0 * n_utt_all_oov / len(te),
        "mean_tok": n_tok / len(te),
    }


print("=" * 88)
print("(A) 测试集 OOV 率   词表仅由训练集构建，测试集统计未登录 token")
print("=" * 88)
print("%-4s %-11s %8s %10s %14s %12s" % ("语言", "分词", "训练集词表", "测试token数", "OOV token率", "整句全OOV率"))
print("-" * 88)
rows = []
for lang in ["en", "ko", "zh"]:
    for tk in ["whitespace", "char"]:
        r = oov_analysis(lang, tk)
        rows.append(r)
        print("%-4s %-11s %8d %10d %13.2f%% %11.2f%%" % (
            r["lang"], r["tok"], r["vocab"], r["n_test_tok"],
            r["oov_token_rate"], r["utt_all_oov_rate"]))
print()
print("解读：中文 + 空格分词下，测试句子几乎每个 token 都是未登录词，")
print("      模型在训练时从未见过这些 token，只能输出同一个类。")

# ---------------- (B) ALBERT 词表覆盖 ----------------
print()
print("=" * 88)
print("(B) 预训练词表对韩语/中文的覆盖（albert-base-v2 vs 多语言 BERT）")
print("=" * 88)
try:
    from transformers import AutoTokenizer

    tok_al = AutoTokenizer.from_pretrained("albert-base-v2")
    tok_mb = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    unk_al = tok_al.unk_token_id
    unk_mb = tok_mb.unk_token_id

    print("%-6s %-34s %10s %12s %12s" % ("语言", "词表", "词表大小", "UNK率", "平均token数"))
    print("-" * 88)
    for lang in ["en", "ko", "zh"]:
        _, te = load(lang)
        sample = te[:1500]
        for name, tk, unk in [("albert-base-v2 (英语)", tok_al, unk_al),
                              ("bert-base-multilingual-cased", tok_mb, unk_mb)]:
            n_tok = n_unk = 0
            for s in sample:
                ids = tk(s)["input_ids"]
                n_tok += len(ids)
                n_unk += sum(1 for i in ids if i == unk)
            print("%-6s %-34s %10d %11.2f%% %12.2f" % (
                lang, name, tk.vocab_size, 100.0 * n_unk / max(1, n_tok), n_tok / len(sample)))
        print()
except Exception as e:
    print("  加载分词器失败（需要网络或本地缓存）:", type(e).__name__, str(e)[:160])

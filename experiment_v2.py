#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiment_v2.py —— 面向 SCI 论文的可复现实验框架

====== 相较 full_experiment_seeded.py 的新增能力 ======
  1. 数据集：新增 MASSIVE（en-US / ko-KR / zh-CN，专业人工本地化），与 Snips 并存
  2. 多种子：--seeds 42,43,44，输出 mean ± std
  3. 零样本跨语言迁移：--mode zero-shot（英文训练 → 韩/中直接测试）
  4. 分词消融：--tokenizer whitespace|morph|char（影响 TextCNN / BiLSTM）
  5. 效率实测：参数量 / 模型体积 / 推理延迟 / 显存峰值
  6. 断点续跑：已完成的 (配置, 种子, 模型) 自动跳过，Colab 断线不用重来
  7. 结构化输出：逐轮 history、混淆矩阵 CSV+PNG、汇总 aggregate

====== 用法示例 ======
  # 主实验：MASSIVE 三语 × 5 模型 × 3 种子（in-language）
  python experiment_v2.py --dataset massive --mode in-language \
      --langs en,ko,zh --seeds 42,43,44 --outdir runs_v2

  # 零样本迁移：英文训练，韩中测试
  python experiment_v2.py --dataset massive --mode zero-shot \
      --langs ko,zh --seeds 42,43,44 --models BERT,XLMR

  # 分词消融（只跑轻量模型，很快）
  python experiment_v2.py --dataset massive --mode in-language --langs zh \
      --models TextCNN,BiLSTM --tokenizer char --seeds 42,43,44

  # 汇总统计
  python experiment_v2.py --analyze --outdir runs_v2

协议与既有工作保持一致：
  - 分类模型 TextCNN 15 epoch / lr 1e-3 / Adam / batch 32
  - BiLSTM      15 epoch / lr 1e-3 / Adam / batch 32
  - Transformer  3 epoch / AdamW + warmup 10% + grad-clip 1.0 / batch 16
                 lr 2e-5（mBERT 等）/ 1e-5（XLMR、XLMR_LARGE）/ max_len 128
  - 全部随机源固定，cudnn.deterministic = True
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # 项目根目录：Snips 的 CSV 都在这里
SEQ_LEN = 30
BERT_MAX_LEN = 128
SPLIT_SEED = 42          # 固定划分，保证不同种子下数据划分一致

# 第二个 MT 系统的语料目录（--mt-root 可覆盖）。
# 默认 massive_mt2 = NLLB-3.3B；换谱系实验（如 DeepL）指向别的目录即可。
MT2_ROOT = os.path.join(ROOT, "massive_mt2")
# --mt-root 也作用于 mt / mt-test / mt-train（留空则维持旧的默认 massive_mt/）
MT_ROOT_OVERRIDE = None

ALL_MODELS = ["TextCNN", "BiLSTM", "BERT", "DistilBERT", "ALBERT", "mBERT", "XLMR", "XLMR_LARGE"]

MASSIVE_ROOT = os.path.join(ROOT, "massive", "1.1", "data")

# 语言代码
LANGS = ["en", "ko", "zh"]

# 数据集适配层
DS = {
    "snips": {
        "locale": {"en": "en", "ko": "korean", "zh": "chinese"},
        "train": {"en": "snips_train.csv", "ko": "snips_train_korean{suffix}.csv",
                  "zh": "snips_train_chinese{suffix}.csv"},
        "test": {"en": "snips_test.csv", "ko": "snips_test_korean{suffix}.csv",
                 "zh": "snips_test_chinese{suffix}.csv"},
        "bert": {"en": "bert-base-uncased", "ko": "bert-base-multilingual-cased",
                 "zh": "bert-base-chinese"},
        "distilbert": {"en": "distilbert-base-uncased",
                       "ko": "distilbert-base-multilingual-cased",
                       "zh": "distilbert-base-multilingual-cased"},
        "albert": {"en": "albert-base-v2", "ko": "albert-base-v2", "zh": "albert-base-v2"},
        "mbert": {"en": "bert-base-multilingual-cased", "ko": "bert-base-multilingual-cased",
                  "zh": "bert-base-multilingual-cased"},
        "xlmr": {"en": "xlm-roberta-base", "ko": "xlm-roberta-base", "zh": "xlm-roberta-base"},
        "xlmr_large": {"en": "xlm-roberta-large", "ko": "xlm-roberta-large",
                       "zh": "xlm-roberta-large"},
    },
    "massive": {
        "locale": {"en": "en-US", "ko": "ko-KR", "zh": "zh-CN",
                   "de": "de-DE", "vi": "vi-VN", "ja": "ja-JP"},
        "bert": {"en": "bert-base-uncased", "ko": "bert-base-multilingual-cased",
                 "zh": "bert-base-chinese", "de": "bert-base-multilingual-cased", "vi": "bert-base-multilingual-cased", "ja": "bert-base-multilingual-cased"},
        "distilbert": {"en": "distilbert-base-uncased",
                       "ko": "distilbert-base-multilingual-cased",
                       "zh": "distilbert-base-multilingual-cased", "de": "distilbert-base-multilingual-cased", "vi": "distilbert-base-multilingual-cased", "ja": "distilbert-base-multilingual-cased"},
        "albert": {"en": "albert-base-v2", "ko": "albert-base-v2", "zh": "albert-base-v2", "de": "albert-base-v2", "vi": "albert-base-v2", "ja": "albert-base-v2"},
        "mbert": {"en": "bert-base-multilingual-cased", "ko": "bert-base-multilingual-cased",
                  "zh": "bert-base-multilingual-cased", "de": "bert-base-multilingual-cased", "vi": "bert-base-multilingual-cased", "ja": "bert-base-multilingual-cased"},
        "xlmr": {"en": "xlm-roberta-base", "ko": "xlm-roberta-base", "zh": "xlm-roberta-base", "de": "xlm-roberta-base", "vi": "xlm-roberta-base", "ja": "xlm-roberta-base"},
        "xlmr_large": {"en": "xlm-roberta-large", "ko": "xlm-roberta-large",
                       "zh": "xlm-roberta-large", "de": "xlm-roberta-large", "vi": "xlm-roberta-large", "ja": "xlm-roberta-large"},
    },
}


# ===========================================================================
# 随机性与工具
# ===========================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id_of(dataset, variant, mode, lang, model, tok, seed, bilstm_mask=False, bilstm_readout=None):
    """统一的 run_id 生成器 —— 必须把 bilstm_mask 编进去，
    否则加了掩码的 BiLSTM 会和未加掩码的视为同一组合而被断点续跑跳过。"""
    m = "_mask" if (model == "BiLSTM" and bilstm_mask) else ""
    if model == "BiLSTM" and bilstm_readout and bilstm_readout not in ("last", "lastmasked"):
        m = "_" + bilstm_readout
    return f"{dataset}_{variant}_{mode}_{lang}_{model}_{tok}{m}_s{seed}"


# ===========================================================================
# 数据加载
# ===========================================================================
def load_snips(lang, data_variant="corrected"):
    """返回 (train_texts, train_labels, test_texts, test_labels)"""
    cfg = DS["snips"]
    # 韩/中文有 _corrected 版本（人工质检修正回写）；英文只有原始版
    suffix = "_corrected" if (data_variant == "corrected" and lang != "en") else ""
    tr = pd.read_csv(os.path.join(ROOT, cfg["train"][lang].format(suffix=suffix)))
    te = pd.read_csv(os.path.join(ROOT, cfg["test"][lang].format(suffix=suffix)))
    return (list(tr["text"]), list(tr["category"]),
            list(te["text"]), list(te["category"]))


def load_massive(lang, variant="localized", root=None):
    """直接从 MASSIVE jsonl 读取，不依赖已失效的 HF dataset 脚本。

    variant='localized' -> 官方人工本地化版本（massive/1.1/data）
    variant='mt'        -> 自己用机翻生成的对照版本（massive_mt/）
    两者 utterance id 与 intent 标签一一对应，构成 C1 的受控对比。
    """
    if root is None:
        root = MASSIVE_ROOT if variant == "localized" else os.path.join(ROOT, "massive_mt")
    loc = DS["massive"]["locale"][lang]
    path = os.path.join(root, loc + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] 找不到 MASSIVE 文件: {path}\n"
                         f"       请先下载 amazon-massive-dataset-1.1.tar.gz 并解压到 {root}")
    tr_t, tr_y, te_t, te_y = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["partition"] == "train":
                tr_t.append(r["utt"]); tr_y.append(r["intent"])
            elif r["partition"] == "test":
                te_t.append(r["utt"]); te_y.append(r["intent"])
    return tr_t, tr_y, te_t, te_y


def load_data(dataset, lang, data_variant="auto"):
    """data_variant: snips -> original|corrected ; massive -> localized|mt"""
    if dataset == "snips":
        v = "corrected" if data_variant not in ("original", "corrected") else data_variant
        tr_t, tr_y, te_t, te_y = load_snips(lang, v)
        # Snips 沿用既有协议：合并后 80/20 重切
        classes = sorted(set(tr_y) | set(te_y))
        l2i = {c: i for i, c in enumerate(classes)}
        X, y = tr_t + te_t, [l2i[vv] for vv in tr_y + te_y]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SPLIT_SEED)
        return Xtr, ytr, Xte, yte, classes
    if dataset == "massive":
        v = data_variant if data_variant in ("localized", "mt", "mt-test", "mt-train",
                                             "mt2", "mt2-test") else "localized"
        ltr_t, ltr_y, lte_t, lte_y = load_massive(lang, "localized")
        if v == "localized":
            tr_t, tr_y, te_t, te_y = ltr_t, ltr_y, lte_t, lte_y
        else:
            mtr_t, mtr_y, mte_t, mte_y = load_massive(lang, "mt",
                                               root=globals().get("MT_ROOT_OVERRIDE"))
            if v == "mt":
                # E4b：机翻训练 + 机翻测试（分布内，同一 MT 系统）
                tr_t, tr_y, te_t, te_y = mtr_t, mtr_y, mte_t, mte_y
            elif v == "mt-train":
                # translate-train：机翻训练 + 本地化测试。
                # 与 mt-test 互补，两者合起来构成 2x2（训练来源 x 测试来源）。
                # 只需要 MT 的 train 分片，不需要 test 分片。
                tr_t, tr_y, te_t, te_y = mtr_t, mtr_y, lte_t, lte_y
            elif v in ("mt2", "mt2-test"):
                # 第二个 MT 系统（--mt-root 指定，默认 massive_mt2）。
                #   mt2      = 系统A机翻训练 + 系统B机翻测试（跨系统夸大量）
                #   mt2-test = 本地化训练   + 系统B机翻测试（跨系统的测试侧成本）
                # 两者都不需要系统B的 train 分片，只需要它的 test 分片。
                a2tr, a2y, a2te, a2ty = load_massive(lang, "mt2", root=MT2_ROOT)
                if v == "mt2":
                    tr_t, tr_y, te_t, te_y = mtr_t, mtr_y, a2te, a2ty
                else:
                    tr_t, tr_y, te_t, te_y = ltr_t, ltr_y, a2te, a2ty
            else:
                # mt-test（E4a）：本地化数据训练 + 机翻数据测试。
                tr_t, tr_y, te_t, te_y = ltr_t, ltr_y, mte_t, mte_y
        classes = sorted(set(tr_y) | set(te_y))
        l2i = {c: i for i, c in enumerate(classes)}
        return tr_t, [l2i[vv] for vv in tr_y], te_t, [l2i[vv] for vv in te_y], classes
    raise SystemExit("未知数据集: " + dataset)


# ===========================================================================
# 分词器（仅作用于 TextCNN / BiLSTM）
# ===========================================================================
def make_tokenizer(kind, lang):
    if kind == "whitespace":
        return lambda s: str(s).lower().split()
    if kind == "char":
        # 去掉空格后按字符切分；对中日韩天然合理
        return lambda s: list(str(s).replace(" ", ""))
    if kind == "morph":
        if lang == "zh":
            try:
                import jieba
            except ImportError:
                # 用 RuntimeError 而非 SystemExit：前者会被逐组合捕获，
                # 让同一批里的其它分词方案继续跑；后者会直接终止整批任务。
                raise RuntimeError("需要 jieba：pip install jieba")
            return lambda s: [w for w in jieba.cut(str(s)) if w.strip()]
        if lang == "ko":
            try:
                from konlpy.tag import Okt
            except ImportError:
                raise RuntimeError("韩语形态素分析需要 konlpy + Java："
                                   "pip install konlpy（Colab 自带 JDK）")
            okt = Okt()
            return lambda s: okt.morphs(str(s))
        # 英文的 morph 退化为 whitespace
        return lambda s: str(s).lower().split()
    raise RuntimeError("未知分词方案: " + kind)


# ===========================================================================
# 模型
# ===========================================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 kernel_sizes=(3, 4, 5), num_kernels=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernels, (k, embedding_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_kernels, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        pooled = [torch.max(torch.relu(c(x)).squeeze(3), 2)[0] for c in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, 1)))


class BiLSTM(nn.Module):
    """三种读出方式。

    readout='last'      : 取最后一个时间步（未修正 —— 该位置几乎总是 PAD）
    readout='lastmasked': 取最后一个非 PAD 时间步，事后乘掩码
        **注意其残留缺陷**：双向输出在该位置拼接了前向与反向两半。
        前向半边已处理 0..t，概括整句；但反向半边是从序列末尾（PAD）倒着走的，
        到 t 时只经过了最后一个真实 token 及其后的全部 PAD，**其状态被 PAD 污染**，
        事后乘掩码无法挽回已经算过的 PAD。
    readout='packed'    : 用 pack_padded_sequence，LSTM 完全不接触 PAD，
        再取最后一个非 PAD 时间步。修正上述污染。
    readout='bothfinal' : 取两个方向的最终隐状态拼接。前向最终态在最后一个真实 token，
        反向最终态在第一个真实 token，**两半都概括整句**。
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 use_mask=False, readout=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.use_mask = use_mask
        # 兼容旧参数：use_mask=True 且未指定 readout 时沿用 'lastmasked'
        self.readout = readout if readout else ("lastmasked" if use_mask else "last")

    def forward(self, x):
        emb = self.embedding(x)
        keep = (x != 0)
        lengths = keep.sum(1).clamp(min=1)
        if self.readout == "packed":
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.size(1))
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            h = out.gather(1, idx).squeeze(1)
        elif self.readout == "bothfinal":
            _, (hn, _) = self.lstm(emb)
            h = torch.cat([hn[0], hn[1]], dim=1)      # 前向末态、反向末态
        else:
            out, _ = self.lstm(emb)
            if self.readout == "lastmasked":
                m = out * keep.unsqueeze(-1).float()
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
                h = m.gather(1, idx).squeeze(1)
            else:
                h = out[:, -1, :]
        return self.fc(self.dropout(h))


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))


class EncDataset(Dataset):
    def __init__(self, ids, mask, labels):
        self.ids, self.mask, self.labels = ids, mask, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i],
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def make_loader(ds, batch_size, shuffle, seed):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2 if os.name != "nt" else 0,
                      generator=torch.Generator().manual_seed(seed))


# ===========================================================================
# 训练与评估
# ===========================================================================
def eval_classic(model, loader, device):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(torch.max(model(xb.to(device)), 1)[1].cpu().numpy())
            golds.extend(yb.numpy())
    return preds, golds


def eval_transformer(model, loader, device):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            preds.extend(torch.max(out.logits, 1)[1].cpu().numpy())
            golds.extend(b["label"].numpy())
    return preds, golds


def train_classic(model, tr_loader, te_loader, epochs, lr, name, device, seed, num_classes):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        preds, golds = eval_classic(model, te_loader, device)
        acc = accuracy_score(golds, preds)
        hist.append({"epoch": ep, "train_loss": round(tot / len(tr_loader), 6),
                     "test_accuracy": round(float(acc), 6)})
        print(f"    {name} Epoch {ep}/{epochs}, Loss: {tot/len(tr_loader):.4f}, "
              f"Acc: {acc:.4f}", flush=True)
    preds, golds = eval_classic(model, te_loader, device)
    return (accuracy_score(golds, preds), f1_score(golds, preds, average="weighted"),
            confusion_matrix(golds, preds, labels=list(range(num_classes))), hist,
            preds, golds)


def train_transformer(model, tr_loader, te_loader, epochs, lr, name, device, seed, num_classes,
                      warmup_frac=0.1, grad_clip=1.0, weight_decay=0.01):
    """线性 warmup + 梯度裁剪 + weight decay。

    大模型在 11.5k 样本上以恒定 lr 微调时会间歇性塌进多数类盆地
    （XLMR-LARGE 上实测约半数种子停在 7% 准确率），这三项是标准补救。
    """
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(tr_loader))
    warmup_steps = max(1, int(warmup_frac * total_steps))
    step = 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in tr_loader:
            cur_lr = lr * (step + 1) / warmup_steps if step < warmup_steps else lr
            for g in opt.param_groups:
                g["lr"] = cur_lr
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            yb = b["label"].to(device)
            opt.zero_grad()
            loss = model(input_ids=ids, attention_mask=am, labels=yb).loss
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            tot += loss.item()
        preds, golds = eval_transformer(model, te_loader, device)
        acc = accuracy_score(golds, preds)
        hist.append({"epoch": ep, "train_loss": round(tot / len(tr_loader), 6),
                     "test_accuracy": round(float(acc), 6)})
        print(f"    {name} Epoch {ep}/{epochs}, Loss: {tot/len(tr_loader):.4f}, "
              f"Acc: {acc:.4f}", flush=True)
    preds, golds = eval_transformer(model, te_loader, device)
    return (accuracy_score(golds, preds), f1_score(golds, preds, average="weighted"),
            confusion_matrix(golds, preds, labels=list(range(num_classes))), hist,
            preds, golds)


def measure_efficiency(model, te_loader, device, is_transformer):
    """参数量 / 体积 / 推理延迟(ms/sample) / 显存峰值"""
    n_param = sum(p.numel() for p in model.parameters())
    size_mb = n_param * 4 / (1024 ** 2)      # fp32 权重
    model.eval()
    # 预热
    with torch.no_grad():
        for i, b in enumerate(te_loader):
            if is_transformer:
                model(input_ids=b["input_ids"].to(device),
                      attention_mask=b["attention_mask"].to(device))
            else:
                model(b[0].to(device))
            if i >= 2:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, time.time()
    with torch.no_grad():
        for b in te_loader:
            if is_transformer:
                bs = b["input_ids"].size(0)
                model(input_ids=b["input_ids"].to(device),
                      attention_mask=b["attention_mask"].to(device))
            else:
                bs = b[0].size(0)
                model(b[0].to(device))
            n += bs
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else None
    return {"params": n_param, "size_mb": round(size_mb, 2),
            "latency_ms_per_sample": round(dt * 1000 / max(1, n), 4),
            "peak_gpu_mb": round(peak, 1) if peak is not None else None}


# ===========================================================================
# 单次运行
# ===========================================================================
def run_once(cfg, device, args, outdir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dataset, mode, lang, model_name, tok_kind, seed = (
        cfg["dataset"], cfg["mode"], cfg["lang"], cfg["model"],
        cfg["tokenizer"], cfg["seed"])
    set_seed(seed)

    # ---- 数据 ----
    train_lang = "en" if mode == "zero-shot" else lang
    dv = args.data
    Xtr, ytr, _, _, classes = load_data(dataset, train_lang, dv)
    _, _, Xte, yte, classes_te = load_data(dataset, lang, dv)
    if classes != classes_te:
        raise SystemExit("[FAIL] 训练/测试语言类别集合不一致（零样本要求同一套意图）")
    num_classes = len(classes)
    if len(Xtr) == 0:
        raise SystemExit(
            "[FAIL] 训练集为空。若机翻数据只生成了 test 划分，"
            "请改用 --data mt-test（= 本地化数据训练 + 机翻数据测试），"
            "它只需 test 侧的机翻数据，且在 DeepL 免费额度内。")
    variant = ("mt" if dv == "mt" else
               "mt-test" if dv == "mt-test" else
               "mt-train" if dv == "mt-train" else
               "mt2" if dv == "mt2" else
               ("corrected" if (dataset == "snips" and dv != "original") else
                ("original" if dataset == "snips" else "localized")))

    if args.limit:
        # 每类固定条数，顺序确定，便于快速自检
        def subsample(X, y):
            per = max(1, args.limit // num_classes)
            seen, idx = Counter(), []
            for i, v in enumerate(y):
                if seen[v] < per:
                    seen[v] += 1
                    idx.append(i)
            return [X[i] for i in idx], [y[i] for i in idx]
        Xtr, ytr = subsample(Xtr, ytr)
        Xte, yte = subsample(Xte, yte)
        print(f"    [自检模式] 每类 {max(1, args.limit // num_classes)} 条 -> "
              f"train {len(Xtr)} / test {len(Xte)}")

    tag = f"{dataset}|{mode}|{lang}|{model_name}|{tok_kind}|seed{seed}"
    print(f"\n  >>> {tag}  (train={len(Xtr)}, test={len(Xte)}, classes={num_classes})",
          flush=True)

    t0 = time.time()
    is_trans = model_name not in ("TextCNN", "BiLSTM")

    if not is_trans:
        tokenize = make_tokenizer(tok_kind, lang)
        toks = [tokenize(t) for t in Xtr]
        vocab = {"<PAD>": 0, "<UNK>": 1}
        cnt = Counter(w for s in toks for w in s)
        for w, _ in cnt.most_common(10000 - 2):
            vocab[w] = len(vocab)

        def enc(texts):
            out = []
            for t in texts:
                s = [vocab.get(w, 1) for w in tokenize(t)][:SEQ_LEN]
                out.append(s + [0] * (SEQ_LEN - len(s)))
            return out

        tr_loader = make_loader(SeqDataset(enc(Xtr), ytr), 32, True, seed)
        te_loader = make_loader(SeqDataset(enc(Xte), yte), 32, False, seed)
        if model_name == "TextCNN":
            model = TextCNN(len(vocab), 100, num_classes)
        else:
            model = BiLSTM(len(vocab), 100, 128, num_classes,
                           use_mask=args.bilstm_mask,
                           readout=getattr(args, 'bilstm_readout', None))
        acc, f1, cm, hist, preds, golds = train_classic(
            model, tr_loader, te_loader,
            args.epochs_classic, 1e-3, model_name,
            device, seed, num_classes)
    else:
        pretrained = DS[dataset][model_name.lower()][lang]
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        e_tr = tokenizer(list(Xtr), truncation=True, padding="max_length",
                         max_length=BERT_MAX_LEN, return_tensors="pt")
        e_te = tokenizer(list(Xte), truncation=True, padding="max_length",
                         max_length=BERT_MAX_LEN, return_tensors="pt")
        tr_loader = make_loader(
            EncDataset(e_tr["input_ids"], e_tr["attention_mask"], ytr), 16, True, seed)
        te_loader = make_loader(
            EncDataset(e_te["input_ids"], e_te["attention_mask"], yte), 16, False, seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_classes)
        # 大模型用小学习率 —— XLMR-LARGE(560M) 在 2e-5 下会塌成多数类
        lr_use = (args.lr_transformer_large
                  if model_name in ("XLMR_LARGE", "XLMR")
                  else args.lr_transformer)
        acc, f1, cm, hist, preds, golds = train_transformer(
            model, tr_loader, te_loader,
            args.epochs_transformer, lr_use,
            f"{model_name}({pretrained})", device, seed,
            num_classes,
            warmup_frac=args.warmup_frac, grad_clip=args.grad_clip)
        pretrained = pretrained
    secs = round(time.time() - t0, 1)

    eff = measure_efficiency(model, te_loader, device, is_trans)

    # ---- 落盘 ----
    run_id = run_id_of(dataset, variant, mode, lang, model_name, tok_kind, seed,
                       args.bilstm_mask)
    os.makedirs(outdir, exist_ok=True)

    # 逐条预测落盘 —— 供配对检验（McNemar 等）使用。
    # 本地化与机翻测试集来自同一批源句，ID 集合与顺序均已验证一致，
    # 因此可按位置配对：同一 idx 在两种条件下对应同一句源句。
    if getattr(args, "save_predictions", False):
        pd.DataFrame({"idx": range(len(golds)),
                      "gold": golds,
                      "pred": list(preds)}).to_csv(
            os.path.join(outdir, f"pred_{run_id}.csv"),
            index=False, encoding="utf-8")

    pd.DataFrame(hist).assign(run_id=run_id).to_csv(
        os.path.join(outdir, f"history_{run_id}.csv"), index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(outdir, f"cm_{run_id}.csv"))
    with open(os.path.join(outdir, f"cm_{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "matrix": cm.tolist()}, f, ensure_ascii=False)
    # 混淆矩阵 PNG
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.32), max(5, num_classes * 0.30)))
    sns.heatmap(cm, annot=num_classes <= 12, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"Confusion Matrix - {run_id}", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"cm_{run_id}.png"), dpi=160)
    plt.close(fig)

    row = {"run_id": run_id, "dataset": dataset, "data_variant": variant, "mode": mode,
           "test_lang": lang,
           "train_lang": train_lang, "model": model_name, "tokenizer": tok_kind,
           "seed": seed, "n_train": len(Xtr), "n_test": len(Xte),
           "num_classes": num_classes, "accuracy": round(float(acc), 6),
           "f1_weighted": round(float(f1), 6), "train_seconds": secs,
           "pretrained": pretrained if is_trans else "",
           "bilstm_mask": bool(args.bilstm_mask) if model_name == "BiLSTM" else None,
           **eff}
    print(f"  <<< {run_id}: Acc={acc*100:.2f}%  F1={f1:.4f}  ({secs}s)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# ===========================================================================
# 汇总分析
# ===========================================================================
def analyze(outdir):
    path = os.path.join(outdir, "results_v2.csv")
    if not os.path.exists(path):
        raise SystemExit("[FAIL] 还没有结果：" + path)
    df = pd.read_csv(path)
    keys = ["dataset", "mode", "test_lang", "model", "tokenizer"]
    g = df.groupby(keys)["accuracy"]
    agg = g.agg(n="count", mean="mean", std="std", min="min", max="max").reset_index()
    agg["mean_pct"] = (agg["mean"] * 100).round(2)
    agg["std_pct"] = (agg["std"] * 100).round(2)
    agg["result"] = agg.apply(
        lambda r: f"{r['mean_pct']:.2f} ± {r['std_pct']:.2f}" if r["n"] > 1
        else f"{r['mean_pct']:.2f}", axis=1)
    agg.to_csv(os.path.join(outdir, "aggregate_v2.csv"), index=False)
    print("=" * 78)
    print("汇总（mean ± std, %）")
    print("=" * 78)
    for (ds, mode), sub in agg.groupby(["dataset", "mode"]):
        print(f"\n--- {ds} / {mode} ---")
        piv = sub.pivot_table(index=["model", "tokenizer"], columns="test_lang",
                              values="result", aggfunc="first")
        print(piv.to_string())
    print(f"\n已写出 {os.path.join(outdir, 'aggregate_v2.csv')}")


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="SCI 论文实验框架 v2")
    ap.add_argument("--dataset", default="massive", choices=["snips", "massive"])
    ap.add_argument("--mode", default="in-language", choices=["in-language", "zero-shot"])
    ap.add_argument("--langs", default="en,ko,zh")
    ap.add_argument("--models", default="all")
    ap.add_argument("--tokenizer", default="whitespace",
                    choices=["whitespace", "morph", "char"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--data", default="auto",
                    choices=["auto", "original", "corrected", "localized", "mt", "mt-test", "mt-train",
                             "mt2", "mt2-test"],
                    help="snips: original|corrected；massive: localized|mt|mt-test|mt-train|mt2|mt2-test"
                         "（mt-test = 本地化训练 + 机翻测试；mt-train = 机翻训练 + 本地化测试；"
                         "mt2 = 系统A机翻训练 + 系统B机翻测试，读 --mt-root；"
                         "mt2-test = 本地化训练 + 系统B机翻测试）；auto 自动选择")
    ap.add_argument("--mt-root", default="",
                    help="第二个 MT 系统的语料目录（mt2 / mt2-test 用）。"
                         "留空则用 <项目根>/massive_mt2；换谱系实验时指向别的目录。")
    ap.add_argument("--outdir", default="runs_v2")
    ap.add_argument("--bilstm-mask", action="store_true")
    ap.add_argument("--bilstm-readout", default=None,
                    choices=[None, "last", "lastmasked", "packed", "bothfinal"])
    ap.add_argument("--bert-max-len", type=int, default=0,
                    help="覆盖 BERT 类模型的截断长度（默认 128）。用于消除"
                         "「英语字符级平均 34.9 字符 > 30」的截断混淆：设为 64 即不截断。")
    ap.add_argument("--seq-len", type=int, default=0,
                    help="覆盖非预训练模型的序列长度（默认 30）。")
    ap.add_argument("--limit", type=int, default=0, help="每类抽样条数，用于自检")
    ap.add_argument("--epochs-classic", type=int, default=15)
    ap.add_argument("--epochs-transformer", type=int, default=3)
    ap.add_argument("--lr-transformer", type=float, default=2e-5)
    ap.add_argument("--lr-transformer-large", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--analyze", action="store_true", help="只做汇总，不训练")
    ap.add_argument("--save-predictions", action="store_true",
                    help="逐条保存测试集预测（pred_<run_id>.csv），供配对检验使用")
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)

    # 覆盖截断长度（M6：消除英语字符级条件的截断混淆）
    global BERT_MAX_LEN, SEQ_LEN, MT2_ROOT
    if args.bert_max_len:
        BERT_MAX_LEN = args.bert_max_len
        print(f"[覆盖] BERT_MAX_LEN = {BERT_MAX_LEN}")
    if args.seq_len:
        SEQ_LEN = args.seq_len
        print(f"[覆盖] SEQ_LEN = {SEQ_LEN}")
    if args.mt_root:
        MT2_ROOT = args.mt_root if os.path.isabs(args.mt_root) else os.path.join(ROOT, args.mt_root)
    if args.mt_root:
        MT_ROOT_OVERRIDE = args.mt_root if os.path.isabs(args.mt_root) else os.path.join(ROOT, args.mt_root)
        globals()["MT_ROOT_OVERRIDE"] = MT_ROOT_OVERRIDE
        print(f"MT 语料根目录：{MT_ROOT_OVERRIDE}  (--mt-root 作用于 mt / mt-test / mt-train)")
        if not os.path.isdir(MT2_ROOT):
            raise SystemExit(f"[FAIL] --mt-root 不存在: {MT2_ROOT}")
        print(f"[覆盖] MT2_ROOT = {MT2_ROOT}")

    if args.analyze:
        analyze(outdir)
        return

    models = ALL_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"未知模型 {bad}，可选 {ALL_MODELS}")
    langs = [l.strip() for l in args.langs.split(",")] if args.langs != "all" else LANGS
    seeds = [int(s) for s in args.seeds.split(",")]

    set_seed(SPLIT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("experiment_v2.py")
    print("=" * 78)
    print(f"设备     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
             "   ⚠️ 未使用 GPU，Transformer 会非常慢"))
    print(f"数据集   : {args.dataset}    模式: {args.mode}")
    print(f"语言     : {langs}    模型: {models}")
    print(f"分词     : {args.tokenizer}   种子: {seeds}")
    print(f"BiLSTM掩码: {args.bilstm_mask}")
    print(f"输出     : {outdir}")
    print(flush=True)

    res_path = os.path.join(outdir, "results_v2.csv")
    os.makedirs(outdir, exist_ok=True)
    done = set()
    if os.path.exists(res_path):
        done = set(pd.read_csv(res_path)["run_id"])
        print(f"断点续跑：已有 {len(done)} 条结果，将跳过已完成组合\n")

    # 生成运行清单
    if args.data == "auto":
        variant = "corrected" if args.dataset == "snips" else "localized"
    elif args.dataset == "snips":
        variant = "original" if args.data == "original" else "corrected"
    else:
        variant = args.data if args.data in ("mt", "mt-test", "mt-train", "mt2", "mt2-test") else "localized"

    jobs = []
    for lang in langs:
        for model in models:
            for tok in ([args.tokenizer] if model in ("TextCNN", "BiLSTM") else ["subword"]):
                for seed in seeds:
                    rid = run_id_of(args.dataset, variant, args.mode, lang, model, tok,
                                    seed, args.bilstm_mask, getattr(args, 'bilstm_readout', None))
                    if rid in done:
                        continue
                    jobs.append({"dataset": args.dataset, "mode": args.mode, "lang": lang,
                                 "model": model, "tokenizer": tok, "seed": seed,
                                 "data_variant": variant})
    print(f"待运行 {len(jobs)} 个组合" + (f"（跳过 {len(done)} 个已完成）" if done else ""))
    if not jobs:
        print("没有需要运行的任务。")
        analyze(outdir)
        return

    manifest = {"started_at": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args), "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": __import__("transformers").__version__,
                             "numpy": np.__version__, "pandas": pd.__version__}}

    rows = []
    job_failures = []
    for i, cfg in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}]", flush=True)
        try:
            rows.append(run_once(cfg, device, args, outdir))
        except Exception as e:
            print(f"  !!! 该组合失败：{type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            job_failures.append({"config": cfg, "error": f"{type(e).__name__}: {e}"})
        # 每个组合后立即写盘 —— Colab 断线也不丢进度
        if rows:
            df = pd.DataFrame(rows)
            if os.path.exists(res_path):
                df = pd.concat([pd.read_csv(res_path), df], ignore_index=True)
            df.to_csv(res_path, index=False)
            rows = []

    if job_failures:
        manifest["failures"] = job_failures
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(outdir, "manifest_v2.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        raise SystemExit(
            "[FAIL] 有 %d 个组合失败；已保留成功结果，可修复后断点续跑。" % len(job_failures))

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(outdir, "manifest_v2.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，结果：{res_path}")
    analyze(outdir)


if __name__ == "__main__":
    main()

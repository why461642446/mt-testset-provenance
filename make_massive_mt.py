#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_massive_mt.py —— 生成 C1（翻译膨胀）所需的机翻对照数据

把 MASSIVE 的 en-US 原句用商业机翻（DeepL）译成韩语/中文，写成
    massive_mt/ko-KR.jsonl
    massive_mt/zh-CN.jsonl
字段与 id / partition / intent / scenario 全部照抄原文，**只替换 utt**，
这样与官方人工本地化版本构成严格配对的受控对比。

配套实验：
    python experiment_v2.py --dataset massive --data mt --langs ko,zh \
        --models all --seeds 42,43,44 --outdir runs_v2

费用与配额
----------
DeepL 免费版 500,000 字符/月。实测需要：
    ko ≈ 16,521 句 × 16 字符 ≈ 264k 字符
    zh ≈ 16,521 句 × 10 字符 ≈ 165k 字符
    合计 ≈ 429k 字符  —— 刚好在免费额度内，但要一次跑完。
因此本脚本：
  * 先把所有待译文本落盘到 cache/*.jsonl，再调用 API；
  * 每译完一批立即写盘，中断后重跑**不会重复计费**；
  * 事先统计并打印字符数，让你先确认会不会超额。

用法
----
    # 1) 先只统计字符数，不调用 API
    python make_massive_mt.py --dry-run --count-only

    # 2) 用模拟翻译跑通全流程（不花钱、不需要 key）
    python make_massive_mt.py --dry-run

    # 3) 真正翻译（需要 DEEPL_API_KEY）
    export DEEPL_API_KEY=xxxxxxxx
    python make_massive_mt.py --targets ko,zh
"""

import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "massive", "1.1", "data", "en-US.jsonl")
OUTDIR = os.path.join(ROOT, "massive_mt")
CACHE = os.path.join(OUTDIR, "cache")

# MASSIVE 的 locale 代码 -> 机翻目标语言代码
DEEPL_TARGET = {"ko": "KO", "zh": "ZH",
                "de": "DE", "vi": "VI", "ja": "JA"}
# NLLB-200 的语言代码（开源模型，无需 API key，可复现性优于商业 API）
NLLB_CODE = {"ko": "kor_Hang", "zh": "zho_Hans",
             "de": "deu_Latn", "vi": "vie_Latn", "ja": "jpn_Jpan"}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LOCALE = {"ko": "ko-KR", "zh": "zh-CN",
          "de": "de-DE", "vi": "vi-VN", "ja": "ja-JP"}


def nllb_translate(texts, target, batch=16, on_result=None, model_name=None):
    """用开源 NLLB-200 翻译。

    选它的理由：
      * 不需要 API key（DeepL key 已失效）
      * 模型版本可钉死，审稿人可完全复现
      * 600M 蒸馏版在 CPU 上也能跑（2,974 句测试集约十几分钟）

    model_name=None 时使用模块默认的 NLLB_MODEL（600M）。传入更大的版本
    （如 facebook/nllb-200-3.3B）可生成"第二个 MT 系统"的对照语料。
    解码为**贪心**（不设 num_beams、不采样），因此给定同一模型版本可复现。

    on_result(start_index, batch_translations)
        每译完一批就回调一次，供调用方**立即落盘**。
        修复说明：早期版本在函数返回后才写缓存，导致"整个语言译完才算一次进度"，
        中途中断会丢失该语言的全部已完成翻译（实测丢失过约 60 分钟的韩语结果）。
        现在改为逐批回调，断点续跑才真正成立。
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = model_name or NLLB_MODEL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      载入 {model_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    # CUDA 上一律 fp16：3.3B 在 fp32 下要 13.2 GB 权重，T4（15.6 GB）会 OOM。
    # fp16 约 6.6 GB，留足激活空间。CPU 保持 fp32。
    if dev == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(dev).eval()
    nparam = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    print(f"      设备: {dev}   参数量: {nparam/1e6:.0f}M   权重精度: {dtype}", flush=True)
    if dev == "cuda":
        print(f"      显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    tgt_id = tok.convert_tokens_to_ids(NLLB_CODE[target])
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=128)
        batch_out = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(batch_out)
        if on_result is not None:
            on_result(i, batch_out)          # 立即回调落盘
        done = min(i + batch, len(texts))
        if done % (batch * 10) == 0 or done == len(texts):
            print(f"      {done}/{len(texts)} 已翻译", flush=True)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def load_en():
    recs = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def deepl_translate(texts, target, api_key, batch=40):
    """调用 DeepL REST API。返回与输入等长的译文列表。"""
    import urllib.parse
    import urllib.request

    # 免费版端点是 api-free.deepl.com，付费版是 api.deepl.com
    endpoint = ("https://api-free.deepl.com/v2/translate"
                if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate")
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        data = [("target_lang", target)] + [("text", t) for t in chunk]
        body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={
            "Authorization": "DeepL-Auth-Key " + api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read().decode("utf-8"))
                out.extend(x["text"] for x in j["translations"])
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"      [retry {attempt+1}/5] {type(e).__name__}: {e} "
                      f"-> {wait}s", flush=True)
                time.sleep(wait)
        done = min(i + batch, len(texts))
        print(f"      {done}/{len(texts)} 已翻译", flush=True)
    return out


def mock_translate(texts, target):
    """--dry-run 用：不调用任何 API，便于验证流程"""
    return [f"[{target}] {t}" for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="ko,zh")
    ap.add_argument("--engine", default="nllb", choices=["nllb", "deepl"],
                    help="机翻引擎。nllb（默认）= 开源模型，无需 key；"
                         "deepl = 商业 API，需要 DEEPL_API_KEY")
    ap.add_argument("--dry-run", action="store_true",
                    help="用模拟翻译，不调用任何模型/API")
    ap.add_argument("--count-only", action="store_true",
                    help="只统计字符数，不翻译")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--nllb-model", default=None,
                    help="覆盖默认的 NLLB 模型（用于生成第二个 MT 系统的对照语料）")
    ap.add_argument("--outdir", default=None,
                    help="覆盖输出目录（默认 massive_mt/；第二个系统写到 massive_mt2/）")
    args = ap.parse_args()

    global OUTDIR, CACHE
    if args.outdir:
        OUTDIR = args.outdir
        CACHE = os.path.join(OUTDIR, "cache")

    if not os.path.exists(SRC):
        raise SystemExit(f"[FAIL] 找不到 {SRC}\n"
                         f"       请先下载 amazon-massive-dataset-1.1.tar.gz 并解压")
    os.makedirs(CACHE, exist_ok=True)

    recs = load_en()
    wanted = set(args.splits.split(","))
    todo = [r for r in recs if r["partition"] in wanted]
    print("=" * 74)
    print("make_massive_mt.py —— 生成 C1 的机翻对照数据")
    print("=" * 74)
    print(f"源文件     : {SRC}")
    print(f"en-US 总句数: {len(recs)}   本次处理(train/dev/test): {len(todo)}")

    chars = sum(len(r["utt"]) for r in todo)
    print()
    print("字符数预估（DeepL 按字符计费）:")
    for t in args.targets.split(","):
        print(f"  {t}: {chars:,} 字符")
    print(f"  合计: {chars * len(args.targets.split(',')):,} 字符"
          f"   （免费额度 500,000/月）")
    print()

    if args.count_only:
        print("--count-only 已指定，到此结束。")
        return

    api_key = os.environ.get("DEEPL_API_KEY", "")
    if args.engine == "deepl" and not args.dry_run and not api_key:
        raise SystemExit("[FAIL] --engine deepl 需要 DEEPL_API_KEY。\n"
                         "       建议改用默认的 --engine nllb（开源模型，无需 key）。")

    for tgt in args.targets.split(","):
        cache_path = os.path.join(CACHE, f"en-US_to_{tgt}.jsonl")
        # ---- 断点续跑：读回已翻译的部分 ----
        done = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        done[d["id"]] = d["mt"]
            print(f"[{tgt}] 缓存已有 {len(done)} 条，将跳过")

        pending = [r for r in todo if r["id"] not in done]
        print(f"[{tgt}] 待翻译 {len(pending)} 条（引擎: "
              f"{'模拟' if args.dry_run else args.engine}）", flush=True)
        if pending:
            texts = [r["utt"] for r in pending]
            cache_fh = open(cache_path, "a", encoding="utf-8")

            def _persist(start, batch_out, _pending=pending, _fh=cache_fh, _done=done):
                """逐批落盘 —— 中断时最多损失一个批次，而不是一整个语言"""
                for k, mt in enumerate(batch_out):
                    r = _pending[start + k]
                    _fh.write(json.dumps({"id": r["id"], "mt": mt},
                                         ensure_ascii=False) + "\n")
                    _done[r["id"]] = mt
                _fh.flush()

            try:
                if args.dry_run:
                    translated = mock_translate(texts, tgt)
                    _persist(0, translated)
                elif args.engine == "nllb":
                    translated = nllb_translate(texts, tgt, on_result=_persist,
                                                model_name=args.nllb_model)
                else:
                    translated = deepl_translate(texts, DEEPL_TARGET[tgt], api_key)
                    _persist(0, translated)
            finally:
                cache_fh.close()

        # ---- 写出与 MASSIVE 同构的 jsonl ----
        out_path = os.path.join(OUTDIR, LOCALE[tgt] + ".jsonl")
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:                       # 保持原始行序，便于逐行对照
                if r["partition"] not in wanted:
                    continue
                if r["id"] not in done:
                    continue
                d = dict(r)                      # id/partition/intent/scenario 原样保留
                d["utt"] = done[r["id"]]         # 只替换句子
                # 血缘：必须记录真实引擎，否则语料元数据与论文不符
                if args.dry_run:
                    d["mt_source"] = "mock"
                elif args.engine == "nllb":
                    d["mt_source"] = args.nllb_model or NLLB_MODEL
                else:
                    d["mt_source"] = "deepl"
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        print(f"[{tgt}] 已写出 {out_path}  ({n} 行)")

    print()
    print("完成。接下来跑机翻对照组：")
    print("  python experiment_v2.py --dataset massive --data mt "
          "--langs ko,zh --models all --seeds 42,43,44 --outdir runs_v2")
    print("注意：必须与 localized 组使用相同的 --langs/--models/--seeds 才能配对比较。")


if __name__ == "__main__":
    main()

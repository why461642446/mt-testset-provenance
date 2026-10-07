# -*- coding: utf-8 -*-
"""生成 audit_zh_bert.ipynb（v2）：四项新检查。"""
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
SRC = open(os.path.join(SP, "audit_zh_bert.py"), encoding="utf-8").read()
MD5 = hashlib.md5(SRC.replace("\r\n", "\n").encode("utf-8")).hexdigest()
print("  audit_zh_bert.py md5 = %s" % MD5)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


HELLO = """# 中文零样本 BERT 的完整审计（v2）

## 要回答什么

论文 4.4 节报告：**`bert-base-chinese`（纯中文词表）在英文上微调后，中文零样本 65.78 ± 1.70**，
比有跨语言对齐的 mBERT（51.66）高 14 点。审稿人认为**最可能是流程问题**。

**v1 只证明了「结果是确定性的」** —— 但一个可重现的错误也会精确重现，所以这**不足以排除流程问题**。
v2 补上四项能真正区分的检查。

## 四项新检查

| # | 检查 | 若出现什么说明是流程问题 |
|---|---|---|
| ① | **记录每次加载实际打开的路径与 MD5** | 路径不是 `en-US.jsonl`，或 MD5 不符 |
| ② | **随机初始化对照**（同结构、不加载预训练权重） | 若它也能在中文上远高于随机 1.67% -> **数据/标签泄漏** |
| ③ | **逐意图拆解** 65.78% | 若集中在少数含数字/拉丁字符的意图 -> 有平凡解释 |
| ④ | **剥离数字与拉丁字符后重测** | 若崩塌 -> 模型在利用共享符号，不是中文语义 |

外加**逐句预测落盘**。

## 跑之前

1. **Settings → Accelerator: GPU T4**
2. **Settings → Internet: On**

## 预计耗时

| 阶段 | 预计 |
|---|---|
| 环境 + 脚本 + MASSIVE | 3 分钟 |
| **`bert-base-chinese` × 3 seeds**（含全部消融） | 约 50 分钟 |
| **随机初始化对照 × 3 seeds** | 约 50 分钟 |
| | **约 1.7 小时** |

**想快一半**：去掉第 4 格的 `--random-init` —— 但 ② 正是最能区分「泄漏」与「迁移」的那项，建议保留。
"""

CELLS = [md(HELLO), md("## 1. 环境自检")]

CELLS.append(code('''import os, sys, subprocess, torch
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, " cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu     ", p.name, " %.1f GB" % (p.total_memory/1e9))
else:
    print("!! 未检测到 GPU —— 请把 Accelerator 设为 GPU T4")
import transformers
print("transformers", transformers.__version__)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'''))

CELLS.append(md("## 2. 写入脚本 + MD5 校验"))
CELLS.append(code("import os\nos.makedirs('/kaggle/working/sci_paper', exist_ok=True)\nprint('ok')"))
CELLS.append(code("%%writefile /kaggle/working/sci_paper/audit_zh_bert.py\n" + SRC))
CELLS.append(code(
    "import hashlib, py_compile\n"
    "EXPECT = [('audit_zh_bert.py', '" + MD5 + "')]\n"
    "bad = []\n"
    "for name, want in EXPECT:\n"
    "    p = '/kaggle/working/sci_paper/' + name\n"
    "    got = hashlib.md5(open(p, 'rb').read().replace(b'\\r\\n', b'\\n')).hexdigest()\n"
    "    ok = (got == want)\n"
    "    print('%-24s %s  %s' % (name, 'OK ' if ok else '!! ', got))\n"
    "    if not ok:\n"
    "        bad.append(name)\n"
    "    else:\n"
    "        py_compile.compile(p, doraise=True)\n"
    "assert not bad, 'MD5 mismatch, stop: ' + str(bad)\n"
    "print()\n"
    "print('all embedded scripts verified.')\n"
))

CELLS.append(md("## 3. 下载 MASSIVE v1.1"))
CELLS.append(code('''import os, subprocess, glob, shutil

url  = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
work = "/kaggle/working"
D    = os.path.join(work, "massive", "1.1", "data")
tgz  = os.path.join(work, "massive-1.1.tar.gz")

if os.path.exists(os.path.join(D, "ko-KR.jsonl")):
    print("已存在，跳过下载")
else:
    if not os.path.exists(tgz) or os.path.getsize(tgz) < 10_000_000:
        print("下载中 ...")
        r = subprocess.run(["curl", "-L", "-o", tgz, url], capture_output=True, text=True)
        print("  curl 返回码 =", r.returncode)
    print("  压缩包大小 = %.1f MB" % (os.path.getsize(tgz) / 1e6))
    os.makedirs(os.path.join(work, "massive"), exist_ok=True)
    subprocess.run(["tar", "-xzf", tgz, "-C", os.path.join(work, "massive")], check=True)
    if not os.path.exists(os.path.join(D, "en-US.jsonl")):
        hits = glob.glob(os.path.join(work, "**", "en-US.jsonl"), recursive=True)
        if hits:
            src = os.path.dirname(hits[0])
            if not os.path.exists(D):
                os.makedirs(os.path.dirname(D), exist_ok=True)
                shutil.move(src, D)

ok = True
for loc in ["en-US", "ko-KR", "zh-CN"]:
    p = os.path.join(D, loc + ".jsonl")
    n = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
    ok = ok and n > 0
    print("  %s %-8s %6d 行" % ("OK " if n else "MISS", loc, n))
assert ok, "MASSIVE 未就位 —— 把本格输出发给作者"
print()
print("MASSIVE 就位。")
'''))

CELLS.append(md("## 4. 审计（核心格，约 1.7 小时）\n\n**只跑中文 BERT（约 50 分钟）**：去掉 `--random-init`。"))
CELLS.append(code('''!cd /kaggle/working/sci_paper && python -u audit_zh_bert.py \\
    --models bert-base-chinese \\
    --seeds 42,43,44 --random-init \\
    --outdir /kaggle/working/audit_zh_bert_v2 2>&1 | tail -80
'''))

CELLS.append(md("## 5. 结果速览与判读"))
CELLS.append(code('''import pandas as pd, os, json, glob
print("=== (1) 数据加载记录：实际打开的路径与 MD5 ===")
for f in glob.glob("/kaggle/working/**/data_load_log.json", recursive=True):
    for r in json.load(open(f, encoding="utf-8")):
        print("  %-4s %s" % (r["locale"], r["abs_path"]))
        print("       md5=%s  train=%d  CJK句占比=%.3f  first=%s"
              % (r["md5"], r["n_train"], r["cjk_frac"], r["first_train_utt"][:50]))
print()
print("=== (2)(3)(4) 审计结果 ===")
for f in glob.glob("/kaggle/working/**/audit_results.csv", recursive=True):
    df = pd.read_csv(f)
    print(); print("=" * 78); print(f, "(%d 行)" % len(df)); print("=" * 78)
    show = [c for c in ["model","random_init","seed","en_acc","ko_acc","zh_acc",
                        "zh_stripped_acc","zh_acc_with_ld","zh_acc_without_ld"] if c in df.columns]
    print(df[show].to_string(index=False))
    print()
    for (ck, ri), g in df.groupby(["model","random_init"]):
        print("  %s%s" % (ck, "  [RANDOM-INIT]" if ri else ""))
        for c in ["en_acc","ko_acc","zh_acc","zh_stripped_acc"]:
            if c in g.columns:
                v = g[c].dropna().values
                print("      %-20s %.2f +/- %.2f" % (c, v.mean(), v.std(ddof=1) if len(v)>1 else 0))
print()
print("  判读:")
print("    * en 应约 85%；随机初始化对照的 en 应远低于此")
print("    * 随机初始化若在 zh 上仍远高于 1.67% -> 数据/标签泄漏")
print("    * zh 剥离数字/拉丁后若崩塌 -> 模型在利用共享符号")
'''))
CELLS.append(code('''# (3) 逐意图拆解
import pandas as pd, json, glob
for f in glob.glob("/kaggle/working/**/audit_results.csv", recursive=True):
    df = pd.read_csv(f)
    if "zh_per_intent" not in df.columns: continue
    for _, r in df.iterrows():
        if r.get("random_init"): continue
        d = r["zh_per_intent"]
        if isinstance(d, str):
            try: d = json.loads(d)
            except Exception: continue
        s = pd.Series(d).sort_values(ascending=False)
        print("=== seed %s 中文逐意图准确率（前 8 / 后 5）===" % r["seed"])
        print(s.head(8).to_string()); print("  ..."); print(s.tail(5).to_string()); print()
'''))

CELLS.append(md("## 6. 打包"))
CELLS.append(code('''import os, shutil, glob
stage = "/kaggle/working/_pkg"
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage, exist_ok=True)
found = 0
for b in ["/kaggle/working/sci_paper", "/kaggle/working"]:
    for src in glob.glob(os.path.join(b, "audit_zh_bert_v2")):
        if not os.path.isdir(src): continue
        n = sum(len(f) for _, _, f in os.walk(src))
        print("FOUND  :", src, "->", n, "files")
        for f in sorted(os.listdir(src)):
            print("         ", f)
        dst = os.path.join(stage, "audit_zh_bert_v2")
        if not os.path.exists(dst):
            shutil.copytree(src, dst); found += 1
out = "/kaggle/working/audit_zh_bert_v2_results"
if os.path.exists(out + ".zip"): os.remove(out + ".zip")
shutil.make_archive(out, "zip", stage)
print()
print("packaged:", out + ".zip", "%.2f MB" % (os.path.getsize(out + ".zip")/1e6), "(%d dirs)" % found)
assert found > 0, "NO RESULT DIR FOUND -- 把本格输出发给作者"
print("Download it from the Output panel on the right.")
'''))

CELLS.append(md("## 7. 交回\n\n下载 `audit_zh_bert_v2_results.zip`，放到项目目录，告诉作者。"))

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(SP, "audit_zh_bert.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("  已写出 audit_zh_bert.ipynb (%d cells, %.0f KB)" % (len(CELLS), os.path.getsize(out) / 1024))

txt = "".join("".join(c["source"]) for c in CELLS)
for n, k in [("内嵌 MD5", MD5 in txt), ("--random-init", "--random-init" in txt),
             ("路径+MD5 记录", "data_load_log.json" in txt),
             ("剥离消融", "zh_stripped_acc" in txt),
             ("逐意图", "zh_per_intent" in txt),
             ("解压路径已修", 'os.path.join(work, "massive")' in txt),
             ("打包断言", "NO RESULT DIR FOUND" in txt)]:
    print("    %-18s %s" % (n, "OK" if k else "!!"))

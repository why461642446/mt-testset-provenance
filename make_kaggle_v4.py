# -*- coding: utf-8 -*-
"""生成两个 Kaggle notebook：

  v4a —— 预训练模型（mBERT / XLM-R）的字符级 2x2
  v4b —— 第二个 MT 系统（NLLB-3.3B）的 MT->MT2 单元格

设计要点：
  * 脚本内嵌（%%writefile），无需上传
  * 内嵌后用 MD5 硬校验，内容不符立即报错
  * MASSIVE 自动下载；MT 语料在 notebook 内用同一模型重新生成（贪心解码，可复现）
  * 每个命令独立成格，中断后可从此格继续（脚本自身按 run_id 断点续跑）
"""
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"

FILES = {
    "experiment_v2.py": open(os.path.join(SP, "experiment_v2.py"), encoding="utf-8").read(),
    "analyze_results.py": open(os.path.join(SP, "analyze_results.py"), encoding="utf-8").read(),
    "make_massive_mt.py": open(os.path.join(SP, "make_massive_mt.py"), encoding="utf-8").read(),
}
MD5 = {k: hashlib.md5(v.replace("\r\n", "\n").encode("utf-8")).hexdigest()
       for k, v in FILES.items()}
for k, v in MD5.items():
    print("  %-24s md5=%s" % (k, v))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def writefile(name, body):
    return code("%%%%writefile /kaggle/working/sci_paper/%s\n%s" % (name, body))


SETUP_MD = """## 2. 建目录 + 写入脚本（内嵌，无需上传）

下面的格子把三个脚本写到 `/kaggle/working/sci_paper/`，随后一格用 MD5 校验内容是否完整。
**如果 MD5 校验失败，不要继续** —— 说明单元格内容被改写或截断。
"""

MASSIVE_MD = """## 3. 下载 MASSIVE v1.1

约 40 MB。只需 en-US / ko-KR / zh-CN 三个 locale，但打包下载最省事。
"""

MTGEN_MD = """## 4. 在 notebook 内重建机翻语料

本 notebook 不依赖外部上传：用与论文相同的模型（`facebook/nllb-200-distilled-600M`）
和**贪心解码**重新生成 `massive_mt/`。同一模型版本下结果可复现。

预计 30–45 分钟（T4）。**按语言分别成格**，中断后重跑该格会从缓存继续。
"""


def env_check():
    return code("""import os, sys, subprocess, torch, platform
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, " cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu     ", p.name, " %.1f GB" % (p.total_memory/1e9))
else:
    print("!! 未检测到 GPU —— 请确认 Settings 里 Accelerator 选的是 GPU")
print("disk    ", subprocess.run(["df","-h","/kaggle/working"],capture_output=True,text=True).stdout.strip().split("\\n")[-1])
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
""")


def write_scripts_cell():
    parts = ["import os\nos.makedirs('/kaggle/working/sci_paper', exist_ok=True)\nprint('ok')\n"]
    return code("".join(parts))


def md5_cell(names):
    checks = ",\n    ".join('("%s", "%s")' % (n, MD5[n]) for n in names)
    body = (
        "import hashlib, py_compile, os\n"
        "EXPECT = [\n    " + checks + "\n]\n"
        "bad = []\n"
        "for name, want in EXPECT:\n"
        "    p = '/kaggle/working/sci_paper/' + name\n"
        "    # 归一化换行后再算，避免 CRLF/LF 造成假失败\n"
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
    )
    return code(body)


def massive_cell():
    return code("""import os, subprocess, glob, shutil

url  = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
work = "/kaggle/working"
D    = os.path.join(work, "massive", "1.1", "data")
tgz  = os.path.join(work, "massive-1.1.tar.gz")

if os.path.exists(os.path.join(D, "ko-KR.jsonl")):
    print("已存在，跳过下载")
else:
    # 1) 下载（带错误检查；不要用 curl -s 静默模式）
    if not os.path.exists(tgz) or os.path.getsize(tgz) < 10_000_000:
        print("下载中 ...")
        r = subprocess.run(["curl", "-L", "-o", tgz, url], capture_output=True, text=True)
        print("  curl 返回码 =", r.returncode)
        if r.stderr.strip():
            print("  stderr:", r.stderr.strip()[-400:])
    print("  压缩包大小 = %.1f MB" % (os.path.getsize(tgz) / 1e6))

    # 2) 先看内部结构再解压 —— 压缩包顶层是 1.1/，必须解到 massive/ 里
    lst = subprocess.run(["tar", "-tzf", tgz], capture_output=True, text=True).stdout.split("\\n")
    tops = sorted({l.split("/")[0] for l in lst if l.strip()})
    print("  压缩包顶层:", tops[:5])
    print("  其中的 jsonl:", [l for l in lst if l.endswith("en-US.jsonl")][:2])

    os.makedirs(os.path.join(work, "massive"), exist_ok=True)
    subprocess.run(["tar", "-xzf", tgz, "-C", os.path.join(work, "massive")], check=True)

    # 3) 不管解到哪，归位到 massive/1.1/data
    if not os.path.exists(os.path.join(D, "en-US.jsonl")):
        hits = glob.glob(os.path.join(work, "**", "en-US.jsonl"), recursive=True)
        print("  解压后找到:", hits[:3])
        if hits:
            s = os.path.dirname(hits[0])
            if not os.path.exists(D):
                os.makedirs(os.path.dirname(D), exist_ok=True)
                shutil.move(s, D)
                print("  已移动到", D)

# 4) 确认
ok = True
for loc in ["en-US", "ko-KR", "zh-CN"]:
    q = os.path.join(D, loc + ".jsonl")
    n = sum(1 for _ in open(q, encoding="utf-8")) if os.path.exists(q) else 0
    ok = ok and n > 0
    print("  %s %-8s %6d 行" % ("OK " if n else "MISS", loc, n))
assert ok, "MASSIVE 未就位 —— 请把本格输出发给作者"
print()
print("MASSIVE 就位。")
""")


def mt_gen(tgt):
    return code("""!cd /kaggle/working/sci_paper && python make_massive_mt.py --targets %s --splits train,dev,test 2>&1 | tail -20
!wc -l /kaggle/working/massive_mt/%s.jsonl
""" % (tgt, "ko-KR" if tgt == "ko" else "zh-CN"))


def run_cell(args, note):
    return code("""# %s
!cd /kaggle/working/sci_paper && python experiment_v2.py %s 2>&1 | tail -25
""" % (note, args))


def pkg_cell(dirs, zipname):
    """注意：experiment_v2.py 的 outdir 相对 HERE（脚本所在目录）解析，
    即 /kaggle/working/sci_paper/<outdir>，不是 /kaggle/working/<outdir>。
    这里两个位置都找，并打印诊断。"""
    d = ", ".join('"%s"' % x for x in dirs)
    body = (
        "import os, shutil, glob, pandas as pd\n"
        "DIRS = [" + d + "]\n"
        "BASES = ['/kaggle/working/sci_paper', '/kaggle/working']\n"
        "stage = '/kaggle/working/_pkg'\n"
        "shutil.rmtree(stage, ignore_errors=True)\n"
        "os.makedirs(stage, exist_ok=True)\n"
        "found = 0\n"
        "for name in DIRS:\n"
        "    src = None\n"
        "    for b in BASES:\n"
        "        cand = os.path.join(b, name)\n"
        "        if os.path.exists(os.path.join(cand, 'results_v2.csv')):\n"
        "            src = cand\n"
        "            break\n"
        "    if src is None:\n"
        "        print('MISSING:', name, '-- 搜索过', [os.path.join(b, name) for b in BASES])\n"
        "        continue\n"
        "    n = sum(len(f) for _, _, f in os.walk(src))\n"
        "    print('FOUND  :', src, '->', n, 'files')\n"
        "    df = pd.read_csv(os.path.join(src, 'results_v2.csv'))\n"
        "    print('         results_v2.csv: %d rows' % len(df))\n"
        "    dst = os.path.join(stage, name)\n"
        "    shutil.copytree(src, dst)\n"
        "    found += 1\n"
        "out = '/kaggle/working/" + zipname + "'\n"
        "if os.path.exists(out + '.zip'):\n"
        "    os.remove(out + '.zip')\n"
        "shutil.make_archive(out, 'zip', stage)\n"
        "sz = os.path.getsize(out + '.zip') / 1e6\n"
        "print()\n"
        "print('packaged:', out + '.zip', '%.1f MB' % sz, '(%d dirs)' % found)\n"
        "assert found > 0, 'NO RESULT DIR FOUND -- 把本格输出发给作者'\n"
        "print('Download it from the Output panel on the right.')\n"
    )
    return code(body)


# ---------------------------------------------------------------- Notebook A
def build_model_nb(model, tag, est_hours, mdesc):
    """一个模型一个 notebook —— 实测 mBERT 877 s / XLM-R 946 s per run，
    两个模型合计约 12 小时，超过 Kaggle 单次 12 小时上限，必须拆开。"""
    cells = [
        md("""# Kaggle 执行手册 %s —— %s 的字符级 2x2

## 补什么

论文目前只有**非预训练**模型（TextCNN / 掩码 BiLSTM）的 2x2。审稿人最可能追问：
结论是否只对脆弱的词级模型成立？本 notebook 在**字符级分词**下补上
**%s** 的完整 2x2（四个格子、两个语言、三个种子）。

## 跑之前

1. 右侧 **Settings → Accelerator** 选 **GPU T4 x1**
2. **Settings → Internet** 必须打开（要下 MASSIVE、HF 模型）
3. 依次运行全部格子；**每格可独立重跑**，脚本按 run_id 断点续跑

## 关键：为什么拆成两个 notebook

实测单次训练时间（T4）：**mBERT 877 s、XLM-R 946 s**。
48 个 run 合计约 **12 小时**，加上建语料会**超过 Kaggle 单次 12 小时上限**。
因此每个 notebook 只跑一个模型。**本 notebook 只跑 %s。**

## 预计耗时（本 notebook）

| 阶段 | 内容 | run 数 | 预计 |
|---|---|---|---|
| 1 | 环境 + 脚本 + MASSIVE | — | 3 分钟 |
| 2 | 重建机翻语料（600M，贪心） | — | 30–45 分钟 |
| 3 | %s 四格 | 24 | **约 %s 小时** |
| | **合计** | **24** | **约 %s 小时** |

**12 小时上限内有充足余量。** 中途中断就重跑相应格子。

## 顺序建议

**先跑 L->L 与 L->MT 两格** —— 它们支撑论文的头条结论；
再跑 MT->MT 与 MT->L（支撑"夸大"结论）。四格都跑完才算完整 2x2。
""" % (tag, model, model, model, model, est_hours, mdesc)),
        md("## 1. 环境自检"),
        env_check(),
        md(SETUP_MD),
        write_scripts_cell(),
        writefile("experiment_v2.py", FILES["experiment_v2.py"]),
        writefile("analyze_results.py", FILES["analyze_results.py"]),
        writefile("make_massive_mt.py", FILES["make_massive_mt.py"]),
        md("### 校验嵌入的脚本"),
        md5_cell(["experiment_v2.py", "analyze_results.py", "make_massive_mt.py"]),
        md("""## 3. 下载 MASSIVE v1.1

约 40 MB。只需 en-US / ko-KR / zh-CN 三个 locale，但打包下载最省事。
"""),
        massive_cell(),
        md(MTGEN_MD),
        mt_gen("ko"),
        mt_gen("zh"),
        code("""!cd /kaggle/working/sci_paper && python experiment_v2.py --dataset massive --data mt --langs ko,zh --models TextCNN --tokenizer char --seeds 42 --limit 60 --outdir /kaggle/working/_smoke 2>&1 | tail -8
print("\\n冒烟测试：能跑通即可，准确率无意义。")
"""),
        md("""## 5. %s 的字符级 2x2

四个格子分别成格，便于中断后单独重跑。
`--data mt` = 机翻训练 + 机翻测试；`--data mt-test` = 本地化训练 + 机翻测试；
`--data mt-train` = 机翻训练 + 本地化测试；缺省 = 本地化训练 + 本地化测试。
""" % model),
        run_cell("--dataset massive --langs ko,zh --models %s --tokenizer char --seeds 42,43,44 --save-predictions --outdir %s"
                 % (model, tag), "%s / L->L：本地化训练 -> 本地化测试" % model),
        run_cell("--dataset massive --data mt-test --langs ko,zh --models %s --tokenizer char --seeds 42,43,44 --save-predictions --outdir %s"
                 % (model, tag), "%s / L->MT：本地化训练 -> 机翻测试（头条结论）" % model),
        run_cell("--dataset massive --data mt --langs ko,zh --models %s --tokenizer char --seeds 42,43,44 --save-predictions --outdir %s"
                 % (model, tag), "%s / MT->MT：机翻训练 -> 机翻测试" % model),
        run_cell("--dataset massive --data mt-train --langs ko,zh --models %s --tokenizer char --seeds 42,43,44 --save-predictions --outdir %s"
                 % (model, tag), "%s / MT->L：机翻训练 -> 本地化测试" % model),
        md("## 6. 汇总与打包"),
        code("""import pandas as pd, os
p = "/kaggle/working/%s/results_v2.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    print("共 %%d 行\\n" %% len(d))
    g = d.groupby(["data_variant","test_lang","model"]).accuracy.agg(["count","mean"])
    g["mean"] = (g["mean"]*100).round(2)
    print(g.to_string())
else:
    print("尚无结果")
""" % tag),
        pkg_cell([tag], tag + "_results"),
        md("""## 7. 交回

下载 `%s_results.zip` 后放到本机项目目录，告诉我路径。
我会用 `runs_loader` 并入，并按固定评测子集重算这四格。
""" % tag),
    ]
    return cells


cellsA = build_model_nb("mBERT", "runs_v4a", "5.8", "约 6.5–7")

cellsC = build_model_nb("XLMR", "runs_v4c", "6.3", "约 7")

# ---------------------------------------------------------------- Notebook B
cellsB = [
    md("""# Kaggle 执行手册 v4b —— 第二个 MT 系统（MT -> MT2）

## 补什么

论文的"夸大 7.5–12.3 点"依赖**单一** MT 系统。若训练与测试出自同一系统，
其系统性偏差会被模型学到，**可能高估**夸大量。本 notebook 用
`facebook/nllb-200-3.3B` 另译一份测试集，得到 **MT(600M) -> MT2(3.3B)** 单元格，
与 MT->MT、MT->L 对比。

## 跑之前

1. Settings → **Accelerator: GPU T4 x1**
2. Settings → **Internet: On**
3. 逐格运行

## 预计耗时

| 阶段 | 内容 | 预计 |
|---|---|---|
| 1 | 环境 + 脚本 + MASSIVE | 3 分钟 |
| 2 | 重建 600M 机翻语料（作为训练集） | 30–45 分钟 |
| 3 | **3.3B 另译测试集** | 40–60 分钟 |
| 4 | MT -> MT2 评测（2 语言 × 2 模型 × 3 种子 = 12 run） | 约 1.5 小时 |
| | **合计** | **约 3 小时** |

## 显存说明

3.3B 在 fp16 下约 6.6 GB，T4 的 16 GB 够用。加载时用 `torch_dtype=float16`。
"""),
    md("## 1. 环境自检"),
    env_check(),
    md(SETUP_MD),
    write_scripts_cell(),
    writefile("experiment_v2.py", FILES["experiment_v2.py"]),
    writefile("analyze_results.py", FILES["analyze_results.py"]),
    writefile("make_massive_mt.py", FILES["make_massive_mt.py"]),
    md("### 校验嵌入的脚本"),
    md5_cell(["experiment_v2.py", "analyze_results.py", "make_massive_mt.py"]),
    md(MASSIVE_MD),
    massive_cell(),
    md("""## 4. 重建 600M 机翻语料（**train + test**）

两个划分都要：
* **train** —— 所有实验的训练集
* **test** —— `--data mt`（同系统 MT→MT）对照的测试集**必须**有它，
  否则与 MT→MT2 的对比就缺了一半

（dev 不需要，跳过可省约 5 分钟。）
"""),
    code("""!cd /kaggle/working/sci_paper && python make_massive_mt.py --targets ko,zh --splits train,test 2>&1 | tail -20
!wc -l /kaggle/working/massive_mt/ko-KR.jsonl /kaggle/working/massive_mt/zh-CN.jsonl
"""),
    md("""## 5. 用 3.3B 另译**测试集** → MT2

输出到 `massive_mt2/`，只含 test 划分。注意：
* 只翻 test（2,974 句 × 2 语言），省时间
* `make_massive_mt.py` 写出的 jsonl 会包含全部三个划分的记录，
  但**只有 test 的 `utt` 是 3.3B 译文**；下面一格会把其他划分删掉，
  以免与 600M 训练集混淆。
"""),
    code("""!cd /kaggle/working/sci_paper && python make_massive_mt.py --targets ko,zh --splits test --nllb-model facebook/nllb-200-3.3B --outdir /kaggle/working/massive_mt2 2>&1 | tail -25
"""),
    code("""# 只保留 test 划分。**容忍缺失文件** —— 若 3.3B 翻译失败，
# 这里只报警告，不让整个 notebook 崩掉。
import json, os, glob

D2 = "/kaggle/working/massive_mt2"
print("massive_mt2 目录内容:", glob.glob(D2 + "/*") if os.path.isdir(D2) else "目录不存在")
print()

missing = []
for loc in ["ko-KR", "zh-CN"]:
    p = os.path.join(D2, "%s.jsonl" % loc)
    if not os.path.exists(p):
        print("!! %-8s 不存在 —— 3.3B 翻译可能失败/OOM" % loc)
        missing.append(loc)
        continue
    keep = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["partition"] == "test":
            keep.append(r)
    if not keep:
        print("!! %-8s 存在但没有 test 记录" % loc)
        missing.append(loc)
        continue
    with open(p, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    print("OK %-8s 保留 %d 条（全部为 test）  例: %s"
          % (loc, len(keep), keep[0]["utt"][:50]))

print()
if missing:
    print("=" * 70)
    print("警告：%s 缺失。**后面的 MT->MT2 实验格会失败。**" % missing)
    print("请检查上一格的 3.3B 翻译输出（是否 CUDA out of memory）。")
    print("=" * 70)
else:
    print("massive_mt2 就绪，可以继续。")
"""),
    md("""## 6. 任务 B —— MT -> MT2

`--data mt2`：用**系统 A（600M）的机翻训练集**训练，用**系统 B（3.3B）的机翻测试集**评测。
`massive_mt2/` 在上一格已生成。

同时跑两个对照，便于直接读出"同系统 vs 跨系统"的差别：
* `--data mt`（MT -> MT，同系统）与 `--data mt-train`（MT -> L，本地化测试）
"""),
    run_cell("--dataset massive --data mt2 --langs ko,zh --models TextCNN,BiLSTM --bilstm-mask --tokenizer char --seeds 42,43,44 --save-predictions --outdir runs_v4b",
             "MT -> MT2：系统 A 训练，系统 B 测试（本 notebook 的核心单元格）"),
    run_cell("--dataset massive --data mt --langs ko,zh --models TextCNN,BiLSTM --bilstm-mask --tokenizer char --seeds 42,43,44 --save-predictions --outdir runs_v4b",
             "对照：MT -> MT（同一系统，应与本机已有结果一致）"),
    run_cell("--dataset massive --data mt-train --langs ko,zh --models TextCNN,BiLSTM --bilstm-mask --tokenizer char --seeds 42,43,44 --save-predictions --outdir runs_v4b",
             "对照：MT -> L（本地化测试）"),
    md("## 7. 汇总与打包"),
    code("""import pandas as pd, os
# 注意：experiment_v2.py 的 outdir 相对脚本目录解析
p = "/kaggle/working/sci_paper/runs_v4b/results_v2.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    g = d.groupby(["data_variant","test_lang","model"]).accuracy.agg(["count","mean"])
    g["mean"] = (g["mean"]*100).round(2)
    print(g.to_string())
"""),
    pkg_cell(["runs_v4b", "massive_mt2"], "runs_v4b_results"),
    md("""## 8. 交回

下载 `runs_v4b_results.zip` 后放到本机项目目录，告诉我路径。

**读法**：把 MT -> MT2 与 MT -> MT、MT -> L 三者并列。
* 若 MT -> MT2 明显低于 MT -> MT，说明同一 MT 系统的偏差**确实**抬高了夸大量，
  论文的 7.5–12.3 点应下调，并写明这是上界。
* 若两者接近，说明夸大量对 MT 系统不敏感，结论更稳。
"""),
]


def dump(cells, path):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  已写出 %s  (%d 个 cell, %.0f KB)" % (os.path.basename(path), len(cells),
                                                os.path.getsize(path) / 1024))


dump(cellsA, os.path.join(SP, "kaggle_run_v4a.ipynb"))
dump(cellsC, os.path.join(SP, "kaggle_run_v4c.ipynb"))
dump(cellsB, os.path.join(SP, "kaggle_run_v4b.ipynb"))

# -*- coding: utf-8 -*-
"""docx -> Markdown 反转，保留标题层级、表格、加粗/斜体/等宽。

用途：把用户改过的 docx 变成可编辑的工作底本，避免后续改写覆盖用户的改动。
"""
import os
import sys

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.stdout.reconfigure(encoding="utf-8")
SP = r"D:\yanjiubaogaoxiangmu\sci_paper"
SRC = SP + r"\Manuscript_What_a_Benchmark_Number_Actually_Measures_final_revised_v2.docx"
DST = SP + r"\manuscript_draft.md"

doc = Document(SRC)

HEAD = {0: "# ", 1: "## ", 2: "### ", 3: "#### "}


def runs_md(par):
    out = []
    for r in par.runs:
        t = r.text
        if not t:
            continue
        if r.bold and r.italic:
            out.append("***%s***" % t)
        elif r.bold:
            out.append("**%s**" % t)
        elif r.italic:
            out.append("*%s*" % t)
        elif r.font.name and "Consol" in str(r.font.name):
            out.append("`%s`" % t)
        else:
            out.append(t)
    s = "".join(out)
    # 合并相邻重复标记
    s = s.replace("****", "")
    return s


def iter_body(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


lines = []
first_head = True
for item in iter_body(doc):
    if isinstance(item, Paragraph):
        t = item.text.strip()
        if not t:
            continue
        style = item.style.name if item.style else ""
        m = None
        if style.startswith("Title"):
            m = 0
        elif style.startswith("Heading"):
            try:
                m = int(style.split()[-1])
            except Exception:
                m = 2
        if m is not None:
            lvl = min(m, 3)
            if lvl == 0 and not first_head:
                lvl = 1
            first_head = False
            lines.append("")
            lines.append(HEAD[lvl] + runs_md(item))
            lines.append("")
        else:
            lines.append(runs_md(item))
            lines.append("")
    else:
        rows = []
        for r in item.rows:
            cells = [c.text.strip().replace("\n", " ") for c in r.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            ncol = rows[0].count("|") - 1
            lines.append(rows[0])
            lines.append("|" + " --- |" * ncol)
            lines.extend(rows[1:])
            lines.append("")

out = "\n".join(lines)
out = "\n".join(l for l in out.split("\n"))
out = out.replace("\n\n\n", "\n\n")
open(DST, "w", encoding="utf-8").write(out)

print("已写出 %s" % os.path.basename(DST))
print("  行数 %d   字符 %d   表格 %d" % (len(out.split("\n")), len(out), len(doc.tables)))

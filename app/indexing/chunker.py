"""语义切块。

切块单位是「一个配置项的完整描述」或「一个业务语义单元」，不是固定窗口。

这个选择对本项目是决定性的。配置项的定义、取值范围、依赖关系、互斥关系、
踩坑提醒在文档里是连续的一段，一旦被固定窗口从中间切开，
"免费阅读模式与付费章节互斥"这句话就可能和"免费阅读模式"这个主语分到两个 chunk，
召回到哪一半都是残缺的。所以这里按 markdown 的标题层级切，
标题本身就是作者划定的语义边界。

超长小节才回落到滑动窗口，且窗口切分时带重叠，尽量不切断句子。
每个 chunk 会扫一遍配置项词表打上 linked_item_keys 标签，
这是后面「图谱实体 → 文本段」桥接的依据。
"""
from __future__ import annotations

import re
from pathlib import Path

from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.foundation.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT_END = re.compile(r"(?<=[。！？；\n])")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按标题切成 (section_path, body)。section_path 形如 '三、全勤奖（full_attendance_bonus）'。"""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    stack: list[str] = []
    current: list[str] = []
    current_path = ""

    for line in lines:
        m = _HEADING.match(line)
        if m:
            if current:
                sections.append((current_path, current))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            current_path = " / ".join(stack)
            current = []
        else:
            current.append(line)
    if current:
        sections.append((current_path, current))

    return [(path, "\n".join(body).strip()) for path, body in sections if "".join(body).strip()]


def _split_long(body: str) -> list[str]:
    """超长小节的滑动窗口切分，按句边界，带重叠。"""
    target = settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars
    sents = [s for s in _SENT_END.split(body) if s.strip()]
    out: list[str] = []
    buf = ""
    for s in sents:
        if len(buf) + len(s) <= target:
            buf += s
            continue
        if buf:
            out.append(buf.strip())
            # 重叠：把上一块的尾巴接到下一块开头，避免跨块的指代断裂
            buf = buf[-overlap:] + s if overlap else s
        else:
            # 单句就超长，硬切
            for i in range(0, len(s), target):
                out.append(s[i : i + target])
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _split_by_paragraph_group(body: str) -> list[str]:
    """小节内按空行分段，再把相邻小段合并到目标长度。

    合并是必要的：文档里大量存在"标题 + 一句话 + 表格"的结构，
    每段单独成 chunk 会产生一堆 20 字的碎片，向量表示极不稳定。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 1 <= settings.chunk_target_chars:
            buf += "\n" + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _link_items(text: str) -> list[str]:
    """扫描 chunk 里出现的配置项。

    用词表匹配而不是 LLM：配置项名称是封闭词表且来自配置平台元数据，
    这件事上词表匹配的准确率是 100%，没有让 LLM 插手的理由。
    """
    reg = get_registry()
    low = text.lower()
    hits: list[str] = []
    for key, item in reg.items.items():
        surfaces = [key, item.name, *item.aliases]
        if any(s.lower() in low for s in surfaces if len(s) >= 2):
            hits.append(key)
    return hits


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_id = path.stem
    # 首个一级标题作为文档标题
    m = re.search(r"^#\s+(.*)$", raw, re.MULTILINE)
    doc_title = m.group(1).strip() if m else doc_id

    chunks: list[Chunk] = []
    for section_path, body in _split_sections(raw):
        # 去掉引言里的元信息行（维护人、更新时间），它们对召回是纯噪声
        body = re.sub(r"^>\s*(维护人|最近更新|适用范围|适用对象|说明)：.*$", "",
                      body, flags=re.MULTILINE).strip()
        if not body:
            continue

        pieces = (
            _split_long(body)
            if len(body) > settings.chunk_max_chars
            else _split_by_paragraph_group(body)
        )
        strategy = "sliding" if len(body) > settings.chunk_max_chars else "semantic"

        for i, piece in enumerate(pieces):
            if len(piece.strip()) < 15:
                continue
            # 把标题拼回正文：小节标题往往是唯一出现配置项全称的地方
            # （正文里会用"它""这个开关"指代），不拼进去会直接影响召回
            text = f"{section_path}\n{piece}" if section_path else piece
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks):03d}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=section_path,
                    text=text,
                    linked_item_keys=_link_items(text),
                    # 标题里出现的配置项 = 这一段的主题。FAQ 类文档标题是问句，
                    # 主题词同样出现在标题里，所以这个规则对两类文档都成立。
                    primary_item_keys=_link_items(section_path),
                    char_len=len(text),
                    strategy=strategy,
                )
            )
    return chunks


def chunk_all(docs_dir: Path | None = None) -> list[Chunk]:
    docs_dir = docs_dir or settings.raw_docs_dir
    out: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        out.extend(chunk_document(path))
    return out

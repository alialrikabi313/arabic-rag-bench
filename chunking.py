# -*- coding: utf-8 -*-
"""طرق التقطيع الثلاث المقارَنة: ثابت، فقرات، دلالي. تُستخدم في المرحلتين 2 و4."""

import re

import numpy as np

from config import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_WORDS,
    LIST_ARTICLE_MAX_SHORT_RATIO,
    LIST_SHORT_SENT_WORDS,
    MAX_CHUNK_WORDS,
    MIN_CHUNK_WORDS,
    SEMANTIC_BREAK_PERCENTILE,
)

# نقطة وعلامتا الاستفهام والتعجّب (لاتينية وعربية) أو سطر جديد
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")
# عناوين ويكيبيديا النصّية: == عنوان ==
_HEADING_RE = re.compile(r"^=+\s*.+?\s*=+$")


def split_sentences(text: str) -> list[str]:
    parts = (s.strip() for s in _SENT_SPLIT_RE.split(text))
    return [s for s in parts if s and not _HEADING_RE.match(s)]


def is_list_article(sentences: list[str]) -> bool:
    short = sum(len(s.split()) <= LIST_SHORT_SENT_WORDS for s in sentences)
    return short / len(sentences) > LIST_ARTICLE_MAX_SHORT_RATIO


def fixed_chunks(
    text: str, size: int = FIXED_CHUNK_WORDS, overlap: int = FIXED_CHUNK_OVERLAP
) -> list[str]:
    words = " ".join(split_sentences(text)).split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def paragraph_chunks(text: str) -> list[str]:
    # الفقرة وحدة طبيعية؛ تُدمج الصغيرة حتى MAX، والفقرة الأطول من MAX تبقى كما هي
    paras = [p.strip() for p in text.split("\n") if p.strip() and not _HEADING_RE.match(p.strip())]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for p in paras:
        w = len(p.split())
        if cur and cur_words + w > MAX_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def semantic_chunks(sentences: list[str], sent_vecs: np.ndarray) -> list[str]:
    if len(sentences) == 1:
        return sentences
    # التضمينات مطبَّعة فالجداء الداخلي هو cosine
    sims = (sent_vecs[:-1] * sent_vecs[1:]).sum(axis=1)
    cut_below = np.percentile(sims, SEMANTIC_BREAK_PERCENTILE)

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for i, sent in enumerate(sentences):
        w = len(sent.split())
        # السقف صارم: نُفرغ قبل إضافة جملة تتجاوزه (الجملة الأطول من السقف وحدها تبقى مقطعاً)
        if cur and cur_words + w > MAX_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(sent)
        cur_words += w
        is_last = i == len(sentences) - 1
        if is_last:
            chunks.append(" ".join(cur))
        elif sims[i] <= cut_below and cur_words >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0

    # ذيل قصير يُدمج بالسابق فقط إن لم يتجاوز السقف؛ وإلا يبقى مستقلاً
    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_CHUNK_WORDS:
        if len(chunks[-2].split()) + len(chunks[-1].split()) <= MAX_CHUNK_WORDS:
            tail = chunks.pop()
            chunks[-1] += " " + tail
    return chunks

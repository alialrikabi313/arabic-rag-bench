# -*- coding: utf-8 -*-
"""اختبارات التقطيع. كل حالة صادفناها فعلاً في عيّنة ويكيبيديا (المرحلتان 1 و2)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import fixed_chunks, is_list_article, paragraph_chunks, semantic_chunks, split_sentences
from config import MAX_CHUNK_WORDS, SAMPLE_JSONL


def _article(title: str) -> str:
    # النصوص الخام خارج الريبو؛ بدون العيّنة المحلية تُتخطّى الاختبارات
    if not SAMPLE_JSONL.exists():
        pytest.skip("عيّنة ويكيبيديا غير موجودة (شغّل 01_fetch_sample.py)")
    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["title"] == title:
                return row["text"]
    pytest.skip(f"المقالة غير موجودة في العيّنة المحلية: {title}")


def test_obituary_list_article_is_flagged():
    # «وفيات 2014»: 94% من جمله ≤ 8 كلمات (اسم، عمر، مهنة) فلا تصلح للاسترجاع
    assert is_list_article(split_sentences(_article("وفيات 2014")))


def test_prose_article_is_not_flagged():
    # «ثقافة الاغتصاب»: نثر طويل؛ لا يجوز أن تسقطه عتبة القوائم
    assert not is_list_article(split_sentences(_article("ثقافة الاغتصاب")))


def test_long_sentences_never_exceed_cap():
    # كانت هذه المقالة تنتج مقاطع حتى 340 كلمة لأن السقف يُفحص بعد إضافة الجملة
    sents = split_sentences(_article("ثقافة الاغتصاب"))
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(sents), 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    longest_single = max(len(s.split()) for s in sents)
    for chunk in semantic_chunks(sents, vecs):
        assert len(chunk.split()) <= max(MAX_CHUNK_WORDS, longest_single)


def test_headings_are_not_chunk_text():
    # عناوين ويكيبيديا النصّية «== مراجع ==» كانت تدخل المقاطع
    text = _article("ثقافة الاغتصاب")
    for chunk in fixed_chunks(text) + paragraph_chunks(text):
        assert "== " not in chunk


def test_fixed_chunks_overlap_covers_whole_article():
    text = _article("ثقافة الاغتصاب")
    words = " ".join(split_sentences(text)).split()
    chunks = fixed_chunks(text)
    assert chunks[0].split()[0] == words[0]
    assert chunks[-1].split()[-1] == words[-1]

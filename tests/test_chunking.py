# -*- coding: utf-8 -*-
"""اختبارات التقطيع. كل حالة صادفناها فعلاً في عيّنة ويكيبيديا (المرحلتان 1 و2)."""

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import fixed_chunks, is_list_article, paragraph_chunks, semantic_chunks, split_sentences
from config import MAX_CHUNK_WORDS


_DATA = Path(__file__).resolve().parent / "data"
_FILES = {
    "وفيات 2014": "wiki_obituaries_2014.txt",
    "ثقافة الاغتصاب": "wiki_rape_culture_excerpt.txt",
}


def _article(title: str) -> str:
    # مقتطفات حقيقية مرفقة بالريبو (انظر tests/data/ATTRIBUTION.md) كي تعمل الاختبارات في أي بيئة
    return (_DATA / _FILES[title]).read_text(encoding="utf-8")


def test_obituary_list_article_is_flagged():
    # «وفيات 2014»: 94% من جمله ≤ 8 كلمات (اسم، عمر، مهنة) فلا تصلح للاسترجاع
    assert is_list_article(split_sentences(_article("وفيات 2014")))


def test_prose_article_is_not_flagged():
    # «ثقافة الاغتصاب»: نثر طويل؛ لا يجوز أن تسقطه عتبة القوائم
    assert not is_list_article(split_sentences(_article("ثقافة الاغتصاب")))


def test_multi_sentence_chunks_never_exceed_cap():
    # كان السقف يُفحص بعد إضافة الجملة فينتج هذا المقتطف مقطعاً من 356 كلمة (حتى 340 في المقالة كاملة).
    # الجملة الواحدة الأطول من السقف تبقى مقطعاً مستقلاً؛ أي مقطع متعدد الجمل يجب ألا يتجاوزه
    sents = split_sentences(_article("ثقافة الاغتصاب"))
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(sents), 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for chunk in semantic_chunks(sents, vecs):
        is_single_sentence = chunk in sents
        assert is_single_sentence or len(chunk.split()) <= MAX_CHUNK_WORDS


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

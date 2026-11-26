# -*- coding: utf-8 -*-
"""المرحلة 2: التقطيع الدلالي وبناء الفهرس.

يقرأ: wiki_sample.jsonl (مخرج المرحلة 1).
ينتج: chunks.sqlite (نص المقاطع وبياناتها) و chunk_embeddings.npy (تضمين مطبَّع).
الخطوات: تقسيم كل مقال إلى جمل ← تضمين الجمل ← قطع عند هبوط التشابه
← إعادة تضمين المقاطع الناتجة ← حفظ.
"""

import json
import sqlite3
import sys

import numpy as np

from chunking import is_list_article, semantic_chunks, split_sentences
from config import (
    CHUNKS_DB,
    EMBED_BATCH,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBEDDINGS_NPY,
    SAMPLE_JSONL,
)


def main() -> None:
    if not SAMPLE_JSONL.exists():
        sys.exit(f"لا توجد عيّنة: شغّل 01_fetch_sample.py أولاً ({SAMPLE_JSONL})")

    # استيراد كسول: المكتبة ثقيلة ولا حاجة لها قبل هذه النقطة
    from sentence_transformers import SentenceTransformer

    print(f"◄ تحميل النموذج {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        articles = [json.loads(line) for line in f]

    rows: list[tuple[int, str, str, int, str]] = []
    kept = 0
    for art in articles:
        sentences = split_sentences(art["text"])
        if is_list_article(sentences):
            print(f"  ✗ قائمة مستبعدة: {art['title']}")
            continue
        kept += 1
        sent_vecs = model.encode(
            sentences, batch_size=EMBED_BATCH, normalize_embeddings=True
        )
        for idx, chunk in enumerate(semantic_chunks(sentences, sent_vecs)):
            rows.append((art["pageid"], art["title"], art["url"], idx, chunk))
    print(f"← {kept}/{len(articles)} مقالة → {len(rows)} مقطع")

    vecs = model.encode(
        [r[4] for r in rows], batch_size=EMBED_BATCH, normalize_embeddings=True
    )
    np.save(EMBEDDINGS_NPY, vecs.astype(np.float32))  # (N, 1024) float32

    # الفهرس يُعاد بناؤه كاملاً كل مرة: العيّنة صغيرة والبناء رخيص
    CHUNKS_DB.unlink(missing_ok=True)
    con = sqlite3.connect(CHUNKS_DB)
    con.executescript(
        """
        CREATE TABLE chunks (
            row_id   INTEGER PRIMARY KEY,
            pageid   INTEGER NOT NULL,
            title    TEXT NOT NULL,
            url      TEXT NOT NULL,
            chunk_no INTEGER NOT NULL,
            text     TEXT NOT NULL
        );
        """
    )
    con.executemany(
        "INSERT INTO chunks (row_id, pageid, title, url, chunk_no, text) VALUES (?,?,?,?,?,?)",
        [(i, *r) for i, r in enumerate(rows)],
    )
    con.commit()
    con.close()

    words = np.array([len(r[4].split()) for r in rows])
    print(
        f"✓ كلمات/مقطع: min {words.min()} · median {int(np.median(words))} · max {words.max()}"
    )
    print(f"✓ {CHUNKS_DB.name} + {EMBEDDINGS_NPY.name} ({vecs.shape})")


if __name__ == "__main__":
    main()

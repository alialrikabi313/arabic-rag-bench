# -*- coding: utf-8 -*-
"""المرحلة 4: مقارنة طرق التقطيع على أسئلة ذهبية.

يقرأ: wiki_sample.jsonl و questions.jsonl.
ينتج: results.md — Hit@k و MRR لكل طريقة تقطيع (ثابت، فقرات، دلالي).
الخطوات: تقطيع المقالات بكل طريقة ← تضمين المقاطع والأسئلة ← بحث cosine
على كل المقاطع ← قياس موضع أول مقطع صحيح.
"""

import json
import sys

import numpy as np

from chunking import (
    fixed_chunks,
    is_list_article,
    paragraph_chunks,
    semantic_chunks,
    split_sentences,
)
from config import (
    EMBED_BATCH,
    BOOTSTRAP_ROUNDS,
    EMBED_DEVICE,
    EMBED_MODEL,
    EVIDENCE_WINDOW_WORDS,
    FIXED_OVERLAP_RATIO,
    FIXED_SIZE_SWEEP,
    QUESTIONS_JSONL,
    RESULTS_MD,
    SAMPLE_JSONL,
    TOP_KS,
)


def evidence_window(quote: str) -> str:
    words = quote.split()
    mid = len(words) // 2
    half = EVIDENCE_WINDOW_WORDS // 2
    return " ".join(words[max(0, mid - half) : mid + half])


def main() -> None:
    if not QUESTIONS_JSONL.exists():
        sys.exit(f"لا توجد أسئلة: شغّل 03_make_questions.py أولاً ({QUESTIONS_JSONL})")

    from sentence_transformers import SentenceTransformer

    print(f"◄ تحميل النموذج {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    with QUESTIONS_JSONL.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    wanted = {q["pageid"] for q in questions}
    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        articles = [a for a in map(json.loads, f) if a["pageid"] in wanted]

    q_vecs = model.encode(
        [q["question"] for q in questions], batch_size=EMBED_BATCH, normalize_embeddings=True
    )

    results = []
    methods = [f"fixed-{n}" for n in FIXED_SIZE_SWEEP] + ["paragraph", "semantic"]
    rng = np.random.default_rng(0)
    for method in methods:
        chunk_texts: list[str] = []
        chunk_page: list[int] = []
        for art in articles:
            if method.startswith("fixed-"):
                size = int(method.split("-")[1])
                chunks = fixed_chunks(art["text"], size, int(size * FIXED_OVERLAP_RATIO))
            elif method == "paragraph":
                chunks = paragraph_chunks(art["text"])
            else:
                sents = split_sentences(art["text"])
                if is_list_article(sents):
                    continue
                vecs = model.encode(sents, batch_size=EMBED_BATCH, normalize_embeddings=True)
                chunks = semantic_chunks(sents, vecs)
            chunk_texts += chunks
            chunk_page += [art["pageid"]] * len(chunks)

        c_vecs = model.encode(chunk_texts, batch_size=EMBED_BATCH, normalize_embeddings=True)
        page_arr = np.array(chunk_page)
        # (Q, N) — مصفوفة تشابه كاملة؛ العيّنة صغيرة فلا حاجة لفهرس تقريبي
        order = np.argsort(-(q_vecs @ c_vecs.T), axis=1)

        firsts: list[int] = []
        for qi, q in enumerate(questions):
            window = evidence_window(q["quote"])
            relevant = np.array(
                [p == q["pageid"] and window in " ".join(t.split()) for p, t in zip(page_arr, chunk_texts)]
            )
            if not relevant.any():
                continue
            ranks = np.nonzero(relevant[order[qi]])[0]
            firsts.append(int(ranks[0]) + 1)

        first_arr = np.array(firsts)
        scored = len(first_arr)
        # فترة ثقة 95% بإعادة سحب الأسئلة مع الإرجاع
        idx = rng.integers(0, scored, size=(BOOTSTRAP_ROUNDS, scored))
        boot_mrr = (1.0 / first_arr[idx]).mean(axis=1)
        boot_h1 = (first_arr[idx] <= 1).mean(axis=1)
        ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

        words = np.array([len(t.split()) for t in chunk_texts])
        results.append(
            {
                "method": method,
                "chunks": len(chunk_texts),
                "median_words": int(np.median(words)),
                "scored": scored,
                **{f"hit@{k}": float((first_arr <= k).mean()) for k in TOP_KS},
                "mrr": float((1.0 / first_arr).mean()),
                "mrr_ci": ci(boot_mrr),
                "hit1_ci": ci(boot_h1),
            }
        )
        print(f"✓ {method}: {results[-1]}")

    ks = [f"Hit@{k}" for k in TOP_KS]
    lines = [
        f"| Method | Chunks | Median words | Questions | {' | '.join(ks)} | Hit@1 95% CI | MRR | MRR 95% CI |",
        "|---|---|---|---|" + "---|" * len(ks) + "---|---|---|",
    ]
    for r in results:
        cells = " | ".join(f"{r[f'hit@{k}']:.3f}" for k in TOP_KS)
        lines.append(
            f"| {r['method']} | {r['chunks']} | {r['median_words']} | {r['scored']} | {cells} "
            f"| {r['hit1_ci'][0]:.2f}–{r['hit1_ci'][1]:.2f} | {r['mrr']:.3f} "
            f"| {r['mrr_ci'][0]:.2f}–{r['mrr_ci'][1]:.2f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ {RESULTS_MD}")


if __name__ == "__main__":
    main()

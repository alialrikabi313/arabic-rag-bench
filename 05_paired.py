# -*- coding: utf-8 -*-
"""المرحلة 5: اختبار مقترن بين طرق التقطيع على مجموعتي الأسئلة (qwen و Gemini).

يقرأ: results.ranks.json و results_gemini.ranks.json (مخرج 04_eval.py).
ينتج: results_paired.md — فرق MRR لكل زوج مع فترة ثقة bootstrap مقترنة على الأسئلة المشتركة.
الاقتران لأن الطريقتين تُقيَّمان على نفس الأسئلة؛ فترات الاستقلال الواسعة تُخفي فروقاً حقيقية.
"""

import json
import sys

import numpy as np

from config import BOOTSTRAP_ROUNDS, _HERE

PAIRS = [("fixed-200", "fixed-100"), ("fixed-150", "fixed-100"), ("paragraph", "fixed-200"), ("semantic", "fixed-100")]
SETS = {"qwen3.8:27b": "results.ranks.json", "gemini-3.5-flash-lite": "results_gemini.ranks.json"}


def paired_diff(a: dict[str, int], b: dict[str, int], rounds: int, seed: int = 0):
    """فرق متوسط 1/rank (a − b) على الأسئلة المسجَّلة في الطريقتين؛ يعيد (فرق، فترة، n)."""
    common = sorted(set(a) & set(b))
    d = np.array([1 / a[q] - 1 / b[q] for q in common])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(rounds, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), len(d)


def main() -> None:
    lines = ["| Question generator | A vs B | MRR(A) − MRR(B) | 95% paired CI | Paired n |", "|---|---|---|---|---|"]
    for gen, fname in SETS.items():
        path = _HERE / fname
        if not path.exists():
            sys.exit(f"ملف الرتب غير موجود: شغّل 04_eval.py أولاً ({path})")
        ranks = json.loads(path.read_text(encoding="utf-8"))
        for a, b in PAIRS:
            diff, (lo, hi), n = paired_diff(ranks[a], ranks[b], BOOTSTRAP_ROUNDS)
            sig = "" if lo <= 0 <= hi else " *"
            lines.append(f"| {gen} | {a} vs {b} | {diff:+.3f} | {lo:+.3f} – {hi:+.3f}{sig} | {n} |")
    lines += ["", "`*` = interval excludes 0."]
    (_HERE / "results_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""المرحلة 3ب: أسئلة ذهبية من مولّد ثانٍ (Gemini) لقياس التحيّز الذاتي.

نفس المقالات ونفس التعليمات ونفس شرط الاقتباس الحرفي في 03_make_questions.py،
لكن بنموذج من عائلة أخرى. ينتج questions_gemini.jsonl بالصيغة نفسها.
يقبل --limit لعيّنة صغيرة أولاً. قابل للاستئناف.
"""

import argparse
import importlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request

from config import (
    CHUNKS_DB,
    GEMINI_MODEL,
    GEMINI_QUESTIONS_JSONL,
    LLM_MAX_ARTICLE_WORDS,
    QUESTIONS_PER_ARTICLE,
    SAMPLE_JSONL,
)

# المرحلة 3 اسمها يبدأ برقم فلا يُستورد بـ import عادي؛ نعيد استخدام التعليمات كما هي
_q = importlib.import_module("03_make_questions")
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def ask_gemini(key: str, title: str, text: str) -> tuple[list[dict], int]:
    words = text.split()[:LLM_MAX_ARTICLE_WORDS]
    prompt = _q._PROMPT.format(n=QUESTIONS_PER_ARTICLE, title=title, text=" ".join(words))
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    for attempt in range(5):
        req = urllib.request.Request(
            _URL.format(m=GEMINI_MODEL), json.dumps(body).encode(),
            {"Content-Type": "application/json", "x-goog-api-key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 4:
                sys.exit(f"Gemini رفض الطلب: HTTP {e.code}")
            time.sleep(5 * (attempt + 1))
    tokens = data["usageMetadata"]["totalTokenCount"]
    return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])["items"], tokens


def ask_with_retry(key: str, title: str, text: str) -> tuple[list[dict] | None, int]:
    # أحياناً يعيد النموذج JSON معطوباً (صادفناه في مقالة فعلية)؛ نعيد المحاولة ثم نتخطى المقالة
    # دون كتابتها فتُعاد في التشغيل التالي (الاستئناف يعتمد على pageid المكتوب)
    spent = 0
    for _ in range(3):
        try:
            items, tokens = ask_gemini(key, title, text)
            return items, spent + tokens
        except json.JSONDecodeError:
            continue
    return None, spent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="عدد المقالات (0 = الكل)")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("عيّن المتغير GEMINI_API_KEY في البيئة")
    if not CHUNKS_DB.exists():
        sys.exit(f"لا يوجد فهرس: شغّل 02_build_index.py أولاً ({CHUNKS_DB})")

    con = sqlite3.connect(CHUNKS_DB)
    accepted = {r[0] for r in con.execute("SELECT DISTINCT pageid FROM chunks")}
    con.close()
    done: set[int] = set()
    if GEMINI_QUESTIONS_JSONL.exists():
        with GEMINI_QUESTIONS_JSONL.open(encoding="utf-8") as f:
            done = {json.loads(line)["pageid"] for line in f}
    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        articles = [a for a in map(json.loads, f) if a["pageid"] in accepted and a["pageid"] not in done]
    if args.limit:
        articles = articles[: args.limit]

    kept = dropped = tokens_total = bad = 0
    with GEMINI_QUESTIONS_JSONL.open("a", encoding="utf-8") as out:
        for i, art in enumerate(articles, 1):
            text_n = _q.norm(art["text"])
            items, tokens = ask_with_retry(key, art["title"], art["text"])
            tokens_total += tokens
            if items is None:
                bad += 1
                print(f"✗ JSON معطوب بعد 3 محاولات: {art['title']}")
                continue
            for item in items:
                # اقتباس غير حرفي = هلوسة؛ الشرط نفسه المطبَّق على المولّد المحلي
                if _q.norm(item["quote"]) not in text_n:
                    dropped += 1
                    continue
                row = {"pageid": art["pageid"], "title": art["title"],
                       "question": item["question"].strip(), "quote": _q.norm(item["quote"])}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            out.flush()
            print(f"… {i}/{len(articles)}  مقبول {kept}  مرفوض {dropped}")
    print(f"✓ مقبول {kept} · مرفوض {dropped} · مقالات معطوبة {bad} · رموز {tokens_total} ← {GEMINI_QUESTIONS_JSONL}")


if __name__ == "__main__":
    main()

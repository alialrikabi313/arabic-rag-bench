# -*- coding: utf-8 -*-
"""المرحلة 3: توليد أسئلة ذهبية للتقييم.

يقرأ: wiki_sample.jsonl والمقالات المقبولة في chunks.sqlite.
ينتج: questions.jsonl — (pageid, question, quote) حيث quote اقتباس حرفي من المقالة.
الخطوات: نموذج محلي يولّد سؤالاً واقتباساً ← حذف ما لا يوجد اقتباسه حرفياً
← حفظ. الدليل اقتباس لا مقطع، فيصلح لمقارنة أي طريقة تقطيع. قابل للاستئناف.
"""

import json
import sqlite3
import sys
import urllib.request

from config import (
    CHUNKS_DB,
    LLM_MAX_ARTICLE_WORDS,
    LLM_MODEL,
    OLLAMA_URL,
    QUESTIONS_JSONL,
    QUESTIONS_PER_ARTICLE,
    SAMPLE_JSONL,
)

_PROMPT = """أمامك مقالة من ويكيبيديا العربية. اكتب {n} أسئلة مختلفة يمكن الإجابة عنها من فقرات مختلفة في المقالة.
شروط كل سؤال:
- يُفهم السؤال وحده دون ذكر «المقالة» أو «النص».
- أعد صياغته بكلماتك ولا تنسخ عبارات الجواب حرفياً.
- quote: جملة واحدة منسوخة حرفاً بحرف من المقالة (من 8 إلى 40 كلمة) تتضمن الجواب.
أجب بـ JSON فقط: {{"items": [{{"question": "...", "quote": "..."}}]}}

العنوان: {title}

{text}"""


def norm(s: str) -> str:
    return " ".join(s.split())


def ask_llm(title: str, text: str) -> list[dict]:
    words = text.split()[:LLM_MAX_ARTICLE_WORDS]
    body = {
        "model": LLM_MODEL,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.2},
        "messages": [
            {
                "role": "user",
                "content": _PROMPT.format(
                    n=QUESTIONS_PER_ARTICLE, title=title, text=" ".join(words)
                ),
            }
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        content = json.load(resp)["message"]["content"]
    return json.loads(content)["items"]


def main() -> None:
    if not CHUNKS_DB.exists():
        sys.exit(f"لا يوجد فهرس: شغّل 02_build_index.py أولاً ({CHUNKS_DB})")

    con = sqlite3.connect(CHUNKS_DB)
    accepted = {r[0] for r in con.execute("SELECT DISTINCT pageid FROM chunks")}
    con.close()

    done: set[int] = set()
    if QUESTIONS_JSONL.exists():
        with QUESTIONS_JSONL.open(encoding="utf-8") as f:
            done = {json.loads(line)["pageid"] for line in f}

    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        articles = [a for a in map(json.loads, f) if a["pageid"] in accepted]

    kept = dropped = 0
    with QUESTIONS_JSONL.open("a", encoding="utf-8") as out:
        for i, art in enumerate(articles, 1):
            if art["pageid"] in done:
                continue
            text_n = norm(art["text"])
            for item in ask_llm(art["title"], art["text"]):
                # اقتباس غير موجود حرفياً = هلوسة، فالسؤال بلا دليل ولا يُقبل
                if norm(item["quote"]) not in text_n:
                    dropped += 1
                    continue
                row = {
                    "pageid": art["pageid"],
                    "title": art["title"],
                    "question": item["question"].strip(),
                    "quote": norm(item["quote"]),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            out.flush()
            print(f"… {i}/{len(articles)}  مقبول {kept}  مرفوض {dropped}")

    print(f"✓ مقبول {kept} · مرفوض (اقتباس غير حرفي) {dropped} ← {QUESTIONS_JSONL}")


if __name__ == "__main__":
    main()

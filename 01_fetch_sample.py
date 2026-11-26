# -*- coding: utf-8 -*-
"""المرحلة 1: جلب عيّنة من مقالات ويكيبيديا العربية.

يقرأ: واجهة ويكيبيديا العربية (مقالات عشوائية من النطاق الرئيسي).
ينتج: wiki_sample.jsonl — سطر لكل مقالة (pageid, title, url, text).
الخطوات: جلب دفعة ← حذف المقالات القصيرة ← إضافة غير المكرَّر ← تكرار حتى
اكتمال العدد. قابل للاستئناف: يقرأ الملف الموجود ويكمل من حيث توقّف.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from config import (
    BATCH_SIZE,
    CANDIDATE_BATCH,
    DATA_DIR,
    MAX_RETRIES,
    MIN_ARTICLE_WORDS,
    MIN_WIKITEXT_BYTES,
    REQUEST_PAUSE_SEC,
    RETRY_AFTER_DEFAULT_SEC,
    SAMPLE_ARTICLES,
    SAMPLE_JSONL,
    USER_AGENT,
    WIKI_API,
)


def _get(params: dict) -> dict:
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # فقط 429 قابل للإعادة؛ أي خطأ HTTP آخر حقيقي فيموت بالرسالة
            if e.code != 429:
                sys.exit(f"خطأ HTTP {e.code} من الويكيبيديا: {e.reason}")
            wait = int(e.headers.get("Retry-After", RETRY_AFTER_DEFAULT_SEC))
            print(f"… 429: انتظار {wait}ث (محاولة {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    sys.exit("استمر حدّ المعدّل (429) بعد كل المحاولات — أعد التشغيل لاحقاً")


def random_candidate_ids() -> list[int]:
    # طلب واحد لـ50 عنواناً مع حجمها؛ نبقي الكبيرة فقط
    data = _get(
        {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": CANDIDATE_BATCH,
            "prop": "revisions",
            "rvprop": "size",
        }
    )
    pages = data.get("query", {}).get("pages", {}).values()
    return [p["pageid"] for p in pages if p["revisions"][0]["size"] >= MIN_WIKITEXT_BYTES]


def fetch_extracts(page_ids: list[int]) -> list[dict]:
    data = _get(
        {
            "action": "query",
            "format": "json",
            "pageids": "|".join(map(str, page_ids)),
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
        }
    )
    return list(data.get("query", {}).get("pages", {}).values())


def load_seen_ids() -> set[int]:
    if not SAMPLE_JSONL.exists():
        return set()
    with SAMPLE_JSONL.open(encoding="utf-8") as f:
        return {json.loads(line)["pageid"] for line in f}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen_ids()
    print(f"◄ موجود مسبقاً: {len(seen)} مقالة")

    # سقف جولات يمنع حلقة لانهائية لو لم يُنتج الخادم مقالات كبيرة
    max_rounds = SAMPLE_ARTICLES * 2
    with SAMPLE_JSONL.open("a", encoding="utf-8") as out:
        for _ in range(max_rounds):
            if len(seen) >= SAMPLE_ARTICLES:
                break
            ids = [i for i in random_candidate_ids() if i not in seen]
            for start in range(0, len(ids), BATCH_SIZE):
                time.sleep(REQUEST_PAUSE_SEC)
                for page in fetch_extracts(ids[start : start + BATCH_SIZE]):
                    text = (page.get("extract") or "").strip()
                    if page["pageid"] in seen or len(text.split()) < MIN_ARTICLE_WORDS:
                        continue
                    title = page["title"]
                    row = {
                        "pageid": page["pageid"],
                        "title": title,
                        # الرابط يحفظ الإسناد المطلوب بترخيص CC BY-SA
                        "url": "https://ar.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                        "text": text,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    seen.add(page["pageid"])
                    if len(seen) >= SAMPLE_ARTICLES:
                        break
                if len(seen) >= SAMPLE_ARTICLES:
                    break
            print(f"… {len(seen)}/{SAMPLE_ARTICLES}", flush=True)
            time.sleep(REQUEST_PAUSE_SEC)
        else:
            sys.exit(f"تعذّر بلوغ {SAMPLE_ARTICLES} مقالة بعد {max_rounds} جولة")

    print(f"✓ تم: {len(seen)} مقالة ← {SAMPLE_JSONL}")


if __name__ == "__main__":
    main()

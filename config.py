# -*- coding: utf-8 -*-
"""الإعدادات: كل المسارات والأرقام هنا. عدّل هنا فقط عند تغيير العيّنة أو المصدر."""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# البيانات الخام خارج الريبو (مجلد شقيق) كي لا تدخل git ولا يُنشر نصّ بالغلط
DATA_DIR = Path(os.environ.get("ARB_DATA_DIR", _HERE.parent / "arabic-rag-bench-data"))
# لقطة مجمّدة من 300 مقالة: الجلب العشوائي غير قابل للتكرار فنثبّت المدخل (المعرّفات في sample_pageids.txt)
SAMPLE_JSONL = DATA_DIR / "wiki_sample_300.jsonl"

WIKI_API = "https://ar.wikipedia.org/w/api.php"

# اسم عميل عام: الويكيبيديا تطلب User-Agent معرِّفاً بلا بيانات شخصية
USER_AGENT = "arabic-rag-bench/0.1 (research sample fetcher)"

# 50 كفت لاختبار الخط؛ 300 لتضييق الضجيج بعد أن كانت فروق 80 سؤالاً غير حاسمة
SAMPLE_ARTICLES = 300

# المقالات القصيرة (بذور stub) لا تصلح لاختبار الاسترجاع
MIN_ARTICLE_WORDS = 150

# المقالات تُجلب على دفعات؛ 20 هو سقف exlimit عند طلب النصوص الكاملة
BATCH_SIZE = 20

# فحص مسبق بحجم الـwikitext (بايت) قبل جلب النص: أغلب المقالات العشوائية بذور قصيرة،
# فالفلترة بالحجم توفّر أغلب الطلبات. 5000 بايت ≈ أقل بقليل من 150 كلمة نظيفة بعد إزالة الترميز
CANDIDATE_BATCH = 50
MIN_WIKITEXT_BYTES = 5000

# 1.0 ثانية أطلقت 429 بعد ~20 طلباً؛ 3.0 مقايضة بين السرعة وحدّ المعدّل
REQUEST_PAUSE_SEC = 3.0

# عند 429 ننتظر Retry-After إن وُجد وإلا هذه القيمة، ثم نعيد حتى سقف المحاولات
RETRY_AFTER_DEFAULT_SEC = 30
MAX_RETRIES = 5

# bge-m3: عربي قوي وسياق طويل وبُعد 1024؛ multilingual-e5 أخفّ لكن أضعف على الفصحى الطويلة
EMBED_MODEL = "BAAI/bge-m3"
EMBED_BATCH = 64

# None = اختيار تلقائي (cuda إن وُجد وإلا cpu)؛ يُفرض عبر ARB_DEVICE=cpu|cuda
EMBED_DEVICE = os.environ.get("ARB_DEVICE")

# التقطيع الدلالي: نقطع عند أخفض N% من تشابه الجمل المتجاورة داخل المقال نفسه.
# نسبية لا مطلقة لأن مقاييس التشابه تختلف بين المقالات؛ القيمة مبدئية بانتظار مراجعة العيّنة
SEMANTIC_BREAK_PERCENTILE = 20

# مقطع أقصر من هذا يُدمج بما قبله، وأطول من هذا يُقطع قسراً ولو لم يهبط التشابه
MIN_CHUNK_WORDS = 40
MAX_CHUNK_WORDS = 250

CHUNKS_DB = DATA_DIR / "chunks.sqlite"
EMBEDDINGS_NPY = DATA_DIR / "chunk_embeddings.npy"

# مقالة أكثر من 60% من جملها ≤ 8 كلمات = قائمة/فيلموغرافيا، لا تصلح لاختبار الاسترجاع.
# 0.5 كانت ستُسقط مقالات نثرية حقيقية؛ 0.6 أسقط 10 من 50 في العيّنة الأولى
LIST_SHORT_SENT_WORDS = 8
LIST_ARTICLE_MAX_SHORT_RATIO = 0.6

# خط الأساس للمقارنة: مقاطع ثابتة بالكلمات مع تداخل (الطريقة الشائعة)
FIXED_CHUNK_WORDS = 150
FIXED_CHUNK_OVERLAP = 30

# توليد أسئلة الاختبار محلياً: qwen3.8:27b أكبر نموذج عربي متاح عندي؛ الأصغر يهلوس اقتباسات
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen3.8:27b"
QUESTIONS_PER_ARTICLE = 2
QUESTIONS_JSONL = DATA_DIR / "questions.jsonl"

# المقالة الأطول تُقتطع قبل إرسالها كي لا يتجاوز السياق؛ الأسئلة تُولَّد من هذا الجزء فقط
LLM_MAX_ARTICLE_WORDS = 1500

RESULTS_MD = _HERE / "results.md"
TOP_KS = (1, 3, 5)

# الدليل الصحيح = مقطع من نفس المقالة يحوي نافذة الكلمات الوسطى من الاقتباس؛
# الوسط لا الطرفان لأن القطع الثابت قد يشطر الاقتباس عند حدّه
EVIDENCE_WINDOW_WORDS = 6

# مسح أحجام الثابت: وسيط الدلالي ~100 والفقرات ~200، فنقارن بأحجام مضاهية.
# التداخل 20% من الحجم ثابتاً في كل الأحجام كي لا يتغيّر معه متغيّر ثانٍ
FIXED_SIZE_SWEEP = (100, 150, 200)
FIXED_OVERLAP_RATIO = 0.2

# فترات ثقة bootstrap على مستوى السؤال؛ 2000 إعادة تكفي لفترة 95% مستقرة
BOOTSTRAP_ROUNDS = 2000

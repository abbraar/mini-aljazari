# app/shared/theme_gemini.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import pathlib
import re
import time
from typing import Optional, Tuple

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None

# فئاتك الثابتة (Parent taxonomy)
CANON_THEMES = {
    "غزل","وطنية","رياضية","دينية","شوق","حزن","عتاب","فخر",
    "تراث","احتفال","بحر","طموح","مرح","سفر","نقد","طبيعة",
}

# إعدادات عبر البيئة
ALLOW_NEW_THEMES = os.getenv("ALLOW_NEW_THEMES", "0") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CACHE_PATH = pathlib.Path(os.getenv("THEME_CACHE_PATH", "theme_llm_cache.json"))
DYNAMIC_PATH = pathlib.Path(os.getenv("THEME_DYNAMIC_PATH", "themes_dynamic.json"))
PROMOTE_MIN_COUNT = int(os.getenv("PROMOTE_MIN_COUNT", "50"))  # كم مرة تظهر الفئة الجديدة قبل “ترقيتها”

def _load_json(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_json(path: pathlib.Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _sanitize_label(lbl: str) -> str:
    # سلاسة بسيطة: قصير، عربي/مسافات فقط
    lbl = (lbl or "").strip()
    lbl = re.sub(r"[\n\r\t]+", " ", lbl)
    lbl = re.sub(r"\s{2,}", " ", lbl)
    return lbl[:40]

def _promote_if_needed(new_label: str) -> None:
    stats = _load_json(DYNAMIC_PATH)
    cnt = int(stats.get(new_label, 0)) + 1
    stats[new_label] = cnt
    _save_json(DYNAMIC_PATH, stats)
    # عند الرغبة: لو cnt >= PROMOTE_MIN_COUNT نطبع تحذير/لوج إن الفئة مرشحة للترقية
    if cnt == PROMOTE_MIN_COUNT:
        print(f"[theme_gemini] Candidate theme reached {PROMOTE_MIN_COUNT}: {new_label}")

def guess_theme_gemini(text: str) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """
    Returns (theme, score, why, parent_theme)
    - theme: قد تكون فئة جديدة إذا ALLOW_NEW_THEMES=1
    - parent_theme: فئة مرجعية من CANON_THEMES (قد تكون None)
    """
    text = (text or "").strip()
    api_key = os.getenv("GEMINI_API_KEY")
    if not text or not api_key or genai is None:
        return None, None, None, None

    cache = _load_json(CACHE_PATH)
    if text in cache:
        r = cache[text]
        return r.get("theme"), r.get("score"), r.get("why"), r.get("parent")

    genai.configure(api_key=api_key)

    # موجهات (Prompt): اسمح بابتكار فئات قصيرة بالعربية
    system_hint = (
        "أنت مصنف موضوعات لأبيات/مقاطع شعرية ولهجات سعودية/عربية.\n"
        "اختر فئة عربية قصيرة (1–3 كلمات) تصف الموضوع بدقة.\n"
        "إن أمكن، أعطِ أيضًا فئة أب (parent_theme) واحدة فقط من القائمة التالية:\n"
        f"{', '.join(sorted(CANON_THEMES))}\n"
        "الرد بصيغة JSON فقط هكذا:\n"
        "{\"theme\": \"...\", \"score\": 0..1, \"why\": \"...\", \"parent_theme\": \"...\"}\n"
        "إذا لم تكن واثقًا من فئة الأب، اترك parent_theme فارغًا."
    )

    prompt = f"{system_hint}\n\nالنص:\n{text}\n"

    # محاولات بسيطة لإعادة المحاولة
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            raw = (resp.text or "").strip().strip("`").strip()
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                m = re.search(r'\"theme\"\s*:\s*\"([^\"]+)\"', raw)
                if m:
                    data = {"theme": m.group(1), "score": 0.6, "why": raw[:120], "parent_theme": ""}

            if not data:
                continue

            theme = _sanitize_label(str(data.get("theme", "")))
            score = float(data.get("score", 0.6)) if str(data.get("score","")).strip() else 0.6
            why   = _sanitize_label(str(data.get("why", "")))[:200]
            parent = str(data.get("parent_theme", "") or "").strip()

            # لو ممنوع فئات جديدة: قيّد الإخراج على CANON_THEMES فقط
            if not ALLOW_NEW_THEMES:
                theme = parent if parent in CANON_THEMES else (theme if theme in CANON_THEMES else None)

            # لو سمحنا بالجديد: نقبل أي label عربي قصير؛ لكن نختزن parent لو كان صحيح
            if theme:
                cache[text] = {"theme": theme, "score": score, "why": why, "parent": (parent if parent in CANON_THEMES else None)}
                _save_json(CACHE_PATH, cache)
                # عدّاد للفئات الجديدة (للترقية لاحقًا)
                if ALLOW_NEW_THEMES and theme not in CANON_THEMES:
                    _promote_if_needed(theme)
                return theme, score, why, (parent if parent in CANON_THEMES else None)
        except Exception:
            time.sleep(0.6 * (attempt + 1))

    return None, None, None, None

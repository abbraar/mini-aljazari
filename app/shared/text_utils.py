# app/shared/text_utils.py
# -*- coding: utf-8 -*-
"""
Arabic text utilities for Al-Jazari:
- Light normalization (norm_ar): safe, display/rules/dialect use.
- Index normalization (norm_ar_index): heavier, retrieval indexing only.
- Rules-based theme tagging with multi-hit, length-weighted scoring.
- Optional Gemini fallback (can invent new Arabic categories if ALLOW_NEW_THEMES=1).
- Display helpers to choose a human-friendly tag.

Env toggles:
    LOG_LEVEL=INFO|DEBUG|...
    ALLOW_NEW_THEMES=0|1          # allow Gemini to output new categories
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()  # this reads .env into os.environ

# ---------------- Logging setup ----------------
logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid duplicate handlers in reloads (e.g., uvicorn --reload)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(_handler)
# default WARNING; override with env: LOG_LEVEL=DEBUG (INFO, ERROR, etc.)
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING))

# ---------------- Optional dependencies (guarded) ----------------
try:
    import arabicprocess as ap  # stopwords / stemming
    logger.debug("arabicprocess loaded successfully.")
except Exception:
    ap = None
    logger.debug("arabicprocess not available")

try:
    import pyarabic.araby as araby  # digit normalization, tashkeel utils
    logger.debug("pyarabic loaded successfully.")
except Exception:
    araby = None
    logger.debug("pyarabic not available")

# Optional Gemini fallback
try:
    from app.shared.theme_gemini import guess_theme_gemini  # returns (theme, score, why, parent)
except Exception:
    guess_theme_gemini = None  # type: ignore

ALLOW_NEW_THEMES = os.getenv("ALLOW_NEW_THEMES", "0") == "1"

# ---------------- Regexes ----------------
_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")  # harakat incl. 0670
_TATWEEL = re.compile(r"\u0640+")                           # ـــ
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")                # strip non-word non-Arabic punct
_MULTI_SPACE = re.compile(r"\s+")
_AR_WORDS = re.compile(r"[\u0600-\u06FF]{3,}")              # (if needed elsewhere)

def _strip_diacritics(s: str) -> str:
    return _DIAC.sub("", s)

def _basic_orthographic_norm(s: str) -> str:
    # widely used Arabic IR unifications
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    return s

def _collapse_ws(s: str) -> str:
    return _MULTI_SPACE.sub(" ", s).strip()

# Translate Arabic-Indic and eastern Arabic-Indic digits to ASCII 0-9
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"  # U+0660..U+0669
_EASTERN_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0..U+06F9 (Persian/Urdu forms)
_ASCII_DIGITS = "0123456789"
_DIGIT_TRANSLATION = str.maketrans({
    **{ord(a): ord(b) for a, b in zip(_ARABIC_INDIC_DIGITS, _ASCII_DIGITS)},
    **{ord(a): ord(b) for a, b in zip(_EASTERN_ARABIC_INDIC_DIGITS, _ASCII_DIGITS)},
})

def _to_ascii_digits(s: str) -> str:
    """Normalize Arabic-Indic digits (e.g., ٠١٢) and eastern forms (۰۱۲) to ASCII (012)."""
    return s.translate(_DIGIT_TRANSLATION)

# ---------------- Normalization ----------------
def norm_ar(s: str) -> str:
    """
    Light, SAFE normalization.
    Use for:
      - dialect classification
      - theme rules
      - display/debug
    """
    if not s:
        return ""
    try:
        s = s.strip()
        s = _strip_diacritics(s)
        s = _basic_orthographic_norm(s)
        s = _collapse_ws(s)
        return s
    except Exception as e:
        logger.error("norm_ar failed: %s", e)
        return (s or "").strip()

def norm_ar_index(
    s: str,
    *,
    remove_tatweel: bool = True,
    normalize_digits: bool = True,
    strip_punct: bool = True,
    stopwords: bool = True,
    stem: bool = False,
) -> str:
    """
    Heavier, INDEX-ORIENTED normalization.
    Use ONLY for retrieval indexing/query (TF-IDF / dense encoders).
    Do NOT use for dialect/theme classification.
    """
    if not s:
        return ""
    try:
        t = norm_ar(s)  # start from safe normalizer

        if remove_tatweel:
            t = _TATWEEL.sub("", t)

        if normalize_digits:
            try:
                # Prefer pyarabic if it exposes normalize_digits; otherwise use local mapping
                if araby is not None and hasattr(araby, "normalize_digits"):
                    t = araby.normalize_digits(t)  # type: ignore[attr-defined]
                else:
                    t = _to_ascii_digits(t)
            except Exception as e:
                logger.error("Digit normalization failed: %s", e)

        if strip_punct:
            t = _PUNCT.sub(" ", t)

        # Optional utilities from arabicprocess (guarded)
        if ap is not None:
            if stopwords:
                try:
                    t = ap.remove_stopwords(t)
                except Exception as e:
                    logger.error("Stopword removal failed: %s", e)
            if stem:
                try:
                    t = ap.stem_text(t)  # careful: may hurt poetry precision
                except Exception as e:
                    logger.error("Stemming failed: %s", e)

        return _collapse_ws(t)
    except Exception as e:
        logger.error("norm_ar_index failed: %s", e)
        return _collapse_ws(s)

# ---------------- Theme rules (priority now by score, not order) ----------------
# Keep lists focused; prefer high-precision cues (multi-word phrases allowed).
THEME_RULES: Dict[str, List[str]] = {
    "غزل": [
        # existing
        "حب","حبيب","حبيبتي","حبيبي","عشق","غرام","هوى","وله","شوق",
        "اشتقت","اهواك","اعشقك","ود","وصال","غلا","عيونك","عينك",
        "جمال","خدود","قبل","شفايف","ضحكتك","خجلك","سهران","حنين",
        "لقانا","وصلك","هواك","وصالك","تغزل","غزلي","الهيام","الولهان",
        "احساسك","احساس",
        # dataset-driven
        "احبك","يا حبيبي","نظرة","نظره","مبسم","مبسّم","بسمة","بسمتي","غنوتي",
        "عطر","شذى","قلبي","روحي","مفتون","مفتونة","ولهفه","ولهفة",
        "يا ساهي الطرف","يا ابو غرة","الزين","الحسن","العاشق","مغرم",
        # precision additions for missed lines
        "يا حياتي","حياتي","يا عمري","عمري","ودي تكون لي","تكون لي","غلاك","غالي",
        "محبة","محبّة","قلبك","قلبي عليك"
    ],
    "وطنية": [
        "سعودي","سعودية","الوطن","وطن","بلاد","موطني","راية","العلم","الملك",
        "سلمان","وحدة","توحد","بيعة","دار","جزيرة","عبدالعزيز","عبد العزيز",
        "مجدي","فخر","بلادي","ترابها","وطننا","المملكة","نهضة","رؤية","2030",
        "وطن العز","دارنا","حكامنا","العلم الأخضر","المجد","رافع","رايتنا",
        "السعودية","يا سعودي","آل السعود","الاخضر","الأخضر","نصر","منصور",
        "حماها","نحماها","حامي الذمم","قبلة الاسلام","ديرتي","ديرة","جزيرتنا",
        "سيري بنا سيري","حبي لأرضك","دار السلام","بلدي","بلادي يا بلادي"
    ],
    "رياضية": [
        "نادي","الاتحاد","الاتي","الهلال","الاهلي","النصر","الشباب","الفيصلي",
        "الاتفاق","الطائي","العدالة","ملعب","مدرج","جمهور","تشجيع","مباراة",
        "بطولة","دوري","كاس","كأس","مدرب","لاعب","هدف","ركلة","شوط","تسديدة",
        "فوز","هزيمة","تعادل","حكم","تبديل","ميدان","كرة",
        "اتحاديين","فارس الملعب","الأخضر","الاخضر","مبروك الكاس","استلمت الكاس",
        "اصفر اتي","اسود اتي","عشق الملايين","اهتفوا للاخضر","قولوا معانا",
        "الملعب","البطل","تشجيع الجمهور"
    ],
    "دينية": [
        "اللهم","سبحان","الحمد لله","استغفر الله","يارب","يا رب","ربي",
        "ربنا","نبي","رسول","محمد","صلى الله عليه وسلم","مكة","المدينة","المدينه",
        "حج","عمرة","دعاء","ايمان","توحيد","مسجد","قبلة","قرآن","سنة","حديث",
        "صلاة","الصلاة","زكاة","صيام","رمضان","الجمعة","قيام الليل","جنة","نار",
        "عذاب","آخرة","توبة","رحمة","استغفار","تسبيح","تهليل","تكبير","تحميد",
        "الصلاة على الرسول","قبلة الاسلام","جنود الله","الشريعة"
    ],
    "شوق": [
        "اشتاق","مشتاق","أشتاق","أوله","حنين","تذكر","تذكرت","رجعة","غبت",
        "غيابك","وداع","رجوع","ما نسيتك","بعدك","طيفك","غيبتك",
        "ولهفه","لهفة","مشتاق لو تدري","النظرة ماهي دليل","يا خلي"
    ],
    "حزن": [
        "فراق","وداع","رحيل","دمع","دموعي","بكيت","جروح","حزن","كسر","وجع",
        "غبت","غياب","وحدي","ضياع","مأساة",
        "قلبي الجريح","هموم","الملام","المعاتب","العتب يوجع"
    ],
    "عتاب": [
        "غدرت","خيانة","جفيت","نسيت","قصرت","زعلت","ظلمتني","غبت عني",
        "ما سألت","ما وفيت","وعدك","كذبت","خان","تجاهلت","عذرك","لوم",
        "اعتذار","اعتذر","عاتب","معاتب","تشره","سامح","تسامح","ترضى","رضا",
        "مافي الزعل","يا حطه اللي يسامح","طولت غيابك","لا تكون عنيد"
    ],
    "فخر": [
        "فخر","عز","كرامة","نخوة","شجاعة","سيف","سلالة","كرم","مرجلة",
        "فزعة","رجال","بطولة","قوة","مروءة","هيبة",
        "المعارك","البطل","الشهم","راعي الشيم","السعودي قوة","نرفع الراية",
        "نصرٌ وعز","مجد","علاها العلم"
    ],
    "تراث": [
        "خيمة","ناقة","ربابة","قهوة","فنجال","البدو","البدوي","عقال","شماغ",
        "صحرى","رمل","دهن العود","فارس","جمل","تراث","عرضة","مزمار",
        "دانه يا ليل لدانه","بوادي","حضر"
    ],
    "احتفال": [
        "زفة","زواج","عرس","فرح","ليلة العمر","مبروك","ملكة","تهنئة","طبول",
        "مزمار","نقوط","مناسبات","حفل","حفلة","عيد","سعادة",
        "الليلة","يكمل فرحنا","صفقوا","غنوا عشانه","مرحبا باللي جاء"
    ],
    "بحر": [
        "بحر","شاطئ","موج","مرسى","صياد","شراع","لؤلؤ","غوص","نسيم",
        "قارب","مركب","سفينة","عاصفة","ملاح","أمواج",
        "العين بحر"
    ],
    "طموح": [
        "حلم","طموح","سهر","تعب","كفاح","اصرار","نجاح","انجاز","هدف",
        "امل","اصراري","أواصل","أتحدى","أحقق","اجتهاد",
        "المعالي","العزايم ما تلين","أحلامي","احلامي","احلام كبيرة","طموحاتي"
    ],
    "مرح": [
        "طرب","نغني","نرقص","سهر","سهرة","فرحة","مزاج","ليلة","نغم",
        "نغمة","عزف","موسيقى","نوتة","تصفيق","بهجة","ضحك",
        "ماشالله","يا هلا","قولوا معانا","دانه يا ليل","يا غنوتي"
    ],
    "سفر": [
        "غربة","سفر","رجعة","بعيد","أرجع","تذكرة","مطار","وداع","أهل",
        "ديرة","مدينة","رجعت","اشتقت للوطن","عودة"
    ],
    "نقد": [
        "زمن","دنيا","ناس","مجتمع","حالنا","سخرية","نفاق","قهر","ظلم",
        "سالفة","زمان","حياة","واقع","تهكم","تعب","هم",
        "المشكلة","اسئلة","زمانك منقضي"
    ],
    "طبيعة": [
        "مطر","غيم","رعد","برق","نسيم","ورد","زهور","ربيع","روضة",
        "زهر","أمطار","سماء","عصافير","حقل","نسمة","غابة"
    ]
}

# --- Compile theme rules into Arabic-friendly token-boundary regexes ---
# We normalize each pattern with norm_ar so matching is consistent with input normalization.
def _compile_theme_rules(rules: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = {}
    for theme, pats in rules.items():
        compiled_list: List[re.Pattern] = []
        for p in pats:
            try:
                # Use LIGHT normalization for rule terms, then guard with whitespace boundaries
                # (?<!\S) and (?!\S) are robust for Arabic token delimiting without breaking diacritics.
                norm_p = norm_ar(p)
                compiled_list.append(re.compile(rf"(?<!\S){re.escape(norm_p)}(?!\S)"))
            except Exception as e:
                logger.error("Failed compiling theme pattern '%s' for theme '%s': %s", p, theme, e)
        compiled[theme] = compiled_list
    return compiled

_COMPILED_THEME_RULES: Dict[str, List[re.Pattern]] = _compile_theme_rules(THEME_RULES)

# --------- Multi-hit, length-weighted scoring (replaces first-match) ---------
def guess_theme_rules_with_match(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (theme, matched_regex_or_literal, matched_span) using LIGHT-normalized text.
    We score themes by (#hits * length_weight). If max score is low, return "اخرى".
    """
    t = norm_ar(text)
    best_theme = "اخرى"
    best_score = 0.0
    best_span = None
    best_pat = None

    for theme, patterns in _COMPILED_THEME_RULES.items():
        score = 0.0
        local_best_span = None
        local_best_pat = None
        for rgx in patterns:
            for m in rgx.finditer(t):
                span = m.group(0)
                # weight by pattern length to prefer multi-word phrases
                w = max(1.0, len(span) / 3.0)
                score += w
                if (local_best_span is None) or (len(span) > len(local_best_span)):
                    local_best_span = span
                    local_best_pat = rgx.pattern
        if score > best_score:
            best_score, best_theme = score, theme
            best_span, best_pat = local_best_span, local_best_pat

    # low-confidence guard: if nothing matched, or score < 1.0, fall back to "اخرى"
    if best_score < 1.0:
        return "اخرى", None, None
    return best_theme, best_pat, best_span

def tag_theme(text: str) -> str:
    """Rules-only theme (kept for backward compatibility)."""
    theme, _, _ = guess_theme_rules_with_match(text)
    return theme

def tag_theme_smart(text: str, use_llm_fallback: bool = True, min_llm_score: float = 0.55) -> str:
    """
    Rules first; if result is 'اخرى' and LLM is enabled, call Gemini.
    If ALLOW_NEW_THEMES=1, we accept the new Gemini label; otherwise use parent if available.
    """
    base, _, _ = guess_theme_rules_with_match(text)
    if base != "اخرى":
        return base

    if not use_llm_fallback or guess_theme_gemini is None:
        return "اخرى"

    llm_theme, score, _, parent = guess_theme_gemini(text)  # type: ignore[misc]
    if not llm_theme or (score or 0.0) < min_llm_score:
        return "اخرى"

    if ALLOW_NEW_THEMES:
        return llm_theme
    return parent or (llm_theme if llm_theme in THEME_RULES else "اخرى")

# ---------------- Display helpers (optional, for API/UI) ----------------
def infer_theme_for_text(text_norm_light: str, smart: bool = False) -> str:
    """
    Try to infer a non-'اخرى' theme from LIGHT-normalized text.
    If smart=True, allows LLM fallback per environment toggles.
    Returns '' if no clear theme is found.
    """
    if smart:
        t = tag_theme_smart(text_norm_light, use_llm_fallback=True)
    else:
        t, _, _ = guess_theme_rules_with_match(text_norm_light)
    return "" if (t == "اخرى") else t

def choose_display_tag(
    doc_theme: str,
    text_norm_light: str,
    dialect: str = "",
    use_dialect_fallback: bool = False,
    smart: bool = False,
) -> str:
    """
    Prefer a clean, human-friendly tag to display:
      1) doc_theme if present and not 'اخرى'
      2) inferred theme from text (rules-based or smart with LLM)
      3) (optional) dialect as a last resort if use_dialect_fallback=True
      4) else '' (show no tag)
    """
    if doc_theme and doc_theme != "اخرى":
        return doc_theme
    inferred = infer_theme_for_text(text_norm_light, smart=smart)
    if inferred:
        return inferred
    if use_dialect_fallback and dialect:
        return dialect
    return ""

# Public symbols
__all__ = [
    "norm_ar", "norm_ar_index",
    "THEME_RULES",
    "guess_theme_rules_with_match", "tag_theme", "tag_theme_smart",
    "infer_theme_for_text", "choose_display_tag",
]


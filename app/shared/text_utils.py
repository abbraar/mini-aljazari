# app/shared/text_utils.py
from __future__ import annotations

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

# ---------------- Logging setup ----------------
logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid duplicate handlers in reloads
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(_handler)
# default WARNING; override with env: LOG_LEVEL=DEBUG (INFO, ERROR, etc.)
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING))

# ---------------- Optional dependencies (guarded) ----------------
try:
    import arabicprocess as ap  # stopwords / stemming
    logger.debug("arabicprocess loaded successfully.")
except Exception as e:
    ap = None
    logger.warning("arabicprocess not available: %s", e)

try:
    import pyarabic.araby as araby  # digit normalization, tashkeel utils
    logger.debug("pyarabic loaded successfully.")
except Exception as e:
    araby = None
    logger.warning("pyarabic not available: %s", e)

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

# ---------------- Theme rules (priority matters; first match wins) ----------------
THEME_RULES: Dict[str, List[str]] = {
    "غزل": [
        "حب", "حبيب", "حبيبتي", "حبيبي", "عشق", "غرام", "هوى", "وله", "شوق",
        "اشتقت", "اهواك", "اعشقك", "ود", "وصال", "غلا", "عيونك", "عينك",
        "جمال", "خدود", "قبل", "شفايف", "ضحكتك", "خجلك", "سهران", "حنين",
        "لقانا", "وصلك", "هواك", "وصالك", "تغزل", "غزلي", "الهيام", "الولهان",
        "احساسك", "احساس"
    ],
    "وطنية": [
        "سعودي", "سعودية", "الوطن", "وطن", "بلاد", "موطني", "راية", "العلم", "الملك",
        "سلمان", "وحدة", "توحد", "بيعة", "دار", "جزيرة", "عبدالعزيز", "عبد العزيز",
        "مجدي", "فخر", "بلادي", "ترابها", "وطننا", "المملكة", "نهضة", "رؤية", "2030",
        "وطن العز", "دارنا", "حكامنا", "العلم الأخضر", "المجد", "رافع", "رايتنا"
    ],
    "رياضية": [
        "نادي", "الاتحاد", "الاتي", "الهلال", "الاهلي", "النصر", "الشباب", "الفيصلي",
        "الاتفاق", "الطائي", "العدالة", "ملعب", "مدرج", "جمهور", "تشجيع", "مباراة",
        "بطولة", "دوري", "كاس", "كأس", "مدرب", "لاعب", "هدف", "ركلة", "شوط", "تسديدة",
        "فوز", "هزيمة", "تعادل", "حكم", "تبديل", "ميدان", "كرة"
    ],
    "دينية": [
        "الله", "اللهم", "سبحان", "الحمد لله", "استغفر الله", "يارب", "يا رب", "ربي",
        "ربنا", "نبي", "رسول", "محمد", "صلى الله عليه وسلم", "مكة", "المدينة", "المدينه",
        "حج", "عمرة", "دعاء", "ايمان", "توحيد", "مسجد", "قبلة", "قرآن", "سنة", "حديث",
        "صلاة", "الصلاة", "زكاة", "صيام", "رمضان", "الجمعة", "قيام الليل", "جنة", "نار",
        "عذاب", "آخرة", "توبة", "رحمة", "استغفار", "تسبيح", "تهليل", "تكبير", "تحميد"
    ],
    "شوق": [
        "اشتاق", "مشتاق", "أشتاق", "أوله", "حنين", "تذكر", "تذكرت", "رجعة", "غبت",
        "غيابك", "وداع", "رجوع", "ما نسيتك", "بعدك", "طيفك", "غيبتك"
    ],
    "حزن": [
        "فراق", "وداع", "رحيل", "دمع", "دموعي", "بكيت", "جروح", "حزن", "كسر", "وجع",
        "غبت", "غياب", "وحدي", "ضياع", "مأساة"
    ],
    "عتاب": [
        "غدرت", "خيانة", "جفيت", "نسيت", "قصرت", "زعلت", "ظلمتني", "غبت عني",
        "ما سألت", "ما وفيت", "وعدك", "كذبت", "خان", "تجاهلت", "عذرك", "لوم"
    ],
    "فخر": [
        "فخر", "عز", "كرامة", "نخوة", "شجاعة", "سيف", "سلالة", "كرم", "مرجلة",
        "فزعة", "رجال", "بطولة", "قوة", "مروءة", "هيبة"
    ],
    "تراث": [
        "خيمة", "ناقة", "ربابة", "قهوة", "فنجال", "البدو", "البدوي", "عقال", "شماغ",
        "صحرى", "رمل", "دهن العود", "فارس", "جمل", "تراث", "عرضة", "مزمار"
    ],
    "احتفال": [
        "زفة", "زواج", "عرس", "فرح", "ليلة العمر", "مبروك", "ملكة", "تهنئة", "طبول",
        "مزمار", "نقوط", "مناسبات", "حفل", "حفلة", "عيد", "سعادة"
    ],
    "بحر": [
        "بحر", "شاطئ", "موج", "مرسى", "صياد", "شراع", "لؤلؤ", "غوص", "نسيم",
        "قارب", "مركب", "سفينة", "عاصفة", "ملاح", "أمواج"
    ],
    "طموح": [
        "حلم", "طموح", "سهر", "تعب", "كفاح", "اصرار", "نجاح", "انجاز", "هدف",
        "امل", "اصراري", "أواصل", "أتحدى", "أحقق", "اجتهاد"
    ],
    "مرح": [
        "طرب", "نغني", "نرقص", "سهر", "سهرة", "فرحة", "مزاج", "ليلة", "نغم",
        "نغمة", "عزف", "موسيقى", "نوتة", "تصفيق", "بهجة", "ضحك"
    ],
    "سفر": [
        "غربة", "سفر", "رجعة", "بعيد", "أرجع", "تذكرة", "مطار", "وداع", "أهل",
        "ديرة", "مدينة", "رجعت", "اشتقت للوطن", "عودة"
    ],
    "نقد": [
        "زمن", "دنيا", "ناس", "مجتمع", "حالنا", "سخرية", "نفاق", "قهر", "ظلم",
        "سالفة", "زمان", "حياة", "واقع", "تهكم", "تعب", "هم"
    ],
    "طبيعة": [
        "مطر", "غيم", "رعد", "برق", "نسيم", "ورد", "زهور", "ربيع", "روضة",
        "زهر", "أمطار", "سماء", "عصافير", "حقل", "نسمة", "غابة"
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
                # (?<!\S) and (?!\S) are robust for Arabic token delimiting without breaking on diacritics.
                norm_p = norm_ar(p)
                compiled_list.append(re.compile(rf"(?<!\S){re.escape(norm_p)}(?!\S)"))
            except Exception as e:
                logger.error("Failed compiling theme pattern '%s' for theme '%s': %s", p, theme, e)
        compiled[theme] = compiled_list
    return compiled

_COMPILED_THEME_RULES: Dict[str, List[re.Pattern]] = _compile_theme_rules(THEME_RULES)

def guess_theme_rules_with_match(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (theme, matched_regex_or_literal, matched_span) using LIGHT-normalized text.
    Priority matters: first match wins.
    """
    t = norm_ar(text)
    for theme, patterns in _COMPILED_THEME_RULES.items():
        for rgx in patterns:
            m = rgx.search(t)
            if m:
                return theme, rgx.pattern, m.group(0)
    return "اخرى", None, None

def tag_theme(text: str) -> str:
    theme, _, _ = guess_theme_rules_with_match(text)
    return theme

# Public symbols
__all__ = [
    "norm_ar", "norm_ar_index",
    "THEME_RULES",
    "guess_theme_rules_with_match", "tag_theme",
]

# ---------------- Quick self-test ----------------
if __name__ == "__main__":
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    logger.setLevel(logging.DEBUG)

    sample = "أحب وطني 🇸🇦 وأهلي كثيراً!!! ٠١٢"
    print("light :", norm_ar(sample))
    print("index :", norm_ar_index(sample))

    # Theme quick check
    ex1 = "أعشقك يا حبيبي"
    ex2 = "راية المملكة خفاقة"
    print("tag(ex1):", tag_theme(ex1), "match:", guess_theme_rules_with_match(ex1))
    print("tag(ex2):", tag_theme(ex2), "match:", guess_theme_rules_with_match(ex2))

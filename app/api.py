# app/api.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import pathlib
import re
from typing import List, Optional, Tuple, Dict

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.shared.text_utils import (
    norm_ar, norm_ar_index, guess_theme_rules_with_match
)
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Paths & data
# -------------------------
DATA_DIR = pathlib.Path("dataset/processed")
DATA_PATH = DATA_DIR / (
    "lyrics_clean_retagged.jsonl"
    if (DATA_DIR / "lyrics_clean_retagged.jsonl").exists()
    else "lyrics_clean.jsonl"
)
THEME_MODEL_PATH = pathlib.Path("models/theme_clf.joblib")
DIALECT_MODEL_PATH = pathlib.Path("models/dialect_clf.joblib")
HF_DIALECT_DIR = pathlib.Path("models/dialect_hf")  # optional HF folder (from train_dialect_saudibert.py)
RETRIEVER_DIR = pathlib.Path("retriever")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\n"
        "Run: python dataset/preprocess.py (and optionally: python dataset/retag.py)"
    )

docs: List[dict] = []
# Prefer retriever/corpus.jsonl to keep alignment with persisted TF-IDF artifacts
corpus_path = RETRIEVER_DIR / "corpus.jsonl"
if corpus_path.exists():
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
else:
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue

# ---- Dual corpora ----
# 1) light-normalized (rules/dialect/intent analysis)
corpus_light: List[str] = [norm_ar(d.get("text", "")) for d in docs]
# 2) index-normalized (retrieval only). Flags via env for easy A/B:
INDEX_STOPWORDS = os.getenv("INDEX_STOPWORDS", "1") == "1"
INDEX_STEM      = os.getenv("INDEX_STEM", "0") == "1"
corpus_norm: List[str] = [
    norm_ar_index(d.get("text", ""), stopwords=INDEX_STOPWORDS, stem=INDEX_STEM)
    for d in docs
]

# -------------------------
# TF-IDF retriever (CHAR N-GRAMS + normalized)
# -------------------------

# Try to load persisted artifacts for faster startup; fallback to fit
vectorizer = None
X = None
try:
    import joblib
    vec_p = RETRIEVER_DIR / "tfidf_vectorizer.joblib"
    X_p = RETRIEVER_DIR / "X_tfidf.joblib"
    if vec_p.exists() and X_p.exists():
        vectorizer = joblib.load(vec_p)
        X = joblib.load(X_p)
except Exception:
    vectorizer = None
    X = None

if vectorizer is None or X is None:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=1,
        max_features=250_000,
    )
    X = vectorizer.fit_transform(corpus_norm)



# -------------------------
# Optional tiny THEME classifier (auto-load if present)
# -------------------------
theme_clf = None
try:
    if THEME_MODEL_PATH.exists():
        import joblib
        theme_clf = joblib.load(THEME_MODEL_PATH)
except Exception:
    theme_clf = None

def guess_theme_ml_with_prob_and_rule(text: str) -> Tuple[str, Optional[float], Tuple[Optional[str], Optional[str]]]:
    """Try ML; always return rule match info for transparency."""
    theme_rule, pat, span = guess_theme_rules_with_match(text)
    if theme_clf is None:
        return theme_rule, None, (pat, span)
    try:
        probs = theme_clf.predict_proba([text])[0]
        labels = theme_clf.classes_
        i = probs.argmax()
        return labels[i], float(probs[i]), (pat, span)
    except Exception:
        return theme_rule, None, (pat, span)

# -------------------------
# Dialect detection
# Preferred: HF BERT in models/dialect_hf/
# Fallbacks: joblib LR model -> k-NN vote -> heuristics
# -------------------------
hf_dialect = None  # (model, tok, id2label, device)
try:
    if HF_DIALECT_DIR.exists():
        import torch  # type: ignore[import]
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore[import]
        tok = AutoTokenizer.from_pretrained(str(HF_DIALECT_DIR))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(HF_DIALECT_DIR))
        # ensure device is set and the model is moved to it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
        mdl.eval()
        with open(HF_DIALECT_DIR / "label_map.json", encoding="utf-8") as f:
            m = json.load(f)
        id2label = {int(k): v for k, v in m["id2label"].items()}
        hf_dialect = (mdl, tok, id2label, device)
except Exception:
    hf_dialect = None

dialect_clf = None
try:
    if DIALECT_MODEL_PATH.exists():
        import joblib
        dialect_clf = joblib.load(DIALECT_MODEL_PATH)
except Exception:
    dialect_clf = None

Hijazi_MARKERS = [
    r"(?<!\S)فين(?!\S)",
    r"(?<!\S)لسه(?!\S)",
    r"(?<!\S)دحين(?!\S)",
    r"(?<!\S)(?:ايش|إيش)(?!\S)",
    r"(?<!\S)مره(?!\S)",
    r"(?<!\S)حق(?:ي|ك|هم)?(?!\S)",
    r"(?<!\S)ليش(?!\S)",
]

def guess_dialect_heuristic(text: str) -> Tuple[str, float, Dict]:
    t = norm_ar(text)
    hits = [p for p in Hijazi_MARKERS if re.search(p, t)]
    if hits:
        return "حجازي", 0.60, {"method": "heuristic", "markers_hit": hits}
    return "عربي (غير محدد)", 0.0, {"method": "heuristic", "markers_hit": hits}

def guess_dialect_knn(text: str, k: int = 7) -> Tuple[str, float, Dict]:
    """No training: use TF-IDF char n-grams nearest neighbors."""
    q = norm_ar(text)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, X)[0]
    top = sims.argsort()[::-1][:k]
    counts: Dict[str, float] = {}
    total = 0.0
    for i in top:
        d = docs[i]
        lab = d.get("dialect") or "عربي (غير محدد)"
        w = float(sims[i])
        counts[lab] = counts.get(lab, 0.0) + w
        total += w
    if total <= 1e-9:
        return "عربي (غير محدد)", 0.0, {"method": "knn", "neighbors": counts}
    label, weight = max(counts.items(), key=lambda kv: kv[1])
    conf = weight / total
    return label, float(conf), {"method": "knn", "neighbors": counts}

def guess_dialect_smart(text: str) -> Tuple[str, Optional[float], Dict]:
    # 0) HF BERT (preferred if present)
    if hf_dialect is not None:
        try:
            import torch  # type: ignore[import]
            mdl, tok, id2label, device = hf_dialect
            tokens = tok(norm_ar(text), truncation=True, max_length=128, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                logits = mdl(**tokens).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            i = int(probs.argmax())
            return id2label[i], float(probs[i]), {"method": "hf"}
        except Exception:
            pass
    # 1) joblib LR model
    if dialect_clf is not None:
        try:
            probs = dialect_clf.predict_proba([text])[0]
            labels = dialect_clf.classes_
            i = probs.argmax()
            return labels[i], float(probs[i]), {"method": "ml"}
        except Exception:
            pass
    # 2) k-NN vote over corpus
    lab, conf, dbg = guess_dialect_knn(text)
    if lab != "عربي (غير محدد)" and conf >= 0.45:
        return lab, conf, dbg
    # 3) Heuristic fallback
    return guess_dialect_heuristic(text)

# -------------------------
# Keyphrases (Arabic-only)
# -------------------------
def keyphrases(text: str, top_k: int = 5) -> List[str]:
    t = norm_ar(text)
    ws = re.findall(r"[\u0600-\u06FF]{3,}", t)
    ws = sorted(set(ws), key=lambda w: (-len(w), w))
    return ws[:top_k]

# -------------------------
# Query expansion for Arabic cultural intents
# -------------------------
EXPAND_MAP = {
    "اعتذار": ["اعتذر", "تعتذر", "ترضى", "رضا", "صفح", "سمح", "عتاب", "خصام", "تصالح", "الصفا"],
    "غزلي":   ["حب", "غرام", "عشق", "حبيب", "شوق", "قلب", "ود", "وصال"],
    "وطني":   ["سعود", "رايه", "علم", "موطني", "بلاد", "فخر", "مجدي", "وطن"],
    "بيت":    ["بيت شعر", "شطر", "قصيده", "سطر"],
}
GENERIC_POETRY = re.compile(r"(?:بيت(?:\s*شعر)?|قوافي|قواف|قافية)")

def expand_query(q: str) -> Tuple[str, List[re.Pattern]]:
    """Return normalized expanded query and compiled intent regexes."""
    t = norm_ar(q)
    words = set(re.findall(r"[\u0600-\u06FF]+", t))
    extra_terms = []
    for k, syns in EXPAND_MAP.items():
        if k in words:
            extra_terms += syns
    theme_hint, _, _ = guess_theme_rules_with_match(t)
    if theme_hint != "اخرى":
        extra_terms.append(theme_hint)
    expanded = t + (" " + " ".join(extra_terms) if extra_terms else "")
    intent_regexes = [
        re.compile(rf"(?<!\S){re.escape(norm_ar(term))}(?!\S)") for term in set(extra_terms)
    ]
    return expanded, intent_regexes

def count_intent_hits(text_norm: str, intent_regexes: List[re.Pattern]) -> int:
    return sum(1 for rgx in intent_regexes if rgx.search(text_norm))

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Mini Al-Jazari API", version="0.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Schemas
# -------------------------
class ClassifyIn(BaseModel):
    text: str

class AskIn(BaseModel):
    question: str
    k: int = 3
    show_sources: bool = True
    # optional knobs:
    theme_hint: Optional[str] = None      # "غزل" | "وطنية" | "رياضية" | "دينية"
    dialect_filter: Optional[str] = None  # "حجازي" | "نجدي" | "شمالي" | "جنوبي"
    strict: bool = False
    min_intent_hits: int = 0

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root() -> dict:
    return {"ok": True, "message": "Mini Al-Jazari API running", "docs_count": len(docs)}

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "utf8": True, "docs_count": len(docs)}

@app.post("/classify")
def classify(inp: ClassifyIn):
    raw = (inp.text or "").strip()
    theme, prob, (pat, span) = guess_theme_ml_with_prob_and_rule(raw)
    dial, d_prob, d_dbg = guess_dialect_smart(raw)

    payload = {
        "dialect": dial,
        "dialect_confidence": round(d_prob, 3) if d_prob is not None else None,
        "theme": theme,
        "confidence": round(prob, 3) if prob is not None else None,
        "keyphrases": keyphrases(raw),
        "analysis": {
            "normalized_text": norm_ar(raw),
            "matched_rule_pattern": pat,
            "matched_text_span": span,
            "dialect_debug": d_dbg,
        },
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )

@app.post("/ask")
def ask(inp: AskIn):
    q_raw = (inp.question or "").strip()
    if not q_raw:
        payload = {"answer": "اكتب سؤالك من فضلك.", "sources": []}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            media_type="application/json; charset=utf-8",
        )

    # Normalize + expand + intended theme
    q_norm, intent_rgx = expand_query(q_raw)  
    intended_theme = (inp.theme_hint or guess_theme_rules_with_match(q_norm)[0])

    # Retrieve with CHAR n-grams
    # Use index-normalized query for TF-IDF search
    q_norm_index = norm_ar_index(q_raw, stopwords=INDEX_STOPWORDS, stem=INDEX_STEM)
    q_vec = vectorizer.transform([q_norm_index])
    sims = cosine_similarity(q_vec, X)[0]

    # Theme/intent-aware scoring
    def themed_score(i: int) -> float:
        base = float(sims[i])
        doc = docs[i]
        doc_theme = doc.get("theme", "") or "اخرى"
        text_norm = corpus_light[i]

        hits = count_intent_hits(text_norm, intent_rgx)
        if hits:
            base *= (1.0 + 0.25 * hits)
        if intended_theme != "اخرى" and doc_theme == intended_theme:
            base *= 1.25
        if intended_theme == "غزل" and doc_theme == "دينية":
            base *= 0.65
        if GENERIC_POETRY.search(text_norm) and hits == 0:
            base *= 0.60
        return base

    idx_sorted = sorted(range(len(sims)), key=lambda i: themed_score(i), reverse=True)

    # candidate pool (oversample for later pruning)
    pool_size = max(inp.k * 8, 40)
    candidates = []
    for i in idx_sorted[:pool_size]:
        d = docs[i]
        # optional dialect filter
        if inp.dialect_filter and (d.get("dialect", "").lower() != inp.dialect_filter.lower()):
            continue

        text_norm = corpus_light[i]
        intent_hits = count_intent_hits(text_norm, intent_rgx)
        theme_match = (d.get("theme", "") == intended_theme)

        if inp.strict:
            if intended_theme != "اخرى" and not theme_match:
                continue
            if intent_hits < max(0, int(inp.min_intent_hits)):
                continue

        preview = d.get("text", "")
        if len(preview) > 120:
            preview = preview[:120] + "…"

        candidates.append({
            "i": i,
            "text": preview,
            "title": d.get("title", ""),
            "writer": d.get("writer", ""),
            "theme": d.get("theme", ""),
            "dialect": d.get("dialect", ""),
            "score": themed_score(i),
            "intent_hits": intent_hits,
            "theme_match": theme_match,
        })

    # final sort & top-k
    candidates.sort(key=lambda r: (r["intent_hits"], r["theme_match"], r["score"]), reverse=True)
    hits = candidates[:inp.k]
    if len(hits) < inp.k and not inp.strict:
        extra = sorted(candidates, key=lambda r: r["score"], reverse=True)[:inp.k]
        seen = {id(h) for h in hits}
        for e in extra:
            if id(e) not in seen:
                hits.append(e)
            if len(hits) >= inp.k:
                break

    lines = []
    for h in hits:
        who = (h["writer"] or "").strip()
        title = f"«{h['title']}»" if h["title"] else "مقطع"
        lines.append(f"- [{h['theme'] or 'غير محدد'}] {title} — {who}\n> {h['text']}")

    answer = "وجدت هذه المقاطع الأقرب لسؤالك:\n" + "\n".join(lines)
    payload = {
        "answer": answer,
        "sources": [{k: v for k, v in h.items() if k != "i"} for h in hits],
        "debug": {
            "intended_theme": intended_theme,
            "dialect_filter": inp.dialect_filter,
            "strict": inp.strict,
            "min_intent_hits": inp.min_intent_hits,
        },
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )

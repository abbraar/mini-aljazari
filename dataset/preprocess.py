# dataset/preprocess.py
import json, re, random, pathlib
from collections import defaultdict

from app.shared.text_utils import norm_ar, THEME_RULES, tag_theme

RAW = pathlib.Path("dataset/raw/SADSLyC.json")
PROC_DIR = pathlib.Path("dataset/processed")
SPLIT_DIR = pathlib.Path("dataset/splits")
RETRIEVER_DIR = pathlib.Path("retriever")
PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)

# Normalization and theme rules are imported from app.shared.text_utils

# THEME_RULES and tag_theme imported


def _ensure_dict(x):
    """If x is a JSON string, parse it; else return as-is."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x

def load_rows():
    # load root
    with open(RAW, encoding="utf-8") as f:
        root = json.load(f)

    rows = []

    if isinstance(root, dict):
        # expect dialect buckets: {"Hijazi": [ {...}, {...} ], "Najdi": [ ... ], ...}
        for dialect_key, items in root.items():
            if not isinstance(items, list):
                continue
            for it in items:
                it = _ensure_dict(it)
                if not isinstance(it, dict):
                    continue
                text   = it.get("Lyrics") or it.get("text") or ""
                title  = it.get("Title")  or ""
                writer = it.get("Writer") or ""
                dialect = it.get("Dialect") or dialect_key or "Arabic"
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "title": title,
                    "writer": writer,
                    "dialect": dialect
                })
    elif isinstance(root, list):
        # flat list of dicts (or strings containing dicts)
        for it in root:
            it = _ensure_dict(it)
            if not isinstance(it, dict):
                continue
            text   = it.get("Lyrics") or it.get("text") or ""
            title  = it.get("Title")  or ""
            writer = it.get("Writer") or ""
            dialect = it.get("Dialect") or "Arabic"
            if not text:
                continue
            rows.append({
                "text": text,
                "title": title,
                "writer": writer,
                "dialect": dialect
            })
    else:
        raise ValueError("Unsupported JSON root structure. Expected dict or list.")

    return rows

def main():
    raw_rows = load_rows()

    # normalize + dedup
    seen = set()
    clean_rows = []
    texts_norm = []
    for r in raw_rows:
        clean_text = norm_ar(r["text"])
        if len(clean_text) < 3:
            continue
        key = (clean_text, r["title"], r["writer"], r["dialect"])
        if key in seen:
            continue
        seen.add(key)
        clean_rows.append({
            "text": clean_text,
            "title": r["title"],
            "writer": r["writer"],
            "dialect": r["dialect"],
            "theme": tag_theme(clean_text),
        })
        texts_norm.append(clean_text)

    # write processed jsonl
    out_path = PROC_DIR / "lyrics_clean.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stratified splits by theme
    by_theme = defaultdict(list)
    for r in clean_rows:
        by_theme[r["theme"]].append(r)

    train, val, test = [], [], []
    random.seed(7)
    for theme, lst in by_theme.items():
        lst = lst[:]  # copy
        random.shuffle(lst)
        n = len(lst)
        n_val = max(1, int(0.1 * n))
        n_test = max(1, int(0.1 * n))
        val.extend(lst[:n_val])
        test.extend(lst[n_val:n_val+n_test])
        train.extend(lst[n_val+n_test:])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(SPLIT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # simple summary
    by_dialect = defaultdict(int)
    for r in clean_rows:
        by_dialect[r["dialect"]] += 1

    print({
        "total": len(clean_rows),
        "by_dialect": dict(by_dialect),
        "by_theme": {k: len(v) for k, v in by_theme.items()},
        "splits": {"train": len(train), "val": len(val), "test": len(test)}
    })

    # --- Persist TF-IDF retriever artifacts for API startup speed ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib
        print("[preprocess] Fitting TF-IDF vectorizer on corpus (char 3-6)...")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=1,
            max_features=250_000,
        )
        X = vectorizer.fit_transform(texts_norm)
        joblib.dump(vectorizer, RETRIEVER_DIR / "tfidf_vectorizer.joblib")
        joblib.dump(X, RETRIEVER_DIR / "X_tfidf.joblib")
        # Also persist a light corpus file to keep original docs aligned
        with open(RETRIEVER_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
            for r in clean_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[preprocess] Saved retriever artifacts to retriever/ ")
    except Exception as e:
        print("[preprocess][warn] Failed to persist TF-IDF artifacts:", e)

if __name__ == "__main__":
    main()

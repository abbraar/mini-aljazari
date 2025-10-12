# dataset/preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocess the SADSLyC dataset:
- Load raw JSON (dict-of-lists by dialect OR flat list of dicts/JSON strings)
- Normalize text (light + index variants)
- Tag theme via rules, with optional Gemini fallback (tag_theme_smart)
- Deduplicate
- Write processed corpus (JSONL) and stratified train/val/test splits
- Fit and persist TF-IDF (char 3-6) retriever artifacts aligned with API settings

Usage:
    python dataset/preprocess.py \
        --raw dataset/raw/SADSLyC.json \
        --proc-dir dataset/processed \
        --split-dir dataset/splits \
        --retriever-dir retriever \
        --seed 7 \
        --val-ratio 0.10 \
        --test-ratio 0.10 \
        --use-gemini                # <- enable Gemini fallback

You can also point to your uploaded file, e.g.:
    python dataset/preprocess.py --raw /mnt/data/SADSLyC.json --use-gemini
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from app.shared.text_utils import (
    norm_ar,
    norm_ar_index,
    tag_theme,         # rules-only (kept for fallback or if --use-gemini not set)
    tag_theme_smart,   # rules + optional Gemini
)

# ---------------------------- CLI ----------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SADSLyC dataset.")
    p.add_argument("--raw", type=pathlib.Path, default=pathlib.Path("dataset/raw/SADSLyC.json"),
                   help="Path to raw SADSLyC JSON (dict or list).")
    p.add_argument("--proc-dir", type=pathlib.Path, default=pathlib.Path("dataset/processed"),
                   help="Output directory for processed JSONL.")
    p.add_argument("--split-dir", type=pathlib.Path, default=pathlib.Path("dataset/splits"),
                   help="Output directory for train/val/test splits.")
    p.add_argument("--retriever-dir", type=pathlib.Path, default=pathlib.Path("retriever"),
                   help="Output directory for TF-IDF artifacts.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for deterministic splits.")
    p.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio per theme.")
    p.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio per theme.")
    p.add_argument("--index-stopwords", action="store_true",
                   help="Apply stopword removal in norm_ar_index (must match API flags).")
    p.add_argument("--index-stem", action="store_true",
                   help="Apply stemming in norm_ar_index (must match API flags).")
    p.add_argument("--use-gemini", action="store_true",
                   help="Use Gemini fallback when rules return 'اخرى'. Requires GEMINI_API_KEY.")
    p.add_argument("--min-llm-score", type=float, default=0.55,
                   help="Minimum Gemini confidence (0..1) to accept its label.")
    return p.parse_args()

# ----------------------- Utilities -----------------------

def ensure_dirs(*paths: pathlib.Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def _ensure_dict(x: Any) -> Optional[Dict[str, Any]]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def _coalesce(*vals: Any, default: str = "") -> str:
    for v in vals:
        if v:
            return str(v)
    return default

def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------ Loading ------------------------

def load_rows(raw_path: pathlib.Path) -> List[Dict[str, str]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    with open(raw_path, encoding="utf-8") as f:
        root = json.load(f)

    rows: List[Dict[str, str]] = []

    if isinstance(root, dict):
        for dialect_key, items in root.items():
            if not isinstance(items, list):
                continue
            for it in items:
                it = _ensure_dict(it)
                if not it:
                    continue
                text   = _coalesce(it.get("Lyrics"), it.get("text"))
                title  = _coalesce(it.get("Title"))
                writer = _coalesce(it.get("Writer"))
                dialect = _coalesce(it.get("Dialect"), dialect_key, default="Arabic")
                if not text.strip():
                    continue
                rows.append({"text": text, "title": title, "writer": writer, "dialect": dialect})
    elif isinstance(root, list):
        for it in root:
            it = _ensure_dict(it) or {}
            text   = _coalesce(it.get("Lyrics"), it.get("text"))
            title  = _coalesce(it.get("Title"))
            writer = _coalesce(it.get("Writer"))
            dialect = _coalesce(it.get("Dialect"), default="Arabic")
            if not text.strip():
                continue
            rows.append({"text": text, "title": title, "writer": writer, "dialect": dialect})
    else:
        raise ValueError("Unsupported JSON root structure. Expected dict or list.")

    return rows

# ---------------------- Preprocess ----------------------

def preprocess_rows(
    raw_rows: List[Dict[str, str]],
    index_stopwords: bool,
    index_stem: bool,
    use_gemini: bool = False,
    min_llm_score: float = 0.55,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Normalize, deduplicate, and tag themes.
    Returns:
      clean_rows   : list of canonical records
      texts_light  : light-normalized texts (for summaries)
      texts_index  : index-normalized texts (for TF-IDF fitting)
    """
    seen = set()
    clean_rows: List[Dict[str, Any]] = []
    texts_light: List[str] = []
    texts_index: List[str] = []

    for r in raw_rows:
        light = norm_ar(r["text"])
        if len(light) < 3:
            continue

        key = (light, r["title"], r["writer"], r["dialect"])
        if key in seen:
            continue
        seen.add(key)

        # Rules first; if requested, fall back to Gemini when rules say 'اخرى'
        if use_gemini:
            theme = tag_theme_smart(light, use_llm_fallback=True, min_llm_score=min_llm_score)
        else:
            theme = tag_theme(light)

        idx_norm = norm_ar_index(r["text"], stopwords=index_stopwords, stem=index_stem)

        clean_rows.append({
            "text": light,           # canonical light text
            "title": r["title"],
            "writer": r["writer"],
            "dialect": r["dialect"],
            "theme": theme,
            # keep both variants for transparency / downstream tasks:
            "text_light": light,
            "text_index": idx_norm,
        })
        texts_light.append(light)
        texts_index.append(idx_norm)

    return clean_rows, texts_light, texts_index

# ---------------------- Splitting ----------------------

def stratified_splits_by_theme(
    rows: List[Dict[str, Any]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_theme: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_theme[r["theme"]].append(r)

    random.seed(seed)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for theme, lst in by_theme.items():
        if not lst:
            continue
        lst = lst[:]  # copy
        random.shuffle(lst)

        n = len(lst)
        n_val = max(1, int(round(val_ratio * n))) if n >= 10 else max(0, int(val_ratio * n))
        n_test = max(1, int(round(test_ratio * n))) if n >= 10 else max(0, int(test_ratio * n))

        n_val = min(n_val, n)
        n_test = min(n_test, max(0, n - n_val))
        n_train = max(0, n - n_val - n_test)

        val.extend(lst[:n_val])
        test.extend(lst[n_val:n_val + n_test])
        train.extend(lst[n_val + n_test: n_val + n_test + n_train])

    return train, val, test

# ---------------- TF-IDF Artifacts ----------------

def persist_tfidf_artifacts(
    texts_index: List[str],
    clean_rows: List[Dict[str, Any]],
    retriever_dir: pathlib.Path,
) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        print("[preprocess] Fitting TF-IDF vectorizer on INDEX-normalized corpus (char 3-6)...")

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=1,
            max_features=250_000,
        )
        X = vectorizer.fit_transform(texts_index)

        joblib.dump(vectorizer, retriever_dir / "tfidf_vectorizer.joblib")
        joblib.dump(X, retriever_dir / "X_tfidf.joblib")

        with open(retriever_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
            for r in clean_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("[preprocess] Saved retriever artifacts to retriever/")
    except Exception as e:
        print("[preprocess][warn] Failed to persist TF-IDF artifacts:", e)

# ------------------------- Main -------------------------

def main() -> None:
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    args = build_args()

    # Keep flags aligned with app/api.py (env overrides also supported)
    index_stopwords = bool(
        os.getenv("INDEX_STOPWORDS", "1" if args.index_stopwords else "0") == "1"
    )
    index_stem = bool(
        os.getenv("INDEX_STEM", "1" if args.index_stem else "0") == "1"
    )

    ensure_dirs(args.proc_dir, args.split_dir, args.retriever_dir)

    raw_rows = load_rows(args.raw)
    clean_rows, texts_light, texts_index = preprocess_rows(
        raw_rows,
        index_stopwords=index_stopwords,
        index_stem=index_stem,
        use_gemini=args.use_gemini,
        min_llm_score=args.min_llm_score,
    )

    # Write processed corpus
    out_path = args.proc_dir / "lyrics_clean.jsonl"
    write_jsonl(out_path, clean_rows)

    # Splits
    train, val, test = stratified_splits_by_theme(
        clean_rows, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    write_jsonl(args.split_dir / "train.jsonl", train)
    write_jsonl(args.split_dir / "val.jsonl", val)
    write_jsonl(args.split_dir / "test.jsonl", test)

    # Simple summaries
    by_theme = defaultdict(int)
    by_dialect = defaultdict(int)
    for r in clean_rows:
        by_theme[r["theme"]] += 1
        by_dialect[r["dialect"]] += 1

    summary = {
        "total": len(clean_rows),
        "by_dialect": dict(by_dialect),
        "by_theme": dict(by_theme),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "index_stopwords": index_stopwords,
        "index_stem": index_stem,
        "used_gemini": args.use_gemini,
        "min_llm_score": args.min_llm_score,
    }
    print(summary)

    with open(args.proc_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Persist TF-IDF
    persist_tfidf_artifacts(texts_index, clean_rows, args.retriever_dir)

if __name__ == "__main__":
    main()

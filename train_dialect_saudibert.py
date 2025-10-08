# train_dialect_saudibert.py
# Stratified split + minority oversampling + BERT fine-tune
# Adds: epoch-wise eval, best-model saving, fp16/bf16 (optional), per-label metrics & confusion matrix

import os
import json
import pathlib
from datetime import datetime
import random
from collections import Counter, defaultdict

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import numpy as np
import torch

from app.shared.text_utils import norm_ar  # your normalizer

# ---- optional metrics (evaluate) ----
try:
    import evaluate  # pip install evaluate
except Exception:
    evaluate = None

DATA = pathlib.Path("dataset/processed/lyrics_clean.jsonl")
# Save to a fresh timestamped directory to avoid overwriting files that may be mmap'ed by a running API
OUT_DIR = pathlib.Path("models") / f"dialect_hf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
set_seed(42)
random.seed(42)
np.random.seed(42)

# ---- load JSONL ----
texts, labels = [], []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = (r.get("text") or "").strip()
        d = (r.get("dialect") or "عربي (غير محدد)").strip()
        if len(t) >= 3:
            texts.append(norm_ar(t))
            labels.append(d)

if not texts:
    raise RuntimeError(f"No data found in {DATA}")

# Label maps
label_list = sorted(set(labels))
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}

# ---- stratified split (80/10/10) ----
X_train, X_tmp, y_train, y_tmp = train_test_split(
    texts, labels, test_size=0.20, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

# ---- oversample minorities in TRAIN ----
ctr = Counter(y_train)
target = min(max(ctr.values()), 2000)  # cap for speed; increase if needed
rng = random.Random(42)

def oversample(X, Y):
    buckets = defaultdict(list)
    for x, lab in zip(X, Y):
        buckets[lab].append(x)
    Xo, Yo = [], []
    for lab in label_list:
        pool = buckets.get(lab, [])
        if not pool:
            continue
        if len(pool) >= target:
            chosen = rng.sample(pool, target)
        else:
            chosen = pool[:] + [rng.choice(pool) for _ in range(target - len(pool))]
        Xo.extend(chosen)
        Yo.extend([lab] * len(chosen))
    return Xo, Yo

X_train, y_train = oversample(X_train, y_train)

def mk_ds(xs, ys):
    return Dataset.from_dict({"text": xs, "label": [label2id[y] for y in ys]})

ds = DatasetDict(
    {
        "train": mk_ds(X_train, y_train),
        "validation": mk_ds(X_val, y_val),
        "test": mk_ds(X_test, y_test),
    }
)

# ---- tokenizer/model ----
MODEL_NAME = os.environ.get("MODEL_NAME", "faisalq/SaudiBERT")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=128)

ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

# ---- metrics ----
if evaluate is not None:
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
            "precision_macro": precision.compute(predictions=preds, references=p.label_ids, average="macro")["precision"],
            "recall_macro": recall.compute(predictions=preds, references=p.label_ids, average="macro")["recall"],
        }
else:
    compute_metrics = None

# ---- TrainingArguments (new-first, legacy fallback) ----
use_fp16 = torch.cuda.is_available()
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8  # Ampere+

try:
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro" if compute_metrics else "eval_loss",
        greater_is_better=True if compute_metrics else False,
        logging_steps=50,
        save_total_limit=2,
        save_safetensors=False,
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,  # prefers bf16 on supported GPUs
        report_to="none",  # set "tensorboard" if you want TB logs
        seed=42,
    )
except TypeError:
    # Legacy-safe minimal config
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_steps=50,
        save_total_limit=2,
        save_safetensors=False,
        seed=42,
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ---- train ----
trainer.train()

# ---- evaluate (validation + test) ----
def evaluate_split(name, ds_split):
    try:
        metrics = trainer.evaluate(ds_split)
        print(f"{name.upper()}:", metrics)
        # detailed per-label report + confusion matrix
        preds = np.argmax(trainer.predict(ds_split).predictions, axis=1)
        refs = np.array(ds_split["label"])
        print(f"\n{name.upper()} classification report:")
        print(classification_report(refs, preds, target_names=label_list, digits=4))
        print(f"{name.upper()} confusion matrix:")
        print(confusion_matrix(refs, preds))
        # save metrics to disk
        with open(OUT_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] {name} evaluation skipped:", e)

evaluate_split("validation", ds_tok["validation"])
evaluate_split("test", ds_tok["test"])

# ---- save model + tokenizer + label map ----
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print("Saved to:", OUT_DIR)
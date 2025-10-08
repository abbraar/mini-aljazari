import os
import json
import pathlib
import random
from collections import Counter, defaultdict

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import numpy as np
from transformers import set_seed
from app.shared.text_utils import norm_ar

# metrics (optional): if 'evaluate' isn't installed, we'll skip metrics during training
try:
    import evaluate
except Exception:
    evaluate = None

DATA = pathlib.Path("dataset/processed/lyrics_clean.jsonl")
OUT_DIR = pathlib.Path("models/dialect_hf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# normalization imported from shared utils

# Seed for reproducibility
set_seed(42)

# --- load ---
texts, labels = [], []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r.get("text") or ""
        d = (r.get("dialect") or "عربي (غير محدد)").strip()
        if len(t) >= 3:
            texts.append(norm_ar(t))
            labels.append(d)

label_list = sorted(set(labels))
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}

# stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    texts, labels, test_size=0.20, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

# oversample minorities in TRAIN
ctr = Counter(y_train)
target = min(max(ctr.values()), 2000)  # cap for speed; raise if you want
rng = random.Random(42)
def oversample(X, y):
    buckets = defaultdict(list)
    for x, lab in zip(X, y):
        buckets[lab].append(x)
    Xo, yo = [], []
    for lab in label_list:
        pool = buckets.get(lab, [])
        if not pool:
            continue
        if len(pool) >= target:
            chosen = rng.sample(pool, target)
        else:
            chosen = pool[:] + [rng.choice(pool) for _ in range(target - len(pool))]
        Xo.extend(chosen)
        yo.extend([lab] * len(chosen))
    return Xo, yo

X_train, y_train = oversample(X_train, y_train)

def mk_ds(xs, ys):
    return Dataset.from_dict({"text": xs, "label": [label2id[y] for y in ys]})

ds = DatasetDict({
    "train": mk_ds(X_train, y_train),
    "validation": mk_ds(X_val, y_val),
    "test": mk_ds(X_test, y_test)
})

MODEL_NAME = os.environ.get("MODEL_NAME", "faisalq/SaudiBERT")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=128)

ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

# metrics during training (optional)
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

# --- TrainingArguments (legacy-safe; NO evaluation_strategy/save_strategy) ---
args = TrainingArguments(
    output_dir=str(OUT_DIR/"runs"),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],  # safe on old/new versions
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Manual evaluation (works on all transformers versions)
try:
    val_metrics = trainer.evaluate(ds_tok["validation"])
    test_metrics = trainer.evaluate(ds_tok["test"])
    print("VALIDATION:", val_metrics)
    print("TEST:", test_metrics)
except Exception as e:
    print("[warn] evaluation skipped:", e)

# save model + tokenizer + label map
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
with open(OUT_DIR/"label_map.json","w",encoding="utf-8") as f:
    json.dump({"label2id":label2id,"id2label":id2label}, f, ensure_ascii=False, indent=2)
print("Saved to:", OUT_DIR)

import argparse

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

from prepare import load

MODEL = "distilbert-base-uncased"

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
args = ap.parse_args()

train, val, _, labels = load()
tok = AutoTokenizer.from_pretrained(MODEL)


def to_ds(df):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)


f1 = evaluate.load("f1")


def metrics(p):
    return f1.compute(predictions=np.argmax(p.predictions, -1), references=p.label_ids, average="macro")


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
targs = TrainingArguments("outputs", num_train_epochs=args.epochs, learning_rate=args.lr, per_device_train_batch_size=32,
                          eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                          metric_for_best_model="f1", weight_decay=0.01, report_to=[])
trainer = Trainer(model=model, args=targs, train_dataset=to_ds(train), eval_dataset=to_ds(val),
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics, tokenizer=tok)
trainer.train()
trainer.save_model("outputs/best")

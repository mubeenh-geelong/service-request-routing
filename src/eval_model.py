import sys

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare import load

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/best"
_, _, test, labels = load()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(test), 64):
        batch = tok(list(test.text[i:i + 64]), truncation=True, max_length=256, padding=True, return_tensors="pt")
        preds += model(**batch).logits.argmax(-1).tolist()

print(classification_report(test.label, preds, target_names=labels, digits=3))
cm = pd.DataFrame(confusion_matrix(test.label, preds), index=labels, columns=labels)
cm.to_csv("outputs/confusion_matrix.csv")
print("confusion matrix saved to outputs/confusion_matrix.csv")

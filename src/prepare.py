import re

import pandas as pd
from sklearn.model_selection import train_test_split

SIGNOFF = re.compile(r"\n\s*(kind regards|regards|thanks|thank you|cheers|sent from my \w+)[\s\S]*$", re.I)
QUOTED = re.compile(r"\n\s*(on .{5,80} wrote:|-{2,} ?original message ?-{2,}|from: .+\nsent: )[\s\S]*$", re.I)


def clean(text):
    text = str(text).replace("\r\n", "\n")
    text = QUOTED.sub("", text)
    text = SIGNOFF.sub("", text)
    text = re.sub(r"\S+@\S+", "[email]", text)
    text = re.sub(r"\b\d{8,}\b", "[number]", text)
    return re.sub(r"\s+", " ", text).strip()


def load(path="data/requests.csv", test_size=3000, seed=42):
    df = pd.read_csv(path)
    df["text"] = (df.subject.fillna("") + ". " + df.body.fillna("")).map(clean)
    df = df[df.text.str.len() > 15]
    df = df.drop_duplicates(subset="text")
    labels = sorted(df.category.unique())
    df["label"] = df.category.map({c: i for i, c in enumerate(labels)})
    test_size = min(test_size, len(df) // 5)
    train, test = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
    train, val = train_test_split(train, test_size=0.1, stratify=train.label, random_state=seed)
    return train, val, test, labels

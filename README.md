# service-request-routing

Deakin capstone (Master of Applied AI, Trimester 1 2026). Team of four.

Our industry partner gets customer service requests through a web form and a person reads each one to decide which team
should handle it. We fine-tuned DistilBERT to suggest the team from the request text, over 18 categories.

Held out test set of 3,000 requests: macro F1 0.79, accuracy 0.86. The weakest categories were the two billing ones, which
people also mix up.

The partner's data is not in this repo and can't be. `src/make_fake_data.py` writes a small fake CSV with the same columns so
the code runs end to end.

## Run

```
pip install -r requirements.txt
python src/make_fake_data.py          # or put the real export at data/requests.csv
python src/train.py --epochs 3
python src/eval_model.py outputs/best
```

## My part

I did the data cleaning (`src/prepare.py`: deduplicating forwarded emails, stripping signatures and reply chains) and the
evaluation script. Training was shared. The confusion matrix and per class report went into our final report.

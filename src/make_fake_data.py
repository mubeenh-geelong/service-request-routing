import random
from pathlib import Path

import pandas as pd

CATS = ["billing_query", "billing_dispute", "account_access", "address_change", "complaint", "feedback",
        "new_service", "cancel_service", "fault_report", "outage", "payment_plan", "refund", "privacy",
        "document_request", "appointment", "hardship", "general", "other_team"]

random.seed(0)
rows = []
for i in range(4000):
    c = random.choice(CATS)
    rows.append({"id": i, "subject": c.replace("_", " "), "body": f"Hi, I have a question about {c.replace('_', ' ')}. "
                 f"Reference {random.randint(10**8, 10**9)}.\n\nKind regards\nA customer", "category": c})
Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/requests.csv", index=False)
print("wrote data/requests.csv")

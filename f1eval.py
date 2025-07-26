import pandas as pd
import evaluate

# Load the squad metric
squad_metric = evaluate.load("squad")

# Read your data (replace with your actual CSV or DataFrame)
df = pd.read_csv("your_file.csv")  # or df = your_dataframe

# Assume your DataFrame has columns: 'preds' and 'references'
predictions = []
references = []

for idx, row in df.iterrows():
    predictions.append({
        "id": str(idx),
        "prediction_text": row["preds"]
    })
    references.append({
        "id": str(idx),
        "answers": {
            "text": [row["references"]],  # wrap GT in a list
            "answer_start": [0]  # dummy start (ignored for F1)
        }
    })

# Compute token-level F1 and EM
results = squad_metric.compute(predictions=predictions, references=references)
print("F1 Score:", results["f1"])
print("Exact Match:", results["exact_match"])


-------------------------

import pandas as pd
import string
import re

# 1. Normalize answer (standard SQuAD-style)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 2. Compute token overlap F1
def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(str(prediction)).split()
    gt_tokens = normalize_answer(str(ground_truth)).split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# 3. Load your CSV (update filename if needed)
df = pd.read_csv("your_file.csv")  # <-- replace with actual filename

# 4. Compute row-wise F1 between your two columns
df["f1_score"] = df.apply(
    lambda row: compute_f1(row["diversity_rag_answer_under_optimal"], row["ground_truth"]),
    axis=1
)

# 5. Report results
print("Average F1 Score:", df["f1_score"].mean())

# Optional: save result
df.to_csv("f1_scored_output.csv", index=False)


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

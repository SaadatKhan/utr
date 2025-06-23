
from pathlib import Path
from typing import List, Dict, Union
import json


def load_hotpotqa(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)          # Hotpot root is a list of items

    items: List[Dict] = []
    for row in data:
        # Flatten all sentences of every context entry
        paragraphs = []
        for title, content in row["context"]:
            if isinstance(content, list):
                paragraphs.extend(content)
            else:
                paragraphs.append(content)

        items.append(
            {
                "id": row.get("_id", row.get("id", "")),
                "question": row["question"],
                # Hotpot stores a single answer string; wrap it for consistency
                "answers": [row["answer"]] if isinstance(row["answer"], str) else row["answer"],
                "context_text": "\n\n".join(paragraphs),
            }
        )
    return items

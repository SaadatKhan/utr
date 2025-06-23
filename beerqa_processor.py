import json
from pathlib import Path
from typing import List, Dict, Union


def load_beerqa(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        jj = json.load(f)

    items: List[Dict] = []
    for row in jj["data"]:
        paragraphs = [p[1] for p in row["context"]]          # take text parts
        items.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": row["answers"],
                "context_text": "\n\n".join(paragraphs),
            }
        )
    return items
"""

from typing import List, Dict, Union
from pathlib import Path
import json


def load_beerqa(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        jj = json.load(f)

    items: List[Dict] = []
    for row in jj["data"]:
        paragraphs = []
        for title, content in row["context"]:
            # content may be a list of sentences or already a string
            if isinstance(content, list):
                paragraphs.extend(content)          # add individual sentences
            else:
                paragraphs.append(content)

        items.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": row["answers"] if isinstance(row["answers"], list) else [row["answers"]],
                "context_text": "\n\n".join(paragraphs),
            }
        )
    return items
"""
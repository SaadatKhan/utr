

from pathlib import Path
from typing import List, Dict, Union
import json


# ── helper: flatten answer field ────────────────────────────────────────────
def _flatten_answers(ans_field) -> List[str]:
    """Turn a nested list / string into a flat list[str]."""
    if isinstance(ans_field, str):
        return [ans_field]

    flat: List[str] = []
    stack = list(ans_field)
    while stack:
        elem = stack.pop()
        if isinstance(elem, list):
            stack.extend(elem)
        else:
            flat.append(str(elem))
    return flat[::-1]  # preserve original order


# ── tolerant raw loader ─────────────────────────────────────────────────────
def _load_raw(path: Union[str, Path]) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 1) Try standard JSON first
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass

    # 2) Try ND-JSON (one JSON object per line)
    ndjson = []
    nd_ok = True
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                ndjson.append(json.loads(line))
            except json.JSONDecodeError:
                nd_ok = False
                break
    if nd_ok and ndjson:
        return ndjson

    # 3) Fallback: concatenated {...}{...}
    records, idx, decoder = [], 0, json.JSONDecoder()
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        records.append(obj)
        idx = end
    return records


# ── public loader ───────────────────────────────────────────────────────────
def load_rgb(path: Union[str, Path]) -> List[Dict]:
    raw_items = _load_raw(path)

    items: List[Dict] = []
    for row in raw_items:
        positives = row.get("positive", [])
        negatives = row.get("negative", [])
        context   = positives + negatives

        items.append(
            {
                "id":          str(row["id"]),
                "question":    row["query"],
                "answers":     _flatten_answers(row["answer"]),
                "context_text": "\n\n".join(context),
            }
        )
    return items

# triviaqa_processor.py
"""
TriviaQA loader for the directory layout:

    data/trivia-qa/qa/<file>.json      ← question set
    data/trivia-qa/web/                ← evidence TXT tree

Returned list items all share the schema expected by main.py:
    {
      "id":            str,
      "question":      str,
      "answers":       list[str],
      "context_text":  str
    }
"""

from pathlib import Path
from typing import List, Dict, Union
import json


# ── helpers ────────────────────────────────────────────────────────────────
def _flatten_answers(ans_block) -> List[str]:
    """
    Convert TriviaQA's answer field into a flat list[str].
    """
    vals: List[str] = []
    if isinstance(ans_block, dict):
        vals.append(ans_block.get("Value", ""))
        vals.extend(ans_block.get("Aliases", []))
    elif isinstance(ans_block, list):
        vals.extend(ans_block)
    else:
        vals.append(str(ans_block))
    return [v for v in map(str, vals) if v]


# ── public loader ──────────────────────────────────────────────────────────
def load_triviaqa(
    json_path: Union[str, Path],
    evidence_root: Union[str, Path] | None = None,
) -> List[Dict]:
    """
    Parameters
    ----------
    json_path :  path to the TriviaQA questions JSON (under qa/)
    evidence_root :  root folder with the TXT files (default: sibling 'web/')

    Returns
    -------
    list[dict]  with keys id, question, answers, context_text
    """
    json_path = Path(json_path)

    # Default evidence directory is '../../web/' relative to qa/ file
    if evidence_root is None:
        evidence_root = json_path.parent.parent / "web"
    evidence_root = Path(evidence_root)

    # --- load the question set -----------------------------------------
    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)["Data"]

    items: List[Dict] = []
    for row in raw_data:
        # ----- collect evidence paragraphs -----------------------------
        paragraphs: List[str] = []
        for sr in row.get("SearchResults", []):
            fname = sr.get("Filename")
            if not fname:
                continue

            fpath = evidence_root / fname
            try:
                paragraphs.append(
                    fpath.read_text(encoding="utf-8", errors="ignore").strip()
                )
            except FileNotFoundError:
                # file missing – just skip this evidence item
                continue
            except Exception as e:
                # any other I/O error – warn & skip this file
                print(f"[TriviaQA] warn: cannot read {fpath} — {e}")
                continue

        # If no evidence at all, skip this question
        if not paragraphs:
            print(f"[TriviaQA] skip: no evidence for QuestionId {row['QuestionId']}")
            continue

        # ----- normalise & store ---------------------------------------
        items.append(
            {
                "id":           row["QuestionId"],
                "question":     row["Question"],
                "answers":      _flatten_answers(row["Answer"]),
                "context_text": "\n\n".join(paragraphs),
            }
        )

    return items

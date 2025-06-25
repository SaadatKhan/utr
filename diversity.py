# ── configuration ─────────────────────────────────────────────────────────
JSON_PATH = "data/hotpot_first200.json"
TOP_K     = 5
ALPHA     = 0          # weight on relevance; (1-α) on Euclidean distance
MAX_ITEMS = 100

# ── imports ───────────────────────────────────────────────────────────────
import logging, os, numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import pandas as pd
from hotpotqa_processor import load_hotpotqa
from utils import split_into_chunks
from generator import Generator
from generator_x import GeneratorX

# ── helper: cosine-only selector ──────────────────────────────────────────
def select_topk_chunks(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    embs = embedder.encode([question] + chunks, normalize_embeddings=True)
    sims = np.dot(embs[1:], embs[0])
    idx  = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in idx], sims[idx].tolist()


# ── helper: cosine + Euclidean-distance selector ─────────────────────────
def select_diverse_topk_chunks(
    question: str,
    chunks: List[str],
    embedder: SentenceTransformer,
    top_k: int = 10,
    *,
    alpha: float = 0.7,
):
    """
    score  =  α · cos(query, chunk)
            + (1-α) · dist_Euc(chunk, mean(memory)),
    where distance ∈ [0, 2] for unit vectors.
    Returns: chunks, total, cosine_part, distance_part
    """
    if not chunks or top_k <= 0:
        return [], [], [], []

    embs = embedder.encode([question] + chunks, normalize_embeddings=True)
    q_emb, c_embs = embs[0], embs[1:]
    cos_q = np.dot(c_embs, q_emb)          # relevance (-1…1)

    selected, tot, cos_part, dist_part = [], [], [], []
    remain   = set(range(len(chunks)))
    mean_vec = None

    while remain and len(selected) < min(top_k, len(chunks)):
        if mean_vec is None:               # first pick
            best   = max(remain, key=lambda i: cos_q[i])
            e_dist = 0.0
            total  = cos_q[best]
        else:
            # Euclidean distance to current memory mean (0…2)
            cos_m   = np.dot(c_embs[list(remain)], mean_vec)
            e_dist_all = np.sqrt(2.0 - 2.0 * cos_m)
            comb    = alpha * cos_q[list(remain)] + (1 - alpha) * e_dist_all
            loc     = int(np.argmax(comb))
            best    = list(remain)[loc]
            e_dist  = e_dist_all[loc]
            total   = comb[loc]

        selected.append(best)
        tot.append(float(total))
        cos_part.append(float(cos_q[best]))
        dist_part.append(float(e_dist))
        remain.remove(best)

        # update mean memory vector (keep unit length)
        if mean_vec is None:
            mean_vec = c_embs[best].copy()
        else:
            mean_vec = np.vstack([mean_vec, c_embs[best]]).mean(axis=0)
            mean_vec /= np.linalg.norm(mean_vec)

    return [chunks[i] for i in selected], tot, cos_part, dist_part


# ── logging setup ────────────────────────────────────────────────────────
LOG_DIR  = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"hotpot_run_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── load data & models ───────────────────────────────────────────────────
items     = load_hotpotqa(JSON_PATH)[:MAX_ITEMS]
embedder  = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
generator = Generator()
generatorx = GeneratorX()

logger.info("Loaded %d HotpotQA items  •  top_k=%d  •  α=%.2f",
            len(items), TOP_K, ALPHA)

# ── iterate ──────────────────────────────────────────────────────────────
rows: list[dict] = [] 
for n, it in enumerate(items, 1):
    q = it["question"]
    chunks = split_into_chunks(it["context_text"])
    if not chunks:
        continue

    cos_chunks, cos_scores = select_topk_chunks(q, chunks, embedder, TOP_K)
    div_chunks, tot, cos_p, dist_p = select_diverse_topk_chunks(
        q, chunks, embedder, TOP_K, alpha=ALPHA
    )

    logger.info("\n━━ Q%03d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", n)
    logger.info("Question     : %s", q)
    logger.info("Ground truth : %s", "; ".join(it["answers"]))

    logger.info("Cosine-only top-%d:", TOP_K)
    for i, (c, s) in enumerate(zip(cos_chunks, cos_scores), 1):
        pv = c.replace("\n", " ")[:120] + ("…" if len(c) > 120 else "")
        logger.info("  %2d. cos=%5.3f | %s", i, s, pv)

    logger.info("Diverse top-%d (α=%.2f):", TOP_K, ALPHA)
    for i, (c, t, co, di) in enumerate(zip(div_chunks, tot, cos_p, dist_p), 1):
        pv = c.replace("\n", " ")[:120] + ("…" if len(c) > 120 else "")
        logger.info("  %2d. total=%5.3f | cos=%5.3f | dist=%5.3f | %s",
                    i, t, co, di, pv)

    # ── generate answers ────────────────────────────────────────────────
    topk_answer = generator.generate(q, cos_chunks)
    mmr_answer  = generator.generate(q, div_chunks)
    no_context = generatorx.generate(q,'')
    rows.append(
        {
            "query":        q,
            "ground_truth": "; ".join(it["answers"]),
            "context":      it["context_text"],
            "topk_answer":  topk_answer,
            "diverse_answer": mmr_answer,
            "no-context":no_context
        }
    )   

    logger.info("Top-k answer : %s", topk_answer)
    logger.info("MMR answer   : %s", mmr_answer)
    logger.info("No COntext   : %s", no_context)

logger.info("Log saved to %s", log_file.resolve())
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
csv_path = RESULTS_DIR / f"hotpot_results_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

pd.DataFrame(rows).to_csv(csv_path, index=False)
logger.info("Saved results CSV to %s", csv_path.resolve())
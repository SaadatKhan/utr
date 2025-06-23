import argparse, os, sys, logging, numpy as np
from datetime import datetime
from typing import List
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, MAX_AEM_SIZE, MODEL_PATH_EMBEDDING
from utils  import read_document, split_into_chunks, show_top_k_chunks, display_aem
from beerqa_processor import load_beerqa
from hotpotqa_processor import load_hotpotqa
from aem    import AEM
from ranker import Ranker, UTILITY_TAGS
from summarizer import Summarizer
from generator  import Generator
from generator_rat import Generator2
from verifier import Verifier
import pandas as pd
from datetime import datetime
from pathlib import Path
from rgb_processor import load_rgb
# ── logging setup ───────────────────────────────────────────────────────────
LOG_DIR = "logs"; os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def tag_priority(tag: str) -> int:
    return UTILITY_TAGS.index(tag) if tag in UTILITY_TAGS else len(UTILITY_TAGS)


# ────────────────────────── pipeline for one item ───────────────────────────
def run_pipeline(
    full_text: str,
    question: str,
    embedder: SentenceTransformer,
    ranker: Ranker,
    summariser: Summarizer,
    generator_aem: Generator2,
    generator: Generator,
    top_k: int = 7,
):
    
    #Returns (aem_answer, topk_answer)
    
    chunks = split_into_chunks(full_text)
    if not chunks:
        logger.warning("No chunks produced.")
        return "", ""

    embs = embedder.encode([question] + chunks, normalize_embeddings=True)
    sims = np.dot(embs[1:], embs[0])

    idx_sorted = np.argsort(sims)[::-1]
    topk_chunks = [chunks[i] for i in idx_sorted[:top_k]]
    show_top_k_chunks(topk_chunks, sims[idx_sorted][:top_k], k=top_k)

    aem = AEM()
    remaining = list(range(len(chunks)))

    while remaining and len(aem.get_aem()) < MAX_AEM_SIZE:
        pool_idx = sorted(remaining, key=lambda i: sims[i], reverse=True)[:top_k]
        pool_chunks = [chunks[i] for i in pool_idx]

        best_chunk, labels = ranker.select_best_chunk(
            aem.get_aem(), pool_chunks, question
        )

        logger.info("Ranker tags & reasons:")
        for lid, (tag, reason) in sorted(
            labels.items(), key=lambda kv: tag_priority(kv[1][0])
        ):
            prev = pool_chunks[lid].replace("\n", " ")[:100]
            logger.info("  (%s) %s — %s", tag, prev, reason)

        summary = summariser.condense(best_chunk)
        aem.update_aem([summary], [1.0])
        logger.info("Added summary to AEM. Current memory:")
        display_aem(aem)

        remaining.remove(pool_idx[pool_chunks.index(best_chunk)])

    aem_answer  = generator_aem.generate(question, aem.get_aem())
    topk_answer = generator.generate(question, topk_chunks)

    return aem_answer, topk_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--file")
    grp.add_argument("--beerqa_json")
    grp.add_argument("--hotpot_json", help="Path to HotpotQA JSON")
    ap.add_argument("--query", help="Needed with --file")
    ap.add_argument("--top_k", type=int, default=10)
    grp.add_argument("--rgb_json", help="Path to RGB dataset JSON")
    args = ap.parse_args()

    logger.info("Loading models once …")
    embedder   = SentenceTransformer(MODEL_PATH_EMBEDDING)
    ranker     = Ranker()
    summariser = Summarizer()
    generator  = Generator()
    generator_rat = Generator2()
    verifier = Verifier()
    rows: list[dict] = [] 
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.beerqa_json:
        items = load_beerqa(args.beerqa_json)
        logger.info("Loaded %d BEERQA items.", len(items))
        for i, it in enumerate(items,1):
            logger.info("\n━━ ITEM %d / %d ━━━━━━━━━━━━━", i, len(items))
            logger.info("Question: %s", it["question"])
            aem_ans, topk_ans = run_pipeline(
                it["context_text"], it["question"],
                embedder, ranker, summariser, generator,
                top_k=args.top_k)

            logger.info("Ground-truth answers: %s", it["answers"])
            logger.info("AEM answer  : %s", aem_ans)
            logger.info("Top-k answer: %s\n", topk_ans)
            a_acc, t_acc = verifier.accuracy()

            logger.info("Ground truth     : %s", it["answers"])
            logger.info("AEM answer       : %s   ✔" if aem_ok else " ✘", aem_ans)
            logger.info("Top-k answer     : %s   ✔" if topk_ok else " ✘", topk_ans)
            logger.info("Running accuracy – AEM: %.2f  |  Top-k: %.2f\n", a_acc, t_acc)
        #logger.info("Top-k answer: %s\n", topk_ans)
    elif args.hotpot_json:
        # ------- HotpotQA batch mode ----------------------------------------
        items = load_hotpotqa(args.hotpot_json)[:100]
        logger.info("Loaded %d HotpotQA items.", len(items))

        for i, it in enumerate(items, 1):
            logger.info("\n━━ ITEM %d / %d ━━━━━━━━━━━━━", i, len(items))
            logger.info("Question: %s", it["question"])

            aem_ans, topk_ans = run_pipeline(
                it["context_text"], it["question"],
                embedder, ranker, summariser,generator_rat, generator,
                top_k=args.top_k)
            try:
                # look for final answer from llama model out and take everything after it
                lower = aem_ans.lower()
                if "final answer" in lower:
                    # split once on the first occurrence
                    tail = aem_ans.split("Final Answer", 1)[1]
                    
                    cleaned_aem = tail.lstrip(":").strip()
                else:
                    cleaned_aem = aem_ans
            except Exception:          
                cleaned_aem = aem_ans
            logger.info("Final ANswer we retrieved: %s", cleaned_aem)
            aem_ok, topk_ok = verifier.verify(it["question"],
                                      it["answers"],
                                      cleaned_aem,
                                      topk_ans)
            a_acc, t_acc = verifier.accuracy()
            # -----------------------------------------------------------------------------------------

            logger.info("Ground-truth answers: %s", it["answers"])
            logger.info("AEM answer  : %s   %s", aem_ans, "✔" if aem_ok  else "✘")
            logger.info("Top-k answer: %s   %s", topk_ans, "✔" if topk_ok else "✘")
            logger.info("Running accuracy – AEM: %.2f | Top-k: %.2f\n", a_acc, t_acc)
            rows.append(
                {
                    "id":       it["id"],
                    "question": it["question"],
                    "context":  it["context_text"],
                    "ground_truth": it["answers"],
                    "aem_answer":   aem_ans,
                    "aem_answer_final": cleaned_aem,
                    "topk_answer":  topk_ans,
                }
            )
        df = pd.DataFrame(rows)
        csv_path = RESULTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved results CSV to %s", csv_path)

    elif args.rgb_json:
        
        items = load_rgb(args.rgb_json)
        logger.info("Loaded %d RGB items.", len(items))

        for i, itm in enumerate(items, 1):
            logger.info("\n━━ ITEM %d / %d ━━━━━━━━━━━━━", i, len(items))
            logger.info("Question: %s", itm["question"])

            aem_ans, topk_ans = run_pipeline(
                itm["context_text"], itm["question"],
                embedder, ranker, summariser,generator_rat, generator,
                top_k=args.top_k)

            try:
                # look for "final answer" 
                lower = aem_ans.lower()
                if "final answer" in lower:
                    # split once on the first occurrence
                    tail = aem_ans.split("Final Answer", 1)[1]
                    cleaned_aem = tail.lstrip(":").strip()
                else:
                    cleaned_aem = aem_ans
            except Exception:       
                cleaned_aem = aem_ans

            aem_ok, topk_ok = verifier.verify(itm["question"], itm["answers"], cleaned_aem, topk_ans)
            a_acc, t_acc = verifier.accuracy()

            # logging
            logger.info("Ground-truth answers: %s", itm["answers"])
            logger.info("AEM answer  : %s   %s", aem_ans,  "✔" if aem_ok  else "✘")
            logger.info("Top-k answer: %s   %s", topk_ans, "✔" if topk_ok else "✘")
            logger.info("Running accuracy – AEM: %.2f | Top-k: %.2f\n", a_acc, t_acc)

            # append CSV row
            rows.append(
                {
                    "id":           itm["id"],
                    "question":     itm["question"],
                    "context":      itm["context_text"],
                    "ground_truth": "; ".join(itm["answers"]),
                    "aem_answer":   aem_ans,
                    "topk_answer":  topk_ans,
                }
            )
        df = pd.DataFrame(rows)
        csv_path = RESULTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved results CSV to %s", csv_path)

    else:
        if not args.query:
            ap.error("--query required with --file")
        text = read_document(args.file)
        aem_ans, topk_ans = run_pipeline(
            text, args.query, embedder,
            ranker, summariser, generator, top_k=args.top_k)

        logger.info("\nAEM answer :\n%s", aem_ans)
        logger.info("\nTop-k answer:\n%s", topk_ans)
        aem_ok, topk_ok = verifier.verify(it["question"],
                                  it["answers"],
                                  aem_ans,
                                  topk_ans)

        

    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()

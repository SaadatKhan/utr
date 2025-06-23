
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import MODEL_PATH_LLM_Generator, RANKER_PATH_LLM, CACHE_DIR  # plus any LLM_* sampling constants if you added them

UTILITY_TAGS = ["VH", "H", "M", "L"]


def _tag_priority(tag: str) -> int:
    """Smaller index = higher utility."""
    return UTILITY_TAGS.index(tag) if tag in UTILITY_TAGS else len(UTILITY_TAGS)


class Ranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RANKER_PATH_LLM, token = '',  cache_dir = CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            RANKER_PATH_LLM,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token = '',
            cache_dir = CACHE_DIR
            
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    def select_best_chunk(
        self,
        aem_memory: List[str],
        candidate_chunks: List[str],
        query: str,
    ) -> Tuple[str, Dict[int, Tuple[str, str]]]:
       

        # 1) run the model + parse
        labels = self._rank_chunks(aem_memory, candidate_chunks, query)

        # 2) fallback if parsing failed entirely
        if not labels:
            labels = {0: ("M", "fallback – no label")}

        # 3) filter IDs that are outside the candidate range
        orig_labels = labels
        labels = {i: tr for i, tr in orig_labels.items()
                  if i < len(candidate_chunks)}

        # if everything was out-of-range, fallback to first chunk
        if not labels:
            labels = {0: ("M", "all ids out-of-range")}

        # 4) choose the best by utility priority
        best_id = min(labels.items(), key=lambda kv: _tag_priority(kv[1][0]))[0]

        return candidate_chunks[best_id], labels

    # ------------------------------------------------------------------ #
    def _rank_chunks(
        self,
        aem_mem: List[str],
        chunks: List[str],
        query: str,
    ) -> Dict[int, Tuple[str, str]]:
        
        known_ctx = "\n".join(f"- {c}" for c in aem_mem) or "(AEM empty)"
        cand_txt  = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are #EvidenceTriage, a concise assistant.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        ## Task
        Given a *question*, some *known context*, and several *new candidate chunks*, \
        you MUST label each candidate with how much **new and relevant information** it adds \
        beyond the known context.  **Make sure to use these labels with <> only**:
        <VH> = very high, <H> = high, <M> = medium, <L> = low / redundant.

        ### OUTPUT
        Return **exactly one line per ID** in this EXACT format. You MUST follow the format:
        [1] <VH> = brief reason
        [2] <M>  = brief reason
        [3] <H>  = brief reason

        Now, do the same for the following input.

        ### INPUT
        Question: {query}

        Known context:
        {known_ctx}

        Candidate chunks:
        {cand_txt}

        ENSURE each ID gets a rating (VH/H/M/L) and STRICTLY ADHERE EXACTLy to the format.
        ### OUTPUT
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        generation = self.pipe(
            prompt,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.1,   # low temp → deterministic-ish
            top_k=1,
            top_p=0.8,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        model_reply = generation[len(prompt):]

        # ---- DEBUG ----  (comment out once happy)
        print("\n[Ranker raw output]\n", model_reply.strip(), "\n[End raw output]\n")

        return self._parse_labels(model_reply)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_labels(text: str) -> Dict[int, Tuple[str, str]]:
        
        labels: Dict[int, Tuple[str, str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not (line.startswith("[") and "]" in line):
                continue
            try:
                idx_part, rest = line.split("]", 1)
                idx = int(idx_part[1:])
                parts = rest.strip().lstrip("<").split(None, 1)   # TAG rest…
                tag = parts[0].strip("><").upper()
                reason = parts[1].lstrip("=:").strip() if len(parts) > 1 else ""
                if tag in UTILITY_TAGS:
                    labels[idx] = (tag, reason or "(no reason)")
            except (ValueError, IndexError):
                continue
        return labels

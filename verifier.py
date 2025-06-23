

from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import MODEL_PATH_LLM, CACHE_DIR, VERIFIER_PATH_LLM   # add these in config.py


class Verifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            VERIFIER_PATH_LLM,   
            token = '', 
            cache_dir = CACHE_DIR
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            VERIFIER_PATH_LLM,
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

        # running statistics
        self.n = 0
        self.aem_correct = 0
        self.topk_correct = 0

    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    def verify(
        self,
        question: str,
        refs: str,
        aem_ans: str,
        topk_ans: str,
    ) -> Tuple[bool, bool]:
        

        # ---------- local helper -----------------------------------------
        def _ask_llm(candidate: str) -> bool:
            """Return True if model replies 'yes', else False."""
            refs_joined = "; ".join(refs)

            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are #AnswerJudge, a strict grader who replies with exactly "
                "\"yes\" or \"no\" — no punctuation, no extra words.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Question: {question}\n\n"
                f"Reference answer :\n{refs}\n\n"
                f"Candidate answer:\n{candidate}\n\n"
                "Respond with exactly \"yes\" if the candidate answer in general matches the reference answer in meaning, otherwise \"no\n. If they explicitly don't talk about the same thing, the answer must be no. "
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

            reply = self.pipe(
                prompt,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"][len(prompt):].strip().lower()

            return reply.startswith("y")

        # ---------- judge both answers -----------------------------------
        aem_ok  = _ask_llm(aem_ans)
        topk_ok = _ask_llm(topk_ans)

        # update running stats
        self.n += 1
        if aem_ok:
            self.aem_correct += 1
        if topk_ok:
            self.topk_correct += 1

        return aem_ok, topk_ok



    # ------------------------------------------------------------------ #
    def accuracy(self) -> Tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        return self.aem_correct / self.n, self.topk_correct / self.n

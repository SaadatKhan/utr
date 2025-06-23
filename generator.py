
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List
from config import MODEL_PATH_LLM_Generator, LLM_TEMPERATURE, LLM_TOP_K, LLM_TOP_P, CACHE_DIR


class Generator:
    

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_LLM_Generator, token = '', cache_dir = CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH_LLM_Generator,
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

    # --------------------------------------------------------------------- #
    def generate(self, query: str, context_chunks: List[str]) -> str:
        
        context = "\n\n".join(f"- {c}" for c in context_chunks) or "(no context)"
        
        
        prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You are an expert question answering assistant. Answer the question **only** using the context below. Do not make things up. <|eot_id|>
                                    <|start_header_id|>user<|end_header_id|>
        ### Question
        {query}

        ### Context
        {context}

        
        You MUST make sure you only answer the question using the context. Since, it is a close-ended QA task, the answer would likely be a very short. Do not generate unnecessary words or descriptive answers.
        ### Answer
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        result = self.pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=LLM_TEMPERATURE,   # 0.1
            top_k=LLM_TOP_K,               # 1  → effectively greedy
            top_p=LLM_TOP_P,               # 0.8 (keeps a narrow nucleus)
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # strip off the prompt prefix
        return result[len(prompt) :].strip()

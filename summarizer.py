
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import MODEL_PATH_LLM, LLM_TOP_P,LLM_TOP_K,LLM_TEMPERATURE, CACHE_DIR

class Summarizer:
   

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_LLM, token = '', cache_dir = CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH_LLM,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token = '',
            cache_dir = CACHE_DIR
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def condense(self, chunk: str) -> str:
        prompt = f"Summarise the following content into one concise paragraph:\n\n{chunk}\n\n### Summary:"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You are a summarizing assistant. Summarize the paragraph below to a very concise.<|eot_id|>
                                    <|start_header_id|>user<|end_header_id|>
        ###Paragraph:
        {chunk}

        You must make sure to summarize concisely. Do not generate unnecessary words as it is costly.
        ### Summary
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        result = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=LLM_TEMPERATURE,   # 0.1
            top_k=LLM_TOP_K,               # 1  → effectively greedy
            top_p=LLM_TOP_P,               # 0.8 (keeps a narrow nucleus)
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return result.strip()[len(prompt):].strip()

MODEL_PATH_EMBEDDING = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
MODEL_PATH_SUMMARY = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH_LLM_Generator = "meta-llama/Meta-Llama-3.1-8B-Instruct"
RANKER_PATH_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"
VERIFIER_PATH_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHUNK_SIZE = 400
MAX_AEM_SIZE = 5
LLM_TEMPERATURE = 0.1
LLM_TOP_K       = 1      # 1 → greedy
LLM_TOP_P       = 0.8    # kept for safety; ≤ 0.8 is conservative


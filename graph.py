# graph_extractor_llama.py
#
# Requires: python -m pip install openai python-dotenv
# Put your API key in a .env file or export it.
import json, textwrap, os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()                      # picks up API key
client = OpenAI()

# ------------------------------------------------------------------
# 1. Llama prompt formatting
# ------------------------------------------------------------------
def format_llama_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format prompt with Llama's special tokens."""
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# ------------------------------------------------------------------
# 2. Prompt templates
# ------------------------------------------------------------------
PLAN_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert question decomposer.  
    Task: Break the user's **single question** into the minimal
    ordered list of sub-questions that must all be answered
    to fully answer the original.  Use the format:
    PLAN:
    1. ...
    2. ...
    3. ...
    Only produce the PLAN section—no extra commentary.
""")

GRAPH_SYSTEM_PROMPT = textwrap.dedent("""
    You are a graph-extraction assistant.
    Produce a JSON object with this schema:
    {
      "nodes": [{"id": "n1", "text": "..."}],
      "edges": [{"source": "n1", "target": "n2", "label": "relation"}],
      "reasoning_path": ["n1", "n3", "n7"]
    }
    • Each sub-question from the plan should correspond to at least one node
    • Choose concise node texts (≤10 tokens)
    • The graph must be connected and cover every sub-question
    • Output only the JSON—no markdown, no explanations.
""")

# ------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI with Llama-formatted prompt."""
    formatted_prompt = format_llama_prompt(system_prompt, user_prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def decompose_question(question: str) -> str:
    user_prompt = f"QUESTION:\n{question}"
    return call_llm(PLAN_SYSTEM_PROMPT, user_prompt)

def build_graph(question: str, plan: str, passages: List[str]) -> Dict:
    ctx_block = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    user_prompt = f"ORIGINAL QUESTION: {question}\n\nDECOMPOSITION PLAN:\n{plan}\n\nCONTEXT PASSAGES:\n{ctx_block}"
    raw_json = call_llm(GRAPH_SYSTEM_PROMPT, user_prompt)
    return json.loads(raw_json)

# ------------------------------------------------------------------
# 4. Public entry point
# ------------------------------------------------------------------
def extract_graph(question: str, passages: List[str]) -> Dict:
    plan = decompose_question(question)
    graph = build_graph(question, plan, passages)
    graph["plan"] = plan
    return graph

# ------------------------------------------------------------------
# 5. Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    toy_q = (
        "What is the capital city of the country where the author of "
        "'The Old Man and the Sea' was born?"
    )
    toy_ctx = [
        "Ernest Hemingway wrote The Old Man and the Sea.",
        "Ernest Hemingway was born in Oak Park, Illinois, United States.",
        "Washington, D.C. is the capital of the United States.",
    ]
    g = extract_graph(toy_q, toy_ctx)
    print(json.dumps(g, indent=2, ensure_ascii=False))
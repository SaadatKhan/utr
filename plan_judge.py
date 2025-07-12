import json
from typing import List, Dict, Any
from openai import OpenAI

class LLMJudge:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", api_key=None, base_url=None):
        """
        Initialize the LLM Judge for evaluating chunk quality against plans.
        
        Args:
            model_name: Llama model name (default: meta-llama/Llama-3.1-8B-Instruct)
            api_key: OpenAI API key
            base_url: Base URL for the API (useful for custom deployments)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.few_shot_examples = self._get_few_shot_examples()
    
    def _get_few_shot_examples(self) -> str:
        """Generate few-shot examples for the judge."""
        return """Plan:
1) Find the protagonist in the movie 'Inception'
2) Find the protagonist's birthplace
3) Find the population of that city

Chunks:
- "Inception is a 2010 science fiction film starring Leonardo DiCaprio as Dom Cobb, a professional thief who infiltrates people's dreams..."
- "Leonardo DiCaprio was born in Los Angeles, California, United States on November 11, 1974..."
- "Los Angeles is the most populous city in California with a population of approximately 4 million people as of 2023..."

Can all questions in the plan be answered using the chunks?
PASS

---

Plan:
1) Find the director of the movie 'Titanic'
2) Find the director's net worth
3) Find the director's upcoming projects

Chunks:
- "Titanic is a 1997 epic romance and disaster film. The movie was a massive box office success..."
- "The film won 11 Academy Awards including Best Picture and Best Director..."
- "Leonardo DiCaprio and Kate Winslet starred as the main characters in this epic love story..."

Can all questions in the plan be answered using the chunks?
FAIL"""

    def evaluate_chunks(self, plan: List[str], chunks: List[str], lambda_value: float) -> Dict[str, Any]:
        """
        Evaluate if the provided chunks can satisfy the given plan.
        
        Args:
            plan: List of questions/steps in the plan
            chunks: List of text chunks retrieved at the given lambda
            lambda_value: The lambda value used for retrieval
            
        Returns:
            Dictionary with evaluation results
        """
        plan_text = "\n".join([f"{i+1}) {step}" for i, step in enumerate(plan)])
        chunks_text = "\n".join([f"- {chunk}" for chunk in chunks])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{self.few_shot_examples}

---

Plan:
{plan_text}

Chunks:
{chunks_text}

Can all questions in the plan be answered using the chunks?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the response
        assessment = self._parse_response(response_text)
        
        return {
            "lambda": lambda_value,
            "can_satisfy_plan": assessment["decision"],
            "explanation": assessment["explanation"],
            "chunks_count": len(chunks),
            "plan_steps": len(plan)
        }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract decision."""
        response = response.strip()
        
        if "PASS" in response:
            decision = True
            explanation = "Plan can be satisfied"
        elif "FAIL" in response:
            decision = False
            explanation = "Plan cannot be satisfied"
        else:
            decision = False
            explanation = "Could not parse response"
        
        return {
            "decision": decision,
            "explanation": explanation
        }
    
    def find_best_lambda(self, plan: List[str], chunks_by_lambda: Dict[float, List[str]]) -> Dict[str, Any]:
        """
        Find the best lambda value by evaluating all lambda options.
        
        Args:
            plan: List of questions/steps in the plan
            chunks_by_lambda: Dictionary mapping lambda values to their retrieved chunks
            
        Returns:
            Dictionary with the best lambda and evaluation results
        """
        results = []
        
        for lambda_val, chunks in chunks_by_lambda.items():
            result = self.evaluate_chunks(plan, chunks, lambda_val)
            results.append(result)
        
        # Find the best lambda (first one that can satisfy the plan)
        best_lambda = None
        for result in sorted(results, key=lambda x: x["lambda"]):
            if result["can_satisfy_plan"]:
                best_lambda = result
                break
        
        # If no lambda can satisfy the plan, choose the one with most chunks (highest diversity)
        if best_lambda is None:
            best_lambda = max(results, key=lambda x: x["chunks_count"])
        
        return {
            "best_lambda": best_lambda["lambda"],
            "can_satisfy_plan": best_lambda["can_satisfy_plan"],
            "explanation": best_lambda["explanation"],
            "all_evaluations": results
        }


# Example usage
if __name__ == "__main__":
    # Initialize judge with Llama model
    judge = LLMJudge(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        api_key="your-api-key-here",
        base_url="https://api.together.xyz/v1"  # Example for Together AI
    )
    
    # Example plan
    plan = [
        "Find the protagonist in the movie 'Inception'",
        "Find the protagonist's birthplace",
        "Find the population of that city"
    ]
    
    # Example chunks for different lambda values
    chunks_by_lambda = {
        0.25: [
            "Inception is a 2010 film starring Leonardo DiCaprio as Dom Cobb",
            "Leonardo DiCaprio was born in Los Angeles, California",
            "Los Angeles has a population of approximately 4 million people"
        ],
        0.50: [
            "Inception is a science fiction film about dreams",
            "The movie stars Leonardo DiCaprio and Marion Cotillard"
        ],
        0.75: [
            "Inception won several Academy Awards",
            "The film was directed by Christopher Nolan"
        ]
    }
    
    # Find best lambda
    result = judge.find_best_lambda(plan, chunks_by_lambda)
    print(f"Best lambda: {result['best_lambda']}")
    print(f"Can satisfy plan: {result['can_satisfy_plan']}")
    print(f"Explanation: {result['explanation']}")
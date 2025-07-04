import re
from typing import List, Dict
from transformers import pipeline
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SimpleGroundednessJudge:
    """
    Simple judge to evaluate if sentences are grounded in context using LLM entailment
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize with a Llama model
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.llm = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        
        # Llama chat format prompt for entailment task
        self.prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise fact-checker. Your task is to determine if a sentence is supported by the given context. Answer only "YES" or "NO".<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Sentence: {sentence}

Is the information in the sentence supported by the context?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def is_grounded(self, sentence: str, context: str) -> bool:
        """
        Check if a sentence is grounded in the context
        
        Args:
            sentence: The sentence to check
            context: The context documents
            
        Returns:
            True if grounded, False otherwise
        """
        prompt = self.prompt_template.format(
            context=context, 
            sentence=sentence
        )
        
        # Generate response
        response = self.llm(prompt, max_length=len(prompt.split()) + 10, temperature=0.1)
        generated_text = response[0]['generated_text']
        
        # Extract only the answer part
        answer = generated_text[len(prompt):].strip()
        
        # Simple parsing - look for YES/NO
        return "YES" in answer.upper()
    
    def evaluate_response(self, response: str, context: str) -> Dict:
        """
        Evaluate groundedness of a response
        
        Args:
            response: Generated response text
            context: Context documents
            
        Returns:
            Dictionary with results
        """
        # Split into sentences
        sentences = self.split_sentences(response)
        
        # Check each sentence
        results = []
        for sentence in sentences:
            is_grounded = self.is_grounded(sentence, context)
            results.append({
                'sentence': sentence,
                'grounded': is_grounded
            })
        
        # Calculate overall score
        grounded_count = sum(1 for r in results if r['grounded'])
        total_count = len(results)
        overall_score = grounded_count / total_count if total_count > 0 else 0
        
        return {
            'overall_score': overall_score,
            'grounded_count': grounded_count,
            'total_count': total_count,
            'sentences': results
        }

# Example usage
if __name__ == "__main__":
    judge = SimpleGroundednessJudge()
    
    context = "The Eiffel Tower is 330 meters tall and located in Paris, France."
    response = "The Eiffel Tower is in Paris and is 330 meters high. It was built on Mars."
    
    result = judge.evaluate_response(response, context)
    print(f"Overall groundedness: {result['overall_score']:.2f}")
    
    for item in result['sentences']:
        status = "✓" if item['grounded'] else "✗"
        print(f"{status} {item['sentence']}")
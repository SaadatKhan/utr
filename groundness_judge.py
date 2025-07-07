import re
from typing import List, Dict
from transformers import pipeline
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class GroundednessJudge:
    """
    Judge to evaluate if sentences are grounded in context and check coverage against key points
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
        self.grounding_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise fact-checker. Your task is to determine if a sentence is supported by the given context. Answer only "YES" or "NO".<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Sentence: {sentence}

Is the information in the sentence supported by the context?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Prompt for checking coverage
        self.coverage_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Check if the given answer mentions or addresses the key point from the context. Answer only "YES" or "NO".<|eot_id|><|start_header_id|>user<|end_header_id|>

Key Point: {key_point}

Answer: {answer}

Does the answer mention or address this key point?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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
        prompt = self.grounding_prompt.format(
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
    
    def check_coverage(self, answer: str, key_point: str) -> bool:
        """
        Check if an answer covers a specific key point
        
        Args:
            answer: The generated answer
            key_point: A key point from the context
            
        Returns:
            True if the key point is covered, False otherwise
        """
        prompt = self.coverage_prompt.format(
            key_point=key_point,
            answer=answer
        )
        
        response = self.llm(prompt, max_length=len(prompt.split()) + 10, temperature=0.1)
        generated_text = response[0]['generated_text']
        
        # Extract only the answer part
        result = generated_text[len(prompt):].strip()
        
        return "YES" in result.upper()
    
    def calculate_coverage_score(self, answer: str, key_points: List[str]) -> Dict:
        """
        Calculate coverage score for an answer against provided key points
        
        Args:
            answer: The generated answer
            key_points: List of key points to check coverage against
            
        Returns:
            Dictionary with coverage results
        """
        if not key_points:
            return {
                'coverage_score': 0.0,
                'covered_points': 0,
                'total_points': 0,
                'key_points': [],
                'coverage_details': []
            }
        
        # Check coverage for each key point
        coverage_details = []
        covered_count = 0
        
        for key_point in key_points:
            is_covered = self.check_coverage(answer, key_point)
            if is_covered:
                covered_count += 1
            
            coverage_details.append({
                'key_point': key_point,
                'covered': is_covered
            })
        
        coverage_score = covered_count / len(key_points) if key_points else 0
        
        return {
            'coverage_score': coverage_score,
            'covered_points': covered_count,
            'total_points': len(key_points),
            'key_points': key_points,
            'coverage_details': coverage_details
        }
    
    def evaluate_response(self, response: str, context: str, key_points: List[str]) -> Dict:
        """
        Evaluate both groundedness and coverage of a response
        
        Args:
            response: Generated response text
            context: Context documents (for groundedness checking)
            key_points: Pre-extracted key points from context (for coverage checking)
            
        Returns:
            Dictionary with groundedness and coverage results
        """
        # Split into sentences
        sentences = self.split_sentences(response)
        
        # Check groundedness for each sentence
        grounding_results = []
        for sentence in sentences:
            is_grounded = self.is_grounded(sentence, context)
            grounding_results.append({
                'sentence': sentence,
                'grounded': is_grounded
            })
        
        # Calculate groundedness score
        grounded_count = sum(1 for r in grounding_results if r['grounded'])
        total_count = len(grounding_results)
        groundedness_score = grounded_count / total_count if total_count > 0 else 0
        
        # Calculate coverage score using provided key points
        coverage_results = self.calculate_coverage_score(response, key_points)
        
        return {
            'groundedness_score': groundedness_score,
            'grounded_count': grounded_count,
            'total_sentences': total_count,
            'sentences': grounding_results,
            'coverage_score': coverage_results['coverage_score'],
            'covered_points': coverage_results['covered_points'],
            'total_key_points': coverage_results['total_points'],
            'key_points': coverage_results['key_points'],
            'coverage_details': coverage_results['coverage_details']
        }
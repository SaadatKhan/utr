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
        self.grounding_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise fact-checker. Your task is to determine if a sentence is supported by the given context. Answer only "YES" or "NO".<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Sentence: {sentence}

Is the information in the sentence supported by the context?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Prompt for extracting key points from context
        self.key_points_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Extract the key factual points from the context. List each distinct fact or piece of information as a separate bullet point.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Extract key points as bullet points:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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
    
    def extract_key_points(self, context: str) -> List[str]:
        """
        Extract key factual points from the context
        
        Args:
            context: The context documents
            
        Returns:
            List of key points
        """
        prompt = self.key_points_prompt.format(context=context)
        
        response = self.llm(prompt, max_length=len(prompt.split()) + 200, temperature=0.1)
        generated_text = response[0]['generated_text']
        
        # Extract only the answer part
        answer = generated_text[len(prompt):].strip()
        
        # Parse bullet points
        key_points = []
        for line in answer.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                # Remove bullet point markers
                point = re.sub(r'^[•\-\*]\s*', '', line).strip()
                if point:
                    key_points.append(point)
        
        return key_points
    
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
    
    def calculate_coverage_score(self, answer: str, context: str) -> Dict:
        """
        Calculate coverage score for an answer
        
        Args:
            answer: The generated answer
            context: The context documents
            
        Returns:
            Dictionary with coverage results
        """
        # Extract key points from context
        key_points = self.extract_key_points(context)
        
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
    
    def evaluate_response(self, response: str, context: str) -> Dict:
        """
        Evaluate both groundedness and coverage of a response
        
        Args:
            response: Generated response text
            context: Context documents
            
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
        
        # Calculate coverage score
        coverage_results = self.calculate_coverage_score(response, context)
        
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

# Demo with longer context and two candidate answers
if __name__ == "__main__":
    judge = SimpleGroundednessJudge()
    
    # Longer context for demo
    context = """
    Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change, primarily due to fossil fuel burning which releases greenhouse gases.
    
    The main greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). CO2 levels have increased by over 40% since pre-industrial times, primarily from burning coal, oil, and gas.
    
    Effects of climate change include rising global temperatures, melting ice caps and glaciers, rising sea levels, and more frequent extreme weather events like hurricanes, droughts, and heatwaves.
    
    The Paris Agreement, signed in 2015, aims to limit global warming to well below 2°C above pre-industrial levels, with efforts to limit it to 1.5°C. Countries committed to reducing greenhouse gas emissions and achieving net-zero emissions by 2050.
    
    Renewable energy sources like solar and wind power are key solutions for reducing emissions. Energy efficiency improvements and electric vehicles also play important roles in mitigation efforts.
    """
    
    # Candidate Answer 1 - Good coverage and groundedness
    answer1 = """
    Climate change is caused primarily by human activities since the 1800s, especially burning fossil fuels. 
    The main greenhouse gases are CO2, methane, and nitrous oxide, with CO2 levels rising over 40% since pre-industrial times. 
    This leads to rising temperatures, melting ice caps, rising sea levels, and extreme weather events. 
    The Paris Agreement aims to limit warming to 1.5-2°C above pre-industrial levels by achieving net-zero emissions by 2050. 
    Solutions include renewable energy like solar and wind, energy efficiency, and electric vehicles.
    """
    
    # Candidate Answer 2 - Partial coverage with some hallucination
    answer2 = """
    Climate change is a serious global issue caused by greenhouse gas emissions. 
    The main cause is burning fossil fuels which releases CO2 into the atmosphere. 
    This has led to rising temperatures and melting glaciers worldwide. 
    Scientists predict that sea levels could rise by 10 meters by 2030 if we don't act quickly. 
    Nuclear power is the only viable solution to completely eliminate carbon emissions.
    """
    
    print("=== EVALUATING ANSWER 1 ===")
    result1 = judge.evaluate_response(answer1, context)
    
    print(f"Groundedness Score: {result1['groundedness_score']:.2f}")
    print(f"Coverage Score: {result1['coverage_score']:.2f}")
    print(f"Grounded Sentences: {result1['grounded_count']}/{result1['total_sentences']}")
    print(f"Covered Key Points: {result1['covered_points']}/{result1['total_key_points']}")
    
    print("\nGroundedness Details:")
    for item in result1['sentences']:
        status = "✓" if item['grounded'] else "✗"
        print(f"  {status} {item['sentence']}")
    
    print("\nCoverage Details:")
    for item in result1['coverage_details']:
        status = "✓" if item['covered'] else "✗"
        print(f"  {status} {item['key_point']}")
    
    print("\n" + "="*50)
    print("=== EVALUATING ANSWER 2 ===")
    result2 = judge.evaluate_response(answer2, context)
    
    print(f"Groundedness Score: {result2['groundedness_score']:.2f}")
    print(f"Coverage Score: {result2['coverage_score']:.2f}")
    print(f"Grounded Sentences: {result2['grounded_count']}/{result2['total_sentences']}")
    print(f"Covered Key Points: {result2['covered_points']}/{result2['total_key_points']}")
    
    print("\nGroundedness Details:")
    for item in result2['sentences']:
        status = "✓" if item['grounded'] else "✗"
        print(f"  {status} {item['sentence']}")
    
    print("\nCoverage Details:")
    for item in result2['coverage_details']:
        status = "✓" if item['covered'] else "✗"
        print(f"  {status} {item['key_point']}")

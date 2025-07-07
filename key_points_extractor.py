import re
from typing import List
from transformers import pipeline


class KeyPointsExtractor:
    """
    Module to extract key factual points from context documents
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
        
        # Prompt for extracting key points from context
        self.key_points_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Extract the key factual points from the context. List each distinct fact or piece of information as a separate bullet point.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Extract key points as bullet points:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
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
# Usage example demonstrating the separation of key points extraction and judging

from key_points_extractor import KeyPointsExtractor
from groundedness_judge import GroundednessJudge

# Demo with longer context and two candidate answers
if __name__ == "__main__":
    # Initialize both modules
    extractor = KeyPointsExtractor()
    judge = GroundednessJudge()
    
    # Longer context for demo
    context = """
    Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change, primarily due to fossil fuel burning which releases greenhouse gases.
    
    The main greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). CO2 levels have increased by over 40% since pre-industrial times, primarily from burning coal, oil, and gas.
    
    Effects of climate change include rising global temperatures, melting ice caps and glaciers, rising sea levels, and more frequent extreme weather events like hurricanes, droughts, and heatwaves.
    
    The Paris Agreement, signed in 2015, aims to limit global warming to well below 2°C above pre-industrial levels, with efforts to limit it to 1.5°C. Countries committed to reducing greenhouse gas emissions and achieving net-zero emissions by 2050.
    
    Renewable energy sources like solar and wind power are key solutions for reducing emissions. Energy efficiency improvements and electric vehicles also play important roles in mitigation efforts.
    """
    
    # Extract key points ONCE from the context
    print("=== EXTRACTING KEY POINTS ===")
    key_points = extractor.extract_key_points(context)
    print(f"Extracted {len(key_points)} key points:")
    for i, point in enumerate(key_points, 1):
        print(f"  {i}. {point}")
    
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
    
    print("\n" + "="*50)
    print("=== EVALUATING ANSWER 1 ===")
    # Pass the pre-extracted key points to the judge
    result1 = judge.evaluate_response(answer1, context, key_points)
    
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
    # Use the SAME key points for consistency
    result2 = judge.evaluate_response(answer2, context, key_points)
    
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
    
    print("\n" + "="*50)
    print("=== SUMMARY ===")
    print(f"Answer 1 - Groundedness: {result1['groundedness_score']:.2f}, Coverage: {result1['coverage_score']:.2f}")
    print(f"Answer 2 - Groundedness: {result2['groundedness_score']:.2f}, Coverage: {result2['coverage_score']:.2f}")


# Alternative usage for batch evaluation
def batch_evaluate_answers(context: str, answers: List[str], model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """
    Efficiently evaluate multiple answers against the same context
    
    Args:
        context: The context documents
        answers: List of candidate answers to evaluate
        model_name: Model to use for evaluation
        
    Returns:
        List of evaluation results for each answer
    """
    # Initialize modules
    extractor = KeyPointsExtractor(model_name)
    judge = GroundednessJudge(model_name)
    
    # Extract key points once
    key_points = extractor.extract_key_points(context)
    
    # Evaluate all answers using the same key points
    results = []
    for i, answer in enumerate(answers):
        print(f"Evaluating answer {i+1}/{len(answers)}...")
        result = judge.evaluate_response(answer, context, key_points)
        results.append(result)
    
    return results, key_points
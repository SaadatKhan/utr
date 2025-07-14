prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a strict judge evaluating text chunks. Only count information that is explicitly stated.<|eot_id|><|start_header_id|>user<|end_header_id|>

{self.few_shot_examples}

---

Plan:
{plan_text}

Chunks:
{chunks_text}

Score each question 0-3 based on how directly it can be answered:
3: EXPLICIT answer with specific facts/numbers stated directly
2: Clear answer with direct relevant information present  
1: Partial information but missing key details
0: No relevant information or requires inference

STRICT RULES:
- Only count information DIRECTLY STATED in chunks
- Do NOT give points for logical leaps or inferences
- Do NOT give points for vague or tangential content

You only have 500 tokens, so dont generate unnecessary tokens.
At the end you must answer in the format:
Number of Answerable Questions: <Total_Score>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

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
1. Yes - Leonardo DiCaprio is explicitly named as the protagonist
2. Yes - Born in Los Angeles, California is directly stated
3. Yes - Population of 4 million is explicitly mentioned
Number of Answerable Questions: 9

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
1. No - Director name not mentioned, only that it won Best Director
2. No - No financial information about anyone provided
3. No - No information about future projects mentioned
Number of Answerable Questions: 0

---

Plan:
1) Find the author of 'Harry Potter'
2) Find the author's age
3) Find the number of books in the series

Chunks:
- "J.K. Rowling wrote the Harry Potter fantasy book series that became globally popular..."
- "The series consists of seven main novels plus several companion books..."
- "Rowling has become one of the most successful authors in modern publishing..."

Can all questions in the plan be answered using the chunks?
1. Yes - J.K. Rowling is explicitly named as the author (Score: 3)
2. No - Age not mentioned anywhere in the chunks (Score: 0)  
3. Yes - Seven main novels is directly stated (Score: 3)
Number of Answerable Questions: 6"""
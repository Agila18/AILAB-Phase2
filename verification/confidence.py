"""
👨‍💻 Person 6: Confidence Score
Responsibility: Compute a confidence score (0.0 to 1.0) for the answer.

CRITICAL REQUIREMENT:
Input: The generated answer, the retrieved context.
Output: Float between 0.0 and 1.0.

INNOVATION OPPORTUNITIES:
- Use sentence transformers to calculate cosine similarity between answer and context.
- If using an LLM API, leverage returned logprobs.
- Combine multiple metrics (e.g., number of citations, length, keyword overlap).
"""

def compute_confidence(answer: str, context: list[dict]) -> float:
    """
    Computes a confidence score for the generated answer.
    
    Args:
        answer (str): The LLM generated answer.
        context (list[dict]): The retrieved chunks.
        
    Returns:
        float: A score from 0.0 to 1.0.
    """
    
    # TODO: Implement confidence scoring logic.
    # E.g. basic keyword overlap or similarity check.
    
    return 0.85

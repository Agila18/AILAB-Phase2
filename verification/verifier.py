"""
👨‍💻 Person 6: Answer Verifier (Agentic Step) (🥉 THIRD IMPORTANT)
Responsibility: Verify if the generated answer is supported by the context.

⚠️ WHY THIS IS CRITICAL:
Adds a crucial trust layer to the system! Prevents wrong answers, hallucinations, and boosts credibility.

CRITICAL REQUIREMENT:
Input: The generated answer, the retrieved context.
Output: Boolean (True if supported, False if unsupported/hallucinated).

INNOVATION OPPORTUNITIES:
- Use a secondary LLM call to cross-check claims against the context (Self-RAG concept).
- Use strict string matching or semantic similarity for verification.
- Return detailed reasons for failure alongside the boolean.
"""

def verify_answer(answer: str, context: list[dict]) -> bool:
    """
    Checks if the answer is grounded in the retrieved context.
    
    Args:
        answer (str): The LLM generated answer.
        context (list[dict]): The retrieved chunks.
        
    Returns:
        bool: True if supported.
    """
    
    # TODO: Implement verification logic.
    # Could be an LLM asking: "Is the claim in the answer supported by this context text? Yes or No"
    
    return True

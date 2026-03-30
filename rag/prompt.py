"""
👨‍💻 Person 5: Prompt Engineering (🚨 CORE SYSTEM COMPONENT 🚨)
Responsibility: Build the prompt using the user query and retrieved context.

⚠️ WHY THIS IS VERY CRITICAL:
This decides:
1. Accuracy
2. Clarity
3. Trustworthiness of the whole RAG system.

CRITICAL REQUIREMENT:
Craft a robust prompt template that INSTRUCTS the LLM to use the provided context 
ONLY, to not guess, and to output citations.

INNOVATION OPPORTUNITIES:
- Add strict rules to handle "I don't know" gracefully.
- Prevent prompt injection attacks ("Ignore all prior instructions").
- Format the context beautifully so the LLM parses it easily.
"""

def build_prompt(query: str, context: list[dict]) -> str:
    """
    Constructs the prompt string to feed to the LLM.
    
    Args:
        query (str): The user's question.
        context (list[dict]): Retrieved chunks.
        
    Returns:
        str: The final prompt string.
    """
    # TODO: Combine the text/source from 'context' and interleave it with the instruction and query.
    
    prompt = f"""
    You are an intelligent Academic Assistant.
    Answer the user's question using ONLY the context provided below.
    If you do not know the answer based on the context, say "I don't know based on the provided documents."
    Do NOT guess.
    
    Context:
    ...
    
    Question: {{query}}
    """
    
    return prompt

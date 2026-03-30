"""
👨‍💻 Person 5: LLM Generator (🚨 CORE SYSTEM COMPONENT 🚨)
Responsibility: Generate the final answer using LangChain.

⚠️ WHY THIS IS CRITICAL:
1. If answer is wrong -> useless system ❌
2. If hallucination happens -> project fails ❌
3. Must ensure answer is ONLY from documents. No guessing allowed!
4. 🚨 REQUIRES OLLAMA: You must have Ollama installed and `llama3` downloaded (`ollama pull llama3`).

CRITICAL REQUIREMENT:
Input: User query, retrieved context (list of dicts).
Function `generate_answer(query, context)` -> "Answer string"

INNOVATION OPPORTUNITIES:
- Use local models (Ollama) or secure API-based LLMs.
- Keep token size within appropriate limits.
- Return structured outputs using LangChain output parsers.
"""

from rag.prompt import build_prompt
# from langchain_community.llms import Ollama
# from core.config import MODEL_NAME

def generate_answer(query: str, context: list[dict]) -> str:
    """
    Generates the answer using an LLM.
    
    Args:
        query (str): The user's question.
        context (list[dict]): The retrieved contextual chunks.
        
    Returns:
        str: The generated answer.
    """
    
    prompt = build_prompt(query, context)
    
    # TODO: Instantiate your Ollama LLM
    # llm = Ollama(model=MODEL_NAME)
    
    # TODO: Invoke the LLM with the prompt and return the string response.
    # response = llm.invoke(prompt)
    # return response
    
    return "This is a dummy answer for: " + query

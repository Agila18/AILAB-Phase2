# rag/generator.py
"""
Person 5: Answer Generator
Responsibility: Generate answers using LLM with retrieved context.

ALIGNED: Uses llama3 via LangChain OllamaLLM (same as app.py).
"""

import os
from langchain_ollama import OllamaLLM
from core.config import MODEL_NAME


def format_context(docs):
    """Format retrieved documents into a numbered context string."""
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        context += f"\n[Source {i+1}] ({source}):\n{doc.page_content}\n"
    return context


def generate_answer(query, docs):
    """
    Generate an answer using the LLM with retrieved context.

    Args:
        query (str): The user's question.
        docs (list): Retrieved LangChain Document objects.

    Returns:
        str: The generated answer.
    """
    context = format_context(docs)

    # Load system prompt
    system_prompt = "You are a helpful CIT student assistant."
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "system_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            system_prompt = f.read().strip()

    prompt = f"""{system_prompt}

Context:
{context}

Question:
{query}

Answer:
"""

    llm = OllamaLLM(model=MODEL_NAME, temperature=0)
    response = llm.invoke(prompt)
    return response
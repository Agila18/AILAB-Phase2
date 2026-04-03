"""
Level 3 · Step 7 : Answer Verification
Checks whether the generated answer is grounded in the retrieved context.
"""


def verify_answer(answer: str, docs) -> bool:
    """
    Returns True if at least one word from the answer matches the context.
    Works with both list[Document] and list[str].
    """
    if not answer or not docs:
        return False

    # Build context string
    if isinstance(docs[0], str):
        context = " ".join(docs)
    else:
        context = " ".join(d.page_content for d in docs)

    context_lower = context.lower()

    # Short-circuit: known refusal phrases are always "verified"
    refusals = ["could not find", "not found", "please contact", "i am a cit"]
    if any(phrase in answer.lower() for phrase in refusals):
        return True

    # At least one non-trivial answer word must appear in context
    answer_words = [w for w in answer.lower().split() if len(w) > 3]
    return any(word in context_lower for word in answer_words)

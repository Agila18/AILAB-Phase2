from rag_engine import RAGEngine

def test_not_found_safety():
    engine = RAGEngine()
    # Ask something completely out of context to trigger low confidence
    result = engine.query("What is the square root of 2? and tell me about penguins.")
    
    print(f"Query: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence')}")
    
    if result.get('answer') == "⚠️ Answer not found in documents" and result.get('confidence') < 0.5:
        print("✅ NOT FOUND safety passed.")
    else:
        print("❌ NOT FOUND safety failed or confidence was too high.")

if __name__ == "__main__":
    test_not_found_safety()

import json
import os
import time
from rag_engine import RAGEngine

# ── RAGAS Metrics (LLM-as-a-Judge) ───────────────────────────────────────────
PROMPTS = {
    "faithfulness": """You are an Auditor. Determine if the ANSWER is faithful to the CONTEXT.
Context: {context}
Answer: {answer}
Return ONLY a score between 0.0 and 1.0. Score:""",

    "context_precision": """You are a Retrieval Analyst. How relevant are these document chunks to the user's specific QUESTION?
Question: {question}
Context: {context}
Return ONLY a score between 0.0 and 1.0. Score:""",

    "answer_correctness": """You are a Grading Expert. How semantically similar is the PREDICTED answer to the EXPECTED ground truth?
Expected: {expected}
Predicted: {predicted}
Return ONLY a score between 0.0 and 1.0. Score:"""
}

def get_llm_score(llm, prompt_type, **kwargs) -> float:
    try:
        prompt = PROMPTS[prompt_type].format(**kwargs)
        resp = llm.invoke(prompt).strip()
        # Extract float from response
        scores = [float(s) for s in resp.replace(":", " ").split() if s.replace(".", "").isdigit()]
        return scores[0] if scores else 0.0
    except:
        return 0.0

def run_evaluation():
    print("🚀 Initialising RAGAS Evaluation Suite v2.0...")
    engine = RAGEngine()
    
    with open("evaluation/test_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    total = len(dataset)
    
    print(f"📊 Running RAGAS evaluation on {total} questions...")
    
    for i, item in enumerate(dataset):
        q = item["question"]
        expected = item["expected"]
        
        print(f"[{i+1}/{total}] Q: {q}")
        res = engine.query(q)
        predicted = res["answer"]
        context_text = "\n\n".join([d.page_content for d in res.get("docs", [])])
        
        # Calculate RAGAS Triple-Metrics
        faithfulness = get_llm_score(engine.llm, "faithfulness", context=context_text, answer=predicted)
        precision = get_llm_score(engine.llm, "context_precision", question=q, context=context_text)
        correctness = get_llm_score(engine.llm, "answer_correctness", expected=expected, predicted=predicted)
        
        results.append({
            "category": item["category"],
            "question": q,
            "expected": expected,
            "predicted": predicted,
            "faithfulness": faithfulness,
            "context_precision": precision,
            "answer_correctness": correctness,
            "latency": res.get("metrics", {}).get("latency", 0.0)
        })
        
    # ── Save Results ────────────────────────────────────────────────────────
    output_path = "evaluation/eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    passed = sum(1 for r in results if r["faithfulness"] > 0.7 and r["answer_correctness"] > 0.7)
    print(f"\n✨ Evaluation Complete! Results saved to {output_path}")
    print(f"✅ Accuracy: {passed}/{total} ({int(passed/total*100) if total > 0 else 0}%)")

if __name__ == "__main__":
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
    run_evaluation()

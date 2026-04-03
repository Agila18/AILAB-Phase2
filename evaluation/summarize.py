import json
import os

def summarize_results():
    input_path = "evaluation/eval_results.json"
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run evaluation/evaluate.py first.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    total = len(results)
    avg_faithfulness = sum(r.get("faithfulness", 0) for r in results) / total if total else 0
    avg_precision = sum(r.get("context_precision", 0) for r in results) / total if total else 0
    avg_correctness = sum(r.get("answer_correctness", 0) for r in results) / total if total else 0
    avg_latency = sum(r.get("latency", 0) for r in results) / total if total else 0
    
    print("\n" + "="*95)
    print("🎓 CIT RAGAS v2.0 RESEARCH-GRADE EVALUATION SUMMARY")
    print("="*95)
    print(f"{'Category':<12} | {'Question':<35} | {'Faithful':<8} | {'Prec.':<8} | {'Correct':<8}")
    print("-" * 95)
    
    for r in results:
        q_disp = (r["question"][:32] + "..") if len(r["question"]) > 34 else r["question"]
        f = f"{int(r.get('faithfulness', 0)*100)}%"
        p = f"{int(r.get('context_precision', 0)*100)}%"
        c = f"{int(r.get('answer_correctness', 0)*100)}%"
        print(f"{r['category']:<12} | {q_disp:<35} | {f:<8} | {p:<8} | {c:<8}")
        
    print("-" * 95)
    print(f"📊 Avg Faithfulness:  {int(avg_faithfulness*100)}%")
    print(f"🎯 Avg Precision:     {int(avg_precision*100)}%")
    print(f"✅ Avg Correctness:   {int(avg_correctness*100)}%")
    print(f"⏱️  Avg Latency:       {round(avg_latency, 2)}s")
    print("="*95 + "\n")

if __name__ == "__main__":
    summarize_results()

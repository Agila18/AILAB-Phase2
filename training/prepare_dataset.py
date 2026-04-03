import json
import os

FEEDBACK_FILE = "feedback/feedback_log.jsonl"
OUTPUT_FILE   = "training/cit_expert_dataset.jsonl"

def prepare_rlhf_dataset():
    print("🧠 Starting RLHF Dataset Preparation for 'CIT Expert Model'...")
    
    if not os.path.exists(FEEDBACK_FILE):
        print(f"⚠️ No feedback file found at {FEEDBACK_FILE}. Skip.")
        return
        
    entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Only include positive RLHF feedback
                if data.get("rating") == "up":
                    entries.append({
                        "instruction": f"Answer the following question about CIT college rules: {data['query']}",
                        "input": "Context: CIT Students Handbook and Official Regulations",
                        "output": data["answer"]
                    })
            except:
                continue
                
    if not entries:
        print("📭 No positive feedback entries found to process.")
        return
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✨ Success! Generated {len(entries)} training pairs in {OUTPUT_FILE}")
    print("👉 Format: Alpaca (Instruction, Input, Output)")

if __name__ == "__main__":
    prepare_rlhf_dataset()

"""
sandbox.py — Stage 0: Run a single zero-shot prediction and evaluate baseline accuracy.
"""

from src.data import load_data_splits
from src.model import QwenInference

# Globally cached model instance to prevent duplicate engine initialization
_cached_model = None

def get_model(model_name: str) -> QwenInference:
    """Get or initialize the cached QwenInference instance."""
    global _cached_model
    if _cached_model is None:
        _cached_model = QwenInference(model_name_or_path=model_name, gpu_memory_utilization=0.7)
        _cached_model.load()
    return _cached_model

def is_program_correct(pred_raw: str, gold: str, table: list[list[str]], exe_ans: str) -> bool:
    """Verify if a generated program is correct by exact match or executed value match."""
    from src.model import extract_answer
    from src.evaluator import evaluate_program
    
    extracted = extract_answer(pred_raw)
    if not extracted:
        return False
        
    def normalize(p):
        return "".join(p.split()).lower()
        
    if normalize(extracted) == normalize(gold):
        return True
        
    if exe_ans:
        try:
            gold_val = float(exe_ans)
            pred_val = evaluate_program(extracted, table)
            if abs(pred_val - gold_val) <= 1e-4:
                return True
        except Exception:
            pass
            
    return False

def run_sandbox_prediction(model_name: str = "QuantTrio/Qwen3.5-4B-AWQ") -> tuple[str, str]:
    """
    Run a zero-shot prediction on the first training example.

    TODO: Implement the zero-shot prediction logic.
    Steps:
      1. Load dataset splits using load_data_splits(train_size=1, dev_size=1).
      2. Extract the first training example (context, question, answer).
      3. Get the loaded model instance using get_model(model_name).
      4. Format a simple zero-shot prompt template:
         "Hãy giải bài toán tài chính sau bằng cách viết chương trình dạng hàm toán học.\n\nBối cảnh:\n{passage}\n\nCâu hỏi: {question}"
      5. Build the final prompt using model.format_prompt() (with enable_thinking=True).
      6. Generate the response using model.generate_text(prompt, max_new_tokens=512, temperature=0.0).
      7. Return (raw_model_response, gold_program).
    """
    # --- YOUR CODE HERE ---
    raise NotImplementedError("run_sandbox_prediction() is not implemented yet.")

def run_sandbox_accuracy_check(model: QwenInference, dev_size: int = 50) -> dict:
    """Evaluate zero-shot baseline accuracy on the development split."""
    _, dev_split = load_data_splits(train_size=1, dev_size=dev_size)
    print(f"\nEvaluating baseline accuracy on {len(dev_split)} dev examples...")
    
    prompt_template = (
        "Hãy giải bài toán tài chính sau bằng cách viết chương trình dạng hàm toán học.\n\n"
        "Bối cảnh:\n{passage}\n\n"
        "Câu hỏi: {question}"
    )
    
    prompts = []
    for ex in dev_split:
        p = ex.get("context") or ""
        q = ex.get("question") or ""
        msg = prompt_template.format(passage=p, question=q)
        full_p = model.format_prompt(
            system_message="Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi dựa trên văn bản và bảng số liệu được cung cấp.",
            user_message=msg,
            enable_thinking=True
        )
        prompts.append(full_p)
        
    results = model.generate_batch(prompts, cot_format=True)
    
    num_correct = 0
    samples = []
    for idx, (ex, res) in enumerate(zip(dev_split, results)):
        g_prog = ex.get("answer") or ""
        table = ex.get("table") or []
        exe_ans = ex.get("exe_ans") or ""
        correct = is_program_correct(res.raw_output, g_prog, table, exe_ans)
        if correct:
            num_correct += 1
            
        samples.append({
            "index": idx,
            "passage": ex.get("context") or "",
            "question": ex.get("question") or "",
            "gold_program": g_prog,
            "pred_raw": res.raw_output,
            "is_correct": correct
        })
            
    accuracy = num_correct / len(dev_split) if dev_split else 0.0
    print(f"=== Zero-Shot Baseline Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({num_correct}/{len(dev_split)})")
    
    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_examples": len(dev_split),
        "samples": samples
    }

if __name__ == "__main__":
    try:
        pred, gold = run_sandbox_prediction()
        print("\n=== Model Output ===")
        print(pred)
        print("\n=== Gold Program ===")
        print(gold)
    except NotImplementedError:
        print("Please implement run_sandbox_prediction first!")

"""
self_reflector.py — Self-optimization reflector: uses Qwen itself as the meta-agent.

Qwen analyses its own failure cases and generates a hypothesis for improvement.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.executor import EvalResult
from src.strategy import Reflection, Strategy, CoTFormat
from pydantic import BaseModel, Field

class ReflectionSchema(BaseModel):
    accuracy_by_type: dict[str, float] = Field(..., description="Mapping from question category to accuracy (0.0 to 1.0)")
    failure_patterns: list[str] = Field(..., description="List of observed error patterns")
    hypothesis: str = Field(..., description="A concrete, testable hypothesis for why the strategy failed")
    summary: str = Field(..., description="One-paragraph prose summary of the reflection")

logger = logging.getLogger(__name__)

_SYSTEM_REFLECT = """\
Bạn là trợ lý nghiên cứu NLP đang phân tích hiệu suất của một chiến lược prompting \
trên bài toán giải toán tài chính tiếng Việt bằng cách lập chương trình (hàm toán học).

Dựa trên kết quả đánh giá và các lỗi sai, hãy:
1. Nhóm các lỗi sai thành các mẫu lỗi (failure patterns) thay vì chỉ nhìn vào loại phép toán. Các mẫu phổ biến:
   - Lỗi đơn vị/tỷ lệ (vd: kết quả cần tỷ lệ nhưng lại nhân 100 thành phần trăm)
   - Sai logic công thức (vd: câu hỏi hỏi tỷ trọng nhưng dùng phép trừ rồi chia, hoặc ngược lại)
   - Vi phạm cú pháp lồng hàm (vd: divide(table_average(...), x))
   - Trích xuất sai số (vd: lấy số lớn nhất/nổi bật nhất thay vì số đúng ngữ cảnh)
2. Phân tích nguyên nhân sâu xa vì sao prompt hiện tại gây ra các lỗi trên.
3. Đề xuất giả thuyết cải thiện có thể kiểm chứng được cho template prompt.

YÊU CẦU ĐỘ DÀI VÀ CẤU TRÚC (BẮT BUỘC):
- Viết ngắn gọn và súc tích. Tổng độ dài toàn bộ bài phân tích/suy nghĩ KHÔNG ĐƯỢC vượt quá 500 từ.
- KHÔNG lặp từ, không viết dông dài, không tự tạo ra văn bản rác hoặc ký tự lặp vô nghĩa.
- Giả thuyết: tối đa 2 câu.
- Tóm tắt (summary): tối đa 1 đoạn văn ngắn (3-4 câu)."""


def _build_reflect_message(strategy: Strategy, eval_result: EvalResult, progressive: bool = True) -> str:
    lines = [
        f"=== Chiến lược ===",
        f"CoT: {strategy.cot_format.value}",
        f"Template: {strategy.prompt_template[:300]!r}",
        f"\n=== Kết quả đánh giá ===",
        f"Độ chính xác tổng: {eval_result.accuracy:.3f} ({eval_result.num_correct}/{eval_result.num_examples})",
        "\nĐộ chính xác theo loại câu hỏi:",
    ]
    for q_type, acc in sorted(eval_result.accuracy_by_type.items(), key=lambda x: x[1]):
        count = eval_result.count_by_type.get(q_type, 0)
        lines.append(f"  {q_type}: {acc:.3f} ({count} ví dụ)")

    # Check for token budget collapse/truncation
    avg_out_tokens = 0
    if eval_result.per_question:
        avg_out_tokens = sum(r.output_tokens for r in eval_result.per_question) / len(eval_result.per_question)
    
    limit = 1024 if strategy.cot_format != CoTFormat.NONE else 256
    if avg_out_tokens >= 0.9 * limit:
        lines.append(
            "\n[CẢNH BÁO CỰC KỲ QUAN TRỌNG: SỰ CỐ GIỚI HẠN TOKEN / LẶP VÔ HẠN]\n"
            f"Số lượng token đầu ra trung bình mỗi câu hỏi ({avg_out_tokens:.1f}) đã đạt sát giới hạn tối đa ({limit}).\n"
            "Mô hình đã viết quá nhiều giải thích/suy nghĩ dông dài hoặc bị lặp vô hạn và BỊ CẮT GIỮA CHỪNG trước khi kịp trả về chương trình!\n"
            "Để khắc phục, bạn PHẢI tắt Chain-of-Thought (chuyển cot_format thành 'none') HOẶC giới hạn nghiêm ngặt độ dài suy nghĩ (tối đa 2 câu) "
            "và yêu cầu trả về trực tiếp định dạng PROGRAM:.\n"
        )

    # ----------------------------------------------------------------
    # Progressive Context Management
    # ----------------------------------------------------------------
    # TODO: Implement progressive decay for top_k failures based on iteration.
    # - If progressive is True:
    #   - For iteration <= 1: select top_k = 5 failures
    #   - For iteration == 2: select top_k = 3 failures
    #   - For iteration >= 3: select top_k = 1 failure
    # - Else:
    #   - select top_k = 5 failures
    # ----------------------------------------------------------------
    iteration = strategy.metadata.iteration
    if progressive:
        if iteration <= 1:
            top_k = 5
        elif iteration == 2:
            top_k = 3
        else:
            top_k = 1
    else:
        top_k = 5

    logger.info("Self-Reflector Progressive Context: iteration=%d, selected top_k=%d failures.", iteration, top_k)
    failures = eval_result.failures(top_k=top_k)
    lines.append("\n=== Các lỗi sai tiêu biểu ===")
    for i, f in enumerate(failures, 1):
        if progressive and iteration >= 3:
            passage_text = f.passage[:200] + "..." if len(f.passage) > 200 else f.passage
            output_text = f.raw_output[:100] + "..." if len(f.raw_output) > 100 else f.raw_output
            lines.append(
                f"\nLỗi {i}:\n"
                f"  Ngữ cảnh (Đoạn văn rút gọn): {passage_text}\n"
                f"  Câu hỏi: {f.question}\n"
                f"  Đúng: {f.gold_answer} (Giá trị: {f.gold_val}) | Dự đoán: {f.predicted_answer} (Giá trị: {f.predicted_val})\n"
                f"  Output: {output_text}"
            )
        else:
            lines.append(
                f"\nLỗi {i}:\n"
                f"  Ngữ cảnh (Đoạn văn/Bảng): {f.passage[:1000]}...\n"
                f"  Câu hỏi: {f.question}\n"
                f"  Đúng: {f.gold_answer} (Giá trị: {f.gold_val}) | Dự đoán: {f.predicted_answer} (Giá trị: {f.predicted_val})\n"
                f"  Output: {f.raw_output[:150]}"
            )

    return "\n".join(lines)


def reflect_self(
    strategy: Strategy,
    eval_result: EvalResult,
    model,  # QwenInference
    max_retries: int = 5,
    progressive: bool = True,
) -> tuple[Reflection, int]:
    """
    Use the Qwen inference model itself to reflect on evaluation results.

    TODO: Implement the reflection loop with Pydantic schema validation.
    Steps:
      1. Build reflect message.
      2. Call model.generate_text to perform a first-pass free-form analysis.
      3. Clean thinking tags from the output.
      4. Perform a second-pass coercion request using model.generate_text(..., guided_json=ReflectionSchema.model_json_schema()).
      5. Validate and parse the returned JSON into the Reflection class.
      6. Return Reflection object and estimated token usage.
    """
    reflect_message = _build_reflect_message(strategy, eval_result, progressive=progressive)
    raw_response = ""
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            analysis_prompt = model.format_prompt(
                system_message=_SYSTEM_REFLECT,
                user_message=reflect_message,
                enable_thinking=True,
            )
            raw_analysis = model.generate_text(
                analysis_prompt,
                max_new_tokens=1024,
                temperature=0.7,
            )
            cleaned_analysis = _strip_thinking(raw_analysis)

            coercion_message = (
                "Chuyển phân tích sau thành JSON hợp lệ theo schema ReflectionSchema. "
                "Chỉ trả về JSON, không thêm markdown hoặc giải thích.\n\n"
                f"Schema fields: accuracy_by_type, failure_patterns, hypothesis, summary.\n\n"
                f"Độ chính xác theo loại hiện có: {json.dumps(eval_result.accuracy_by_type, ensure_ascii=False)}\n\n"
                f"Phân tích:\n{cleaned_analysis}"
            )
            coercion_prompt = model.format_prompt(
                system_message=_SYSTEM_REFLECT,
                user_message=coercion_message,
                enable_thinking=False,
            )
            raw_response = model.generate_text(
                coercion_prompt,
                max_new_tokens=512,
                temperature=0.0,
                guided_json=ReflectionSchema.model_json_schema(),
            )

            parsed = _parse_reflection_schema(raw_response)
            top_failures = [_failure_to_dict(f) for f in eval_result.failures(top_k=5)]
            reflection = Reflection(
                strategy_id=strategy.id,
                accuracy_by_type=parsed.accuracy_by_type,
                top_failures=top_failures,
                hypothesis=parsed.hypothesis,
                summary=parsed.summary,
                raw_response=raw_response,
            )
            token_usage = _estimate_meta_tokens(
                model,
                analysis_prompt,
                raw_analysis,
                coercion_prompt,
                raw_response,
            )
            return reflection, token_usage
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Reflection attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                exc,
            )

    logger.warning("Falling back to heuristic reflection after parse failures: %s", last_error)
    return _fallback_reflection(strategy, eval_result, raw_response), 0


def _strip_thinking(text: str) -> str:
    """Remove Qwen-style thinking blocks and return concise visible content."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty reflection response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in reflection response")
    return json.loads(match.group(0))


def _parse_reflection_schema(text: str) -> ReflectionSchema:
    data = _extract_json_object(_strip_thinking(text))
    return ReflectionSchema.model_validate(data)


def _failure_to_dict(failure) -> dict:
    return {
        "question_id": failure.question_id,
        "question": failure.question,
        "gold_answer": failure.gold_answer,
        "predicted_answer": failure.predicted_answer,
        "question_type": failure.question_type,
        "gold_val": failure.gold_val,
        "predicted_val": failure.predicted_val,
        "raw_output": failure.raw_output,
    }


def _estimate_meta_tokens(model, *texts: str) -> int:
    total = 0
    for text in texts:
        if not text:
            continue
        try:
            total += int(model.count_tokens(text))
        except Exception:
            total += max(1, len(text.split()))
    return total


def _fallback_reflection(strategy: Strategy, eval_result: EvalResult, raw_response: str = "") -> Reflection:
    weakest_type = "unknown"
    if eval_result.accuracy_by_type:
        weakest_type = min(eval_result.accuracy_by_type, key=eval_result.accuracy_by_type.get)

    top_failures = [_failure_to_dict(f) for f in eval_result.failures(top_k=5)]
    failure_patterns = sorted({f.get("question_type", "unknown") for f in top_failures}) or ["unknown"]
    hypothesis = (
        f"Chiến lược hiện tại yếu nhất ở nhóm {weakest_type}; cần làm rõ cách trích xuất số, "
        "chọn phép toán và định dạng DSL để giảm lỗi."
    )
    summary = (
        "Fallback reflection: không thể parse JSON từ meta-agent, nên dùng thống kê "
        f"đánh giá hiện có. Accuracy tổng là {eval_result.accuracy:.3f}; "
        f"các mẫu lỗi nổi bật: {', '.join(failure_patterns)}."
    )
    return Reflection(
        strategy_id=strategy.id,
        accuracy_by_type=dict(eval_result.accuracy_by_type),
        top_failures=top_failures,
        hypothesis=hypothesis,
        summary=summary,
        raw_response=raw_response,
    )

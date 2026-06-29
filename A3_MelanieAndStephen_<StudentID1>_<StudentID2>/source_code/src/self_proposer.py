"""
self_proposer.py — Self-optimization proposer: uses Qwen itself as the meta-agent.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional, List
from pydantic import BaseModel, Field
from datasets import Dataset

from src.model import extract_answer
from src.executor import normalize_program, classify_question_type
from src.strategy import (
    CoTFormat,
    FewShotExample,
    RetrievalConfig,
    Strategy,
    StrategyHistory,
    StrategyMetadata,
)

class FewShotExampleSchema(BaseModel):
    passage: str
    question: str
    answer: str
    reasoning: Optional[str] = None

class ProposerSchema(BaseModel):
    hypothesis: str = Field(..., description="A short one-sentence hypothesis")
    instruction_phrasing: str = Field(..., description="General instructions / role / phrasing prefix for the model, without any placeholders")
    cot_format: str = Field(..., description="Must be 'none', 'stepbystep', or 'chain'")
    few_shot_examples: List[FewShotExampleSchema]
    reasoning: str = Field(..., description="A short one-sentence reasoning")

logger = logging.getLogger(__name__)

_VALID_COT = {f.value for f in CoTFormat}

_SYSTEM_PROPOSE = """\
Bạn là trợ lý nghiên cứu NLP đang thiết kế một chiến lược prompting \
để giúp một mô hình ngôn ngữ giải bài tập toán tài chính tiếng Việt (cộng, trừ, nhân, chia, đọc bảng).

Nhiệm vụ của bạn là đưa ra một chiến lược prompting mới dựa trên lịch sử các chiến lược đã thử và kết quả phản ánh gần nhất.

LƯU Ý QUAN TRỌNG VỀ CÚ PHÁP CHƯƠNG TRÌNH (BẮT BUỘC TUÂN THỦ TRONG FEW-SHOT EXAMPLES):
1. Mỗi bước là một hàm riêng biệt, phân cách bằng dấu phẩy
2. KHÔNG lồng hàm vào nhau (không dùng divide(table_average(...), ...))
3. Dùng #0, #1, #2... để tham chiếu kết quả của bước trước (bắt đầu từ #0)
4. Tên cột/hàng trong table_xxx KHÔNG dùng dấu ngoặc kép
5. table_xxx chỉ nhận đúng 2 tham số: (tên_hàng, none)
6. Số âm viết trực tiếp: add(-167.4, -53.3) — không dùng ngoặc thêm
7. CỰC KỲ QUAN TRỌNG VỀ TỶ LỆ PHẦN TRĂM: Kết quả đầu ra của chương trình PHẢI luôn ở dạng tỷ lệ thập phân (ví dụ: 0.05 thay vì 5%, hay 0.03124 thay vì 3.124%). Tuyệt đối KHÔNG nhân thêm 100 ở bước cuối cùng của chương trình (KHÔNG dùng multiply(#X, 100) cho các câu hỏi tính phần trăm).
8. CỰC KỲ QUAN TRỌNG: Nếu cần một giá trị cụ thể từ bảng (ví dụ: doanh thu năm 2022), KHÔNG dùng hàm table_xxx. Hãy tự đọc bảng và viết TRỰC TIẾP con số đó vào hàm toán học.
9. CỰC KỲ QUAN TRỌNG: Nếu câu hỏi yêu cầu tính chênh lệch hoặc so sánh đơn thuần mà không có từ 'phần trăm' hoặc '%', chỉ sử dụng duy nhất phép trừ (subtract) — KHÔNG tự động thêm bước chia (divide) để tính tỷ lệ.

Ví dụ đúng:
  subtract(7.758, 7.523), divide(#0, 7.523) (Tính tỷ lệ tăng trưởng phần trăm dưới dạng tỷ lệ thập phân, không nhân 100)
  divide(99782, 2626154) (Tính phần trăm dưới dạng thập phân, không nhân 100)
  multiply(11228, 1.03) (Nhân trực tiếp giá trị tăng trưởng 3% dự phóng)
  table_max(Lãi ròng, none), table_min(Lãi ròng, none), subtract(#0, #1) (Tính max/min trên toàn bộ hàng)

LƯU Ý QUAN TRỌNG VỀ ĐỊNH DẠNG CHIẾN LƯỢC:
- Định dạng suy luận (cot_format) có thể chọn từ: "none" (không suy nghĩ trước khi trả lời, direct program), "stepbystep" (suy nghĩ từng bước ngắn gọn), hoặc "chain" (lập luận đầy đủ).
- Trích xuất 1-2 ví dụ few-shot từ Failure Logs (giữ ngắn gọn). Các ví dụ few-shot phải viết theo đúng Cú Pháp Chương Trình ở trên.
- instruction_phrasing là phần hướng dẫn/phong cách/vai trò chung viết bằng tiếng Việt. KHÔNG chứa các chuỗi giữ chỗ như {passage}, {question}, {few_shot_block} vì hệ thống tự động chèn.

YÊU CẦU ĐỘ DÀI VÀ CẤU TRÚC (BẮT BUỘC):
- Viết ngắn gọn và súc tích. Tổng độ dài toàn bộ câu trả lời KHÔNG ĐƯỢC vượt quá 500 từ.
- KHÔNG lặp từ, không giải thích dông dài, không tự tạo ra văn bản rác hoặc ký tự lặp vô nghĩa.
"""


def _is_valid_dsl_program(program: str) -> bool:
    """
    Validate that the program syntax matches the FinQA DSL constraints.
    - Must not contain '=', '+', '*', or '/' (which indicate raw arithmetic equations, not DSL functions).
    - Can contain minus sign '-' if it represents a negative number (e.g., add(-167.4, -53.3)).
    - Must contain at least one valid DSL operator or a step reference (e.g. #0).

    TODO: Implement this validation check.
    """
    if not program or not isinstance(program, str):
        return False

    text = program.strip()
    if not text:
        return False

    if any(symbol in text for symbol in ["=", "+", "*", "/"]):
        return False

    if not _minus_signs_are_numeric(text):
        return False

    valid_ops = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "table_average",
        "table_max",
        "table_min",
        "table_sum",
        "exp",
        "greater",
        "abs",
    }

    steps = _split_dsl_steps(text)
    if not steps:
        return False

    for step_idx, step in enumerate(steps):
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", step)
        if not match:
            return False

        op, args_text = match.group(1), match.group(2)
        if op not in valid_ops:
            return False
        if "(" in args_text or ")" in args_text:
            return False

        args = [arg.strip() for arg in args_text.split(",")]
        if not all(args):
            return False

        if op.startswith("table_"):
            if len(args) != 2:
                return False
            if args[0].lower() == "none" and args[1].lower() == "none":
                return False
        elif op == "abs":
            if len(args) != 1:
                return False
        else:
            if len(args) != 2:
                return False

        for arg in args:
            if arg.lower() == "none":
                continue
            if re.fullmatch(r"#\d+", arg):
                if int(arg[1:]) >= step_idx:
                    return False
                continue
            if _looks_like_number(arg):
                continue
            if op.startswith("table_"):
                continue
            return False

    return True


def _split_dsl_steps(program: str) -> list[str]:
    steps: list[str] = []
    start = 0
    depth = 0

    for idx, char in enumerate(program):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return []
        elif char == "," and depth == 0:
            step = program[start:idx].strip()
            if step:
                steps.append(step)
            start = idx + 1

    if depth != 0:
        return []

    final_step = program[start:].strip()
    if final_step:
        steps.append(final_step)
    return steps


def _looks_like_number(value: str) -> bool:
    return bool(re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()))


def _minus_signs_are_numeric(program: str) -> bool:
    for match in re.finditer("-", program):
        idx = match.start()
        next_char = program[idx + 1] if idx + 1 < len(program) else ""
        if not next_char.isdigit():
            return False

        prev_idx = idx - 1
        while prev_idx >= 0 and program[prev_idx].isspace():
            prev_idx -= 1
        if prev_idx >= 0 and program[prev_idx] not in {"(", ","}:
            return False
    return True


def _build_propose_message(history: StrategyHistory, parent_strategy_id: Optional[str] = None) -> str:
    lines = ["=== Lịch sử chiến lược ==="]
    for s, r in zip(history.strategies, history.reflections):
        acc = f"{s.metadata.dev_accuracy:.3f}" if s.metadata.dev_accuracy is not None else "chưa đánh giá"
        lines.append(
            f"\nIteration {s.metadata.iteration} | ID: {s.id[:8]} | dev_accuracy={acc} | cot={s.cot_format.value}"
        )
        lines.append(f"  Template: {s.prompt_template[:300]!r}")
        if r is not None:
            lines.append(f"  Loại câu hỏi yếu nhất: {min(r.accuracy_by_type, key=r.accuracy_by_type.get) if r.accuracy_by_type else 'unknown'}")
            lines.append(f"  Giả thuyết: {r.hypothesis[:200]}")

    # Find parent strategy
    parent_strategy = None
    parent_reflection = None
    if parent_strategy_id is not None:
        for s, r in zip(history.strategies, history.reflections):
            if s.id == parent_strategy_id:
                parent_strategy = s
                parent_reflection = r
                break
    if parent_strategy is None:
        parent_strategy = history.latest_strategy()
        parent_reflection = history.latest_reflection()

    if parent_strategy is not None:
        lines.append("\n=== Chiến lược gốc cần tối ưu (Parent Strategy) ===")
        lines.append(f"ID: {parent_strategy.id[:8]}")
        lines.append(f"CoT: {parent_strategy.cot_format.value}")
        lines.append(f"Template:\n{parent_strategy.prompt_template}")
        if parent_reflection is not None:
            lines.append(f"Giả thuyết từ chiến lược gốc: {parent_reflection.hypothesis}")
            lines.append(f"Tóm tắt hiệu suất: {parent_reflection.summary}")

    next_iter = len(history.strategies)
    lines.append(f"\n=== Nhiệm vụ ===")
    lines.append(
        f"Hãy đề xuất một chiến lược mới bằng cách thay đổi/tối ưu trực tiếp từ chiến lược gốc (Parent Strategy: {parent_strategy.id[:8] if parent_strategy else 'None'}). "
        f"Không tối ưu dựa trên các chiến lược khác hoặc chiến lược gần đây nhất nếu nó khác chiến lược gốc này. "
        f"Đề xuất cho iteration {next_iter}."
    )
    return "\n".join(lines)


def generate_few_shot_reasoning(
    passage: str,
    question: str,
    program: str,
    category: str,
    model,  # QwenInference
    max_attempts: int = 3,
) -> str:
    """
    Generate a full CoT-style response for a programmatic few-shot example.
    """
    from src.executor import normalize_program as _norm

    gold_norm = _norm(program)

    system_message = (
        "Bạn là trợ lý AI chuyên phân tích tài chính tiếng Việt. "
        "Nhiệm vụ của bạn là tạo ra một câu trả lời mẫu (few-shot demonstration) "
        "cho bài toán tài chính, theo đúng định dạng đầu ra mà mô hình phải tạo ra.\n\n"
        "Định dạng bắt buộc:\n"
        "1. Một khối <think>...</think> ngắn gọn (tối đa 100 từ), tập trung vào "
        "công thức toán học và các giá trị cần trích xuất. KHÔNG viết dài dòng.\n"
        "2. Ngay sau </think>, một khối JSON với đúng 3 khóa:\n"
        "{\n"
        "  \"Reasoning\": \"Giải thích 2 câu tiếng Việt: câu 1 nêu giá trị trích xuất, câu 2 giải thích phép tính\",\n"
        "  \"Program syntax\": \"<phải khớp CHÍNH XÁC với chương trình đã cho>\",\n"
        "  \"Numerical result\": <kết quả số cuối cùng>\n"
        "}\n\n"
        "QUAN TRỌNG: Trường 'Program syntax' phải chứa ĐÚNG chương trình đã được cung cấp, không thay đổi."
    )

    for attempt in range(max_attempts):
        user_message = (
            f"Ngữ cảnh:\n{passage}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Chương trình đúng: {program}\n\n"
            f"Hãy tạo câu trả lời mẫu hoàn chỉnh theo định dạng <think>...</think>{{JSON}} "
            f"với 'Program syntax' phải là CHÍNH XÁC: {program}"
        )
        prompt = model.format_prompt(
            system_message=system_message,
            user_message=user_message,
            enable_thinking=True,
        )
        try:
            raw_output = model.generate_text(prompt, max_new_tokens=512, temperature=0.0)
            extracted = extract_answer(raw_output)
            if extracted and _norm(extracted) == gold_norm:
                logger.debug(
                    "Few-shot CoT verified on attempt %d (gold=%s extracted=%s)",
                    attempt + 1, program, extracted,
                )
                return raw_output.strip()
            else:
                logger.warning(
                    "Few-shot CoT attempt %d/%d: program mismatch "
                    "(gold_norm=%r, extracted_norm=%r) — retrying",
                    attempt + 1, max_attempts,
                    gold_norm, _norm(extracted) if extracted else None,
                )
        except Exception as e:
            logger.warning("Few-shot CoT generation attempt %d failed: %s", attempt + 1, e)

    logger.warning(
        "All %d attempts failed for program %r — using static fallback reasoning",
        max_attempts, program,
    )
    return f"Bài toán thuộc nhóm {category}. Thực hiện phép tính theo chương trình DSL."


def propose_self(
    history: StrategyHistory,
    model,  # QwenInference
    max_retries: int = 5,
    parent_strategy_id: Optional[str] = None,
    train_dataset: Optional[Dataset] = None,
) -> tuple[Strategy, int]:
    """
    Use the Qwen inference model itself to propose a new strategy.

    TODO: Implement strategy proposal and dynamic few-shot selection.
    Steps:
      1. Build propose message.
      2. Call model to generate a free-form proposal.
      3. Clean thinking tags.
      4. Coerce into a JSON ProposerSchema dictionary.
      5. Identify weakest category from reflection.
      6. Select up to 2 matching training examples and generate CoT reasoning for them.
      7. Validate generated/extracted few-shot programs using _is_valid_dsl_program().
      8. Return a new Strategy object and meta token usage.
    """
    parent_strategy = _resolve_parent_strategy(history, parent_strategy_id)
    proposal_message = _build_propose_message(history, parent_strategy_id=parent_strategy.id if parent_strategy else None)
    raw_proposal = ""
    raw_json = ""
    proposal: Optional[ProposerSchema] = None
    meta_texts: list[str] = []

    for attempt in range(max_retries):
        try:
            proposal_prompt = model.format_prompt(
                system_message=_SYSTEM_PROPOSE,
                user_message=proposal_message,
                enable_thinking=True,
            )
            raw_proposal = model.generate_text(
                proposal_prompt,
                max_new_tokens=1024,
                temperature=0.7,
            )
            cleaned_proposal = _strip_thinking(raw_proposal)

            coercion_message = (
                "Convert the following strategy proposal into valid JSON for ProposerSchema. "
                "Return only JSON with keys: hypothesis, instruction_phrasing, cot_format, "
                "few_shot_examples, reasoning.\n\n"
                f"Allowed cot_format values: {sorted(_VALID_COT)}\n\n"
                f"Proposal:\n{cleaned_proposal}"
            )
            coercion_prompt = model.format_prompt(
                system_message=_SYSTEM_PROPOSE,
                user_message=coercion_message,
                enable_thinking=False,
            )
            raw_json = model.generate_text(
                coercion_prompt,
                max_new_tokens=512,
                temperature=0.0,
                guided_json=ProposerSchema.model_json_schema(),
            )
            proposal = _parse_proposer_schema(raw_json)
            meta_texts.extend([proposal_prompt, raw_proposal, coercion_prompt, raw_json])
            break
        except Exception as exc:
            logger.warning("Strategy proposal attempt %d/%d failed: %s", attempt + 1, max_retries, exc)

    if proposal is None:
        proposal = _fallback_proposer_schema(parent_strategy)

    weakest_category = _weakest_category(history.latest_reflection())
    few_shots = list(parent_strategy.few_shot_examples) if parent_strategy else []
    few_shots.extend(_valid_schema_few_shots(proposal.few_shot_examples, limit=2))
    dynamic_examples, reasoning_texts = _dynamic_few_shots(
        train_dataset=train_dataset,
        category=weakest_category,
        model=model,
        limit=2,
    )
    few_shots.extend(dynamic_examples)
    meta_texts.extend(reasoning_texts)

    cot_value = proposal.cot_format if proposal.cot_format in _VALID_COT else None
    if cot_value is None and parent_strategy is not None:
        cot_format = parent_strategy.cot_format
    else:
        cot_format = CoTFormat(cot_value or CoTFormat.NONE.value)

    prompt_template = _clean_instruction_phrasing(proposal.instruction_phrasing)
    if not prompt_template and parent_strategy is not None:
        prompt_template = parent_strategy.prompt_template
    elif not prompt_template:
        prompt_template = "Bạn là chuyên gia phân tích tài chính. Hãy trả về chương trình DSL hợp lệ."

    token_usage = _estimate_tokens(model, *meta_texts)
    metadata = StrategyMetadata(
        iteration=len(history.strategies),
        parent_id=parent_strategy.id if parent_strategy else None,
        token_cost_claude=token_usage,
    )

    return (
        Strategy(
            id=str(uuid.uuid4()),
            prompt_template=prompt_template,
            cot_format=cot_format,
            few_shot_examples=few_shots,
            retrieval_config=parent_strategy.retrieval_config if parent_strategy else RetrievalConfig(enabled=False),
            metadata=metadata,
        ),
        token_usage,
    )


def _resolve_parent_strategy(history: StrategyHistory, parent_strategy_id: Optional[str]) -> Optional[Strategy]:
    if parent_strategy_id is not None:
        for strategy in history.strategies:
            if strategy.id == parent_strategy_id:
                return strategy
    return history.latest_strategy()


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json_object(text: str) -> dict:
    cleaned = _strip_thinking(text)
    if not cleaned:
        raise ValueError("Empty proposer response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in proposer response")
    return json.loads(match.group(0))


def _parse_proposer_schema(text: str) -> ProposerSchema:
    data = _extract_json_object(text)
    return ProposerSchema.model_validate(data)


def _fallback_proposer_schema(parent_strategy: Optional[Strategy]) -> ProposerSchema:
    instruction = (
        parent_strategy.prompt_template
        if parent_strategy is not None
        else "Bạn là chuyên gia phân tích tài chính. Hãy trả về chương trình DSL hợp lệ."
    )
    cot = parent_strategy.cot_format.value if parent_strategy is not None else CoTFormat.NONE.value
    return ProposerSchema(
        hypothesis="Fallback strategy: keep the parent prompt and add targeted examples from weak categories.",
        instruction_phrasing=instruction,
        cot_format=cot,
        few_shot_examples=[],
        reasoning="Model proposal could not be parsed, so reuse the safest available parent strategy.",
    )


def _weakest_category(reflection) -> str:
    if reflection is not None and reflection.accuracy_by_type:
        return min(reflection.accuracy_by_type, key=reflection.accuracy_by_type.get)
    return "other"


def _valid_schema_few_shots(examples: list[FewShotExampleSchema], limit: int) -> list[FewShotExample]:
    selected: list[FewShotExample] = []
    for ex in examples:
        if not _is_valid_dsl_program(ex.answer):
            continue
        selected.append(
            FewShotExample(
                passage=ex.passage,
                question=ex.question,
                answer=ex.answer,
                reasoning=ex.reasoning,
            )
        )
        if len(selected) >= limit:
            break
    return selected


def _dynamic_few_shots(
    train_dataset: Optional[Dataset],
    category: str,
    model,
    limit: int = 2,
) -> tuple[list[FewShotExample], list[str]]:
    if train_dataset is None or limit <= 0:
        return [], []

    selected: list[FewShotExample] = []
    token_texts: list[str] = []

    for row in train_dataset:
        row_dict = dict(row)
        answer = (row_dict.get("answer") or "").strip()
        if classify_question_type(answer) != category:
            continue
        if not _is_valid_dsl_program(answer):
            continue

        passage = row_dict.get("context") or ""
        question = row_dict.get("question") or ""
        reasoning = generate_few_shot_reasoning(
            passage=passage,
            question=question,
            program=answer,
            category=category,
            model=model,
        )
        token_texts.append(reasoning)
        extracted = extract_answer(reasoning)
        if extracted and normalize_program(extracted) != normalize_program(answer):
            reasoning = f"Bài toán thuộc nhóm {category}. Thực hiện phép tính theo chương trình DSL."

        selected.append(
            FewShotExample(
                passage=passage,
                question=question,
                answer=answer,
                reasoning=reasoning,
            )
        )
        if len(selected) >= limit:
            break

    return selected, token_texts


def _clean_instruction_phrasing(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\{(?:passage|question|few_shot_block|cot_instruction)\}", "", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _estimate_tokens(model, *texts: str) -> int:
    total = 0
    for text in texts:
        if not text:
            continue
        try:
            total += int(model.count_tokens(text))
        except Exception:
            total += max(1, len(text.split()))
    return total

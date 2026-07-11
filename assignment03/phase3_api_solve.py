"""
phase3_api_solve.py — solve FinQA-VN with a hosted API model (DeepSeek / Gemini).

Allowed by Phase-3 rules 2 & 3 (external / hosted API models are permitted as part
of a documented, reproducible pipeline). Frontier models are far stronger than the
9B open models, so this is a candidate new best or a strong diverse ensemble voter.

Dependency-free: raw HTTPS via urllib (no SDK install). Reads keys from .env.
Reuses the project's DSL prompt (build_prompt + injected DSL rules) and executes
each generated program with the DSL evaluator, exactly like the local pipeline.

Usage:
  python3 phase3_api_solve.py --provider deepseek --model deepseek-chat --split dev
  python3 phase3_api_solve.py --provider gemini   --model gemini-2.0-flash --split test
Outputs: predict_out/api-<provider>-<model>_<split>.csv (+ _details.json)
For dev it also prints accuracy vs gold.
"""
from __future__ import annotations
import argparse, csv, json, re, ssl, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.data import _load_json_split          # noqa: E402
from src.executor import build_prompt          # noqa: E402
from src.evaluator import evaluate_program     # noqa: E402
from src.strategy import Strategy              # noqa: E402

SYSTEM = (
    "Bạn là một trợ lý AI chuyên phân tích tài chính tiếng Việt. "
    "Nhiệm vụ của bạn là viết chương trình dạng các hàm toán học để trả lời câu hỏi "
    "dựa trên văn bản và bảng số liệu được cung cấp."
)


def load_env(path=HERE / ".env"):
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def tof(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_program(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"PROGRAM:\s*(.+)", text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    # fallback: last line containing a DSL op call
    for line in reversed(text.strip().splitlines()):
        if re.search(r"(add|subtract|multiply|divide|table_\w+)\s*\(", line):
            return line.strip().strip("`").strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _post(url, payload, headers, timeout=90):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def call_deepseek(key, model, system, user, max_tokens=512, temperature=0.0):
    url = "https://api.deepseek.com/chat/completions"
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = _post(url, payload, headers)
    return resp["choices"][0]["message"]["content"]


def call_gemini(key, model, system, user, max_tokens=512, temperature=0.0):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": system + "\n\n" + user}]}],
               "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
    headers = {"Content-Type": "application/json"}
    resp = _post(url, payload, headers)
    cand = resp["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def call_openai(key, model, system, user, max_tokens=512, temperature=0.0):
    url = "https://api.openai.com/v1/chat/completions"
    # Newer OpenAI models (o-series, gpt-5) reject temperature!=1 and use
    # max_completion_tokens; older ones use max_tokens. Try modern first.
    base = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = _post(url, {**base, "max_completion_tokens": max_tokens, "temperature": temperature}, headers)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        resp = _post(url, {**base, "temperature": temperature, "max_tokens": max_tokens}, headers)
    return resp["choices"][0]["message"]["content"]


def call_anthropic(key, model, system, user, max_tokens=512, temperature=0.0):
    url = "https://api.anthropic.com/v1/messages"
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    resp = _post(url, payload, headers)
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


CALLERS = {"deepseek": call_deepseek, "gemini": call_gemini,
           "openai": call_openai, "anthropic": call_anthropic}


def _sample_once(row, user, provider, key, model, temperature, max_tokens, retries=3):
    """One retrying API call -> (raw, program, executed_value|None)."""
    last = ""
    for attempt in range(retries):
        try:
            raw = CALLERS[provider](key, model, SYSTEM, user, max_tokens, temperature)
            prog = parse_program(raw)
            try:
                val = evaluate_program(prog, row["table"])
            except Exception:
                val = None
            return raw, prog, val
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            time.sleep(2 * (attempt + 1) + (5 if e.code == 429 else 0))
        except Exception as e:
            last = str(e)[:80]
            time.sleep(2 * (attempt + 1))
    return f"ERROR:{last}", "", None


def solve_one(row, strat, provider, key, model, retries=3, max_tokens=512,
              sc_k=1, sc_temp=0.7):
    user = build_prompt(strat, row["context"], row["question"])
    if sc_k <= 1:
        raw, prog, val = _sample_once(row, user, provider, key, model, 0.0, max_tokens, retries)
        return {"id": row["id"], "question": row["question"], "raw_output": raw,
                "program": prog, "predicted_value": val if val is not None else 0.0}
    # self-consistency: k samples at sc_temp, majority-vote by executed value
    samples = [_sample_once(row, user, provider, key, model, sc_temp, max_tokens, retries)
               for _ in range(sc_k)]
    groups = []  # {"val","count","prog","raw"}
    for raw, prog, val in samples:
        if val is None:
            continue
        for g in groups:
            if abs(g["val"] - val) <= 1e-4:
                g["count"] += 1
                break
        else:
            groups.append({"val": val, "count": 1, "prog": prog, "raw": raw})
    if groups:
        best = max(groups, key=lambda g: g["count"])
        return {"id": row["id"], "question": row["question"], "raw_output": best["raw"],
                "program": best["prog"], "predicted_value": best["val"],
                "sc_votes": best["count"], "sc_k": sc_k}
    raw, prog, _ = samples[0]
    return {"id": row["id"], "question": row["question"], "raw_output": raw,
            "program": prog, "predicted_value": 0.0, "sc_votes": 0, "sc_k": sc_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["deepseek", "gemini", "openai", "anthropic"], required=True)
    ap.add_argument("--model", required=True, help="deepseek-chat | gemini-2.5-flash | gpt-5 / gpt-4.1 | claude-sonnet-5 / claude-opus-4-8")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--strategy-path", default="strategies/table_aware_strategy.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="raise to ~4096-8192 for reasoning models (R1, gemini-2.5-pro, o-series)")
    ap.add_argument("--sc-k", type=int, default=1,
                    help="self-consistency: sample k programs at --sc-temp and majority-vote by executed value")
    ap.add_argument("--sc-temp", type=float, default=0.7, help="sampling temperature when --sc-k > 1")
    args = ap.parse_args()

    env = load_env()
    key_names = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY",
                 "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    key = env.get(key_names[args.provider])
    if not key:
        sys.exit(f"missing {key_names[args.provider]} for {args.provider} in .env")

    strat = Strategy.from_json(Path(args.strategy_path).read_text(encoding="utf-8"))
    fname = "dev.json" if args.split == "dev" else "test.json"
    rows = _load_json_split(HERE / "data" / fname)
    if args.limit:
        rows = rows[:args.limit]

    sc_note = f" sc-k={args.sc_k}@T{args.sc_temp}" if args.sc_k > 1 else ""
    print(f"Solving {len(rows)} {args.split} rows with {args.provider}/{args.model} "
          f"(concurrency {args.concurrency}{sc_note})...")
    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(solve_one, r, strat, args.provider, key, args.model, 3,
                          args.max_tokens, args.sc_k, args.sc_temp): r["id"] for r in rows}
        for f in as_completed(futs):
            res = f.result()
            results[res["id"]] = res
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(rows)}")

    tag = f"api-{args.provider}-{args.model}".replace("/", "-")
    if args.sc_k > 1:
        tag += f"-sc{args.sc_k}"
    out = HERE / "predict_out"
    out.mkdir(exist_ok=True)
    ordered = [results[r["id"]] for r in rows]
    with (out / f"{tag}_{args.split}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Usage", "predicted_value"])
        for r in ordered:
            w.writerow([r["id"], "Public", r["predicted_value"]])
    (out / f"{tag}_{args.split}_details.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.split == "dev":
        gold = {r["id"]: tof(r["exe_ans"]) for r in rows}
        ok = sum(1 for r in ordered if gold.get(r["id"]) is not None
                 and tof(r["predicted_value"]) is not None
                 and abs(tof(r["predicted_value"]) - gold[r["id"]]) <= 1e-4)
        n = sum(1 for g in gold.values() if g is not None)
        print(f"\nDEV accuracy: {ok}/{n} = {100*ok/n:.2f}%   ({tag})")
    print(f"wrote predict_out/{tag}_{args.split}.csv")


if __name__ == "__main__":
    main()

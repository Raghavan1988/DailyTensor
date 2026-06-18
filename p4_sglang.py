from __future__ import annotations

import sglang as sgl


# Regex tips:
#  - Anchor implicitly via what comes before in the prompt — SGLang's regex
#    is applied to the upcoming gen() output ONLY, not to the whole stream.
#  - Don't use overly broad classes like `.*` — they make the FSM huge and
#    burn through max_tokens. Be specific.
NUMERIC_ANSWER_RE = r"-?\d+(\.\d+)?"               # 42, -3.5, etc.
FUNCTION_SIG_RE   = r"def [a-z_][a-z_0-9]*\([^)]*\)( -> [^:]+)?:"
CONFIDENCE_RE     = r"(0\.\d{2}|1\.00)"            # 0.00 .. 1.00


@sgl.function
def tutor_pipeline(s, question: str):
    """One-shot tutoring evaluator with classification + branched generation."""
    # ----- Step 1: shared prefix -----
    # Everything from here to the first fork/branch is a single radix-tree
    # entry. Two requests sharing the same question literally share these
    # KV blocks on the server — even without explicit fork().
    s += sgl.system(
        "You are an expert tutor. Be concise and pedagogically correct. "
        "Match the requested output format exactly."
    )
    s += sgl.user(question)
    s += sgl.assistant_begin()

    # ----- Step 2: classify -----
    # sgl.select constrains decoding to one of the listed choices via
    # joint-logprob scoring. It's effectively a deterministic classifier
    # head built from your own decoder model — no extra parameters.
    s += "Category: " + sgl.select("kind", choices=["math", "coding", "concept"])

    # ----- Step 3: branch -----
    # The `if` reads the SAMPLED value of "kind". By the time this Python
    # statement runs, the interpreter has already materialized that string.
    # No round trip back to the client; the whole branch runs server-side.
    if s["kind"] == "math":
        s += "\nFinal numeric answer: "
        s += sgl.gen("answer", regex=NUMERIC_ANSWER_RE, max_tokens=16)

    elif s["kind"] == "coding":
        # Note we open the fenced code block in the literal text so the
        # regex only has to match the signature, not the entire fence.
        s += "\nFunction signature:\n```python\n"
        s += sgl.gen("answer", regex=FUNCTION_SIG_RE, max_tokens=40)
        s += "\n    ...\n```"

    else:  # "concept"
        s += "\nExplanation: "
        # Free-form gen: no regex constraint. stop="\n\n" halts on a blank
        # line. Note this is decoded-string matching; if your tokenizer
        # emits a different newline encoding, adjust accordingly.
        s += sgl.gen("answer", max_tokens=200, stop="\n\n")

    # ----- Step 4: self-confidence -----
    # Reusing the same program: the previous tokens stay in KV cache, so
    # this step only prefills the new literal text plus the new gen.
    s += "\nMy confidence (0.00-1.00): "
    s += sgl.gen("confidence", regex=CONFIDENCE_RE, max_tokens=5)
    s += sgl.assistant_end()


def main() -> None:
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

    examples = [
        "What is the derivative of x^3 at x=2?",                  # math
        "Write a Python function that returns the n-th Fibonacci.",  # coding
        "Why does dropout help generalization?",                  # concept
    ]
    states = tutor_pipeline.run_batch(
        [{"question": q} for q in examples],
        num_threads=4,
        progress_bar=True,
    )
    for q, st in zip(examples, states):
        print(f"Q: {q}")
        print(f"  kind={st['kind']}  answer={st['answer'][:80]!r}")
        print(f"  conf={st['confidence']}\n")


if __name__ == "__main__":
    main()

# ===== Nuances to absorb =====
#
# Why this pipeline beats "LLM-call + Python-parse + LLM-call":
#   Three reasons.
#   (a) RPC overhead: one program = one server interaction, not three.
#   (b) Tokenization waste: each round trip re-tokenizes the entire
#       accumulated context. Server-side execution doesn't.
#   (c) Cache locality: KV blocks stay resident. A multi-call client
#       pipeline may evict and re-prefill between calls.
#
# Comparison with vLLM:
#   vLLM gives you ONE constrained sampling per request. To do the same
#   pipeline in vLLM you'd issue three requests and stitch them in Python;
#   prefix caching helps with the prefill cost but you still pay the RPC
#   round trips and lose the cleanliness of expressing the flow in one
#   block. For complex pipelines (agents, multi-step graders, structured
#   pipelines), SGLang's program model wins.
#
# Gotcha: regex + tokenizer mismatches
#   If your regex starts with `\d`, but the tokenizer almost always emits
#   a leading " " token before the first digit, the FSM mask will reject
#   every legal continuation and you'll get an empty / repeat output.
#   Defense: either prepend a space literal before the gen, or write your
#   regex to permit a leading space: `r" ?-?\d+(\.\d+)?"`. Test with a
#   known prompt first.
#
# Determinism:
#   sgl.select is logprob-scored. Ties break by enumeration order. For
#   reproducible rubric outputs, set temperature=0 on free-form gens and
#   keep the choice list stable across runs.

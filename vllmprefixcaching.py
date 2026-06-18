"""
Problem 1 — vLLM Prefix Caching for an LLM-as-Judge Arena Loop
==============================================================

Capstone tie-in (InferTutor Arena):
    An arena judges pairwise tutor responses. Every judge call sends the SAME
    judge instructions + SAME student question, but a DIFFERENT (responseA,
    responseB) pair. If you naively call the engine N times, vLLM will
    re-prefill the entire prefix every time — that's wasted FLOPs and KV memory.

Goal:
    Measure end-to-end throughput with `enable_prefix_caching` ON vs OFF and
    feel why the speedup is *block-aligned*, not character-aligned.

Concepts you must internalize:
    1. vLLM stores the KV cache in fixed-size BLOCKS (default 16 tokens). The
       PagedAttention scheduler hashes each prefix block and reuses any block
       whose hash matches across requests. So "shared prefix" really means
       "shared prefix tokens that fill an integer number of 16-token blocks".
    2. Cache hits are byte-exact on tokens — not characters, not embeddings.
       A different chat template, a stray whitespace, or a swapped role order
       breaks the hit. Always run prompts through the *same* tokenizer +
       template pipeline you used when the cache was populated.
    3. Prefix caching has near-zero overhead, but it only pays off when the
       shared prefix is LONG vs the generated suffix. For 1-token verdicts on
       a 4k-token judge prompt — huge win. For 800-token chain-of-thought
       answers on a 200-token prompt — marginal.
    4. Prefix caching is currently incompatible (or buggy) with some features
       depending on vLLM version: prompt_logprobs, certain LoRA setups,
       multi-modal inputs. Check the compatibility matrix in your version.

Run:
    # GPU required. Pick a model that fits your VRAM.
    pip install "vllm>=0.6.3"
    python p1_vllm_prefix_cache.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from vllm import LLM, SamplingParams


# A deliberately long judge system prompt — in real arenas these are
# ~2-4k tokens of rubric, examples, and tie-breaking rules. We pad here so
# the prefix occupies many KV blocks; otherwise prefix caching has nothing
# meaningful to reuse.
JUDGE_SYSTEM = (
    "You are an impartial expert judge for an educational tutoring arena. "
    "Read the student question and TWO tutor responses, then answer with "
    "EXACTLY one token: 'A', 'B', or 'tie'. Do not explain. "
    "Tie-breakers (in order): factual correctness > pedagogical clarity > "
    "conciseness > friendliness. "
) * 30  # ~3-4k tokens depending on tokenizer — enough to dwarf the per-pair tail.

STUDENT_Q = "Walk me through backprop through a scaled dot-product attention block."

# 64 synthetic pairs. The (rA, rB) tails differ; everything before is identical.
PAIRS = [
    (f"Tutor A v{i}: gradient of softmax(QK^T/sqrt(d)) gives ...",
     f"Tutor B v{i}: by chain rule we compute dL/dV first, then ...")
    for i in range(64)
]


def build_prompt(rA: str, rB: str) -> str:
    """Construct prompt with the SHARED prefix FIRST.

    Critical: vLLM hashes prefix blocks from the LEFT. If your "shared"
    content lives in the middle of the prompt, you get zero cache hits.
    A common mistake is templating like f"{user}\n{system}\n..." — that
    puts the variable user content first and defeats caching entirely.
    """
    return (
        f"<|system|>\n{JUDGE_SYSTEM}\n"
        f"<|user|>\nStudent question: {STUDENT_Q}\n\n"
        f"[A]: {rA}\n[B]: {rB}\n\n"
        f"Your verdict: "
    )


@dataclass
class RunResult:
    label: str
    wall_seconds: float
    total_output_tokens: int

    @property
    def throughput(self) -> float:
        return self.total_output_tokens / self.wall_seconds


def benchmark(llm: LLM, prompts: list[str], label: str) -> RunResult:
    # Warmup: the first request pays JIT, CUDA-graph capture, and (for the
    # cached run) the cost of *populating* the cache. Never include warmup
    # in your timing — you'll measure setup, not steady state.
    llm.generate(prompts[:2], SamplingParams(max_tokens=1, temperature=0.0))

    t0 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))
    wall = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return RunResult(label, wall, total_tokens)

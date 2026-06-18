rmuthure@rmuthure-mn8374 DailyTensor % cat p7_vllm_speculative_decoding.py 
"""
Problem 7 — vLLM speculative decoding for low-latency tutoring.
A small DRAFT model proposes K tokens; the TARGET model verifies
them in ONE forward. Accepted tokens commit; the first rejected
token is re-sampled from the target's distribution. Big win at
batch=1, marginal-or-worse at high concurrency.
"""
import time
from vllm import LLM, SamplingParams

PROMPTS = [
    "Explain backpropagation through softmax in 3 short paragraphs.",
    "Walk a student through Bayes' rule with one worked example.",
    "Why does layer normalization stabilize transformer training?",
]
# Greedy decode so output is comparable across runs. Speculative
# decoding is mathematically exact w.r.t. the target's distribution
# (rejection sampling), so at temperature=0 outputs should match
# non-speculative greedy modulo FP noise. If they don't, file a bug.
SP = SamplingParams(max_tokens=400, temperature=0.0)

def bench(label, **kw):
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.85,
              max_model_len=4096,
              **kw)
    llm.generate(PROMPTS[:1], SP)        # warmup (JIT, CUDA-graph capture)
    t = time.perf_counter()
    outs = llm.generate(PROMPTS, SP)
    wall = time.perf_counter() - t
    tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"{label:18s} wall={wall:.2f}s   tok/s={tok/wall:.1f}")
    del llm                              # release VRAM before next engine

bench("baseline")

# Draft-model speculation. The draft MUST share tokenizer + vocab with
# the target. Llama-3.2-1B / Llama-3.1-8B share a tokenizer. Cross-family
# pairs (Llama draft, Qwen target) silently corrupt — vLLM may not even
# error; you'll just get garbage. Always sanity-check the first output.
bench(
    "spec=draft-1B",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    # K = number of tokens drafted per target forward. Larger K = more
    # potential speedup IF the draft is on-distribution, but also more
    # wasted draft compute when rejection happens early. Acceptance
    # decays roughly geometrically with position in the K-tuple. K=5
    # is a common sweet spot; measure for your workload.
    num_speculative_tokens=5,
)

# n-gram speculation: no draft model. The engine scans prompt + so-far
# generation for a matching n-gram and proposes the continuation it saw.
# Free (no extra weights), surprisingly effective when outputs quote /
# repeat from the prompt — code completion, RAG with citations,
# structured rewrites, JSON-shaped outputs that echo schema keys.
bench(
    "spec=ngram",
    speculative_model="[ngram]",      # special string sentinel
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,        # longest n-gram to look up
    ngram_prompt_lookup_min=2,        # shortest (1 = aggressive, noisy)
)

# Why it works at batch=1:
#   At low batch the target's forward is MEMORY-BANDWIDTH bound, not FLOP
#   bound — loading the weights costs ~the same whether you score 1 token
#   or K. If M of K are accepted, you got M tokens per target forward
#   instead of 1. Net speedup ≈ (M+1) minus the draft model's own cost
#   amortized over K (so a 1B draft against an 8B target eats ~12% of
#   the budget per K).
#
# Why it can break even or REGRESS:
#   - High concurrent batch (Problem 5 territory): target forward is now
#     FLOP-bound; spec forwards steal capacity from other sequences.
#   - Low acceptance rate: out-of-domain draft, very high temperature,
#     long structured outputs the small draft can't anticipate.
#   - Incompatible with guided decoding (Problem 3) in many vLLM
#     versions — the draft doesn't see the FSM mask, so verification
#     rejects systematically. Don't stack them blindly.
#
# Tuning loop:
#   Log acceptance rate (vLLM prints `acceptance_rate` in spec_decode
#   metrics). Sweep K ∈ {3, 5, 7}. If acceptance < ~50% your draft is
#   too far from the target; try a closer-family draft or switch to
#   n-gram. If acceptance > 80% try larger K.
#
# Arena tie-in:
#   Single-user tutor chats (one student, one tutor) are batch≈1
#   workloads — speculative decoding is a real ~1.5-2x latency win.
#   Multi-tutor arena backends at high QPS shift to batch>>1 and the
#   gains shrink; measure both regimes before committing.

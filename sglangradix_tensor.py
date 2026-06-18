from __future__ import annotations

import time
from typing import Iterable

import sglang as sgl


# A long, fixed system + question prefix. We want it long enough that the
# radix tree dedup is a meaningful fraction of the prefill cost. ~1k tokens
# is sufficient for clear measurement.
SHARED_PREFIX = (
    "You are an expert tutor. Be concise, correct, and pedagogically helpful. "
    "Always offer one worked example before stating the general rule. "
) * 25
SHARED_QUESTION = "Explain why softmax saturates for large logits."


@sgl.function
def tutor_answer(s, jitter: str = ""):
    """Program with a (mostly) shared prefix.

    `jitter` lets us BREAK prefix sharing on purpose: by varying it across
    calls we ensure the prefixes diverge in the very first user-content
    token, defeating dedup. With jitter="" all calls share the full prefix.
    """
    s += sgl.system(SHARED_PREFIX)
    # The jitter is appended INSIDE the user turn so it's part of the
    # token stream. If jitter is empty, the user turn is identical across
    # calls and the radix tree dedups everything up to the gen() boundary.
    s += sgl.user(f"{jitter}{SHARED_QUESTION}")
    s += sgl.assistant(sgl.gen("answer", max_tokens=128, temperature=0.7))


def run_workload(jitters: Iterable[str], label: str) -> dict:
    """Run a batch and return timing + cache stats."""
    # Reset the engine cache between experiments so prior workload doesn't
    # bias the measurement. flush_cache() clears the radix tree.
    sgl.flush_cache()

    inputs = [{"jitter": j} for j in jitters]
    t0 = time.perf_counter()
    states = tutor_answer.run_batch(inputs, num_threads=8, progress_bar=False)
    wall = time.perf_counter() - t0

    # Pull runtime stats. The exact attribute names depend on your SGLang
    # version — the snippet below tries a few common locations. If your
    # version differs, run `print(dir(backend))` to discover the right one.
    backend = sgl.get_default_backend()
    stats = {}
    for attr in ("get_server_info", "get_engine_info", "get_runtime_info"):
        if hasattr(backend, attr):
            try:
                stats = getattr(backend, attr)() or {}
                break
            except Exception as e:
                stats = {"_error": str(e)}
                break

    return {
        "label": label,
        "wall": wall,
        "n_requests": len(inputs),
        "preview": states[0]["answer"][:60] if states else "",
        "cache_stats": stats,
    }


def main() -> None:
    # Start an in-process engine and make it the default backend so our
    # @sgl.function calls use it. tp_size=1 for single GPU; raise it for
    # tensor parallelism across multiple GPUs on one host.
    engine = sgl.Engine(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        # Equivalent to gpu_memory_utilization. SGLang has its own knob.
        mem_fraction_static=0.85,
        # Disable prefix cache for the "broken" baseline below by
        # toggling this — but here we leave it on; we break sharing at
        # the workload level by varying jitter.
        # disable_radix_cache=False,
    )
    sgl.set_default_backend(engine)

    # ----- Experiment A: FULL prefix sharing -----
    # All 32 requests have the same prefix; the radix tree should
    # report a high cache hit rate. Wall time per request after the first
    # one should be dominated by decode, not prefill.
    a = run_workload(jitters=[""] * 32, label="SHARED")

    # ----- Experiment B: NO prefix sharing -----
    # Each request has a different leading byte sequence, so the user
    # turns diverge at token 0 of the user content. Only the system
    # prompt is shared. Expect higher wall time and lower hit rate.
    b = run_workload(
        jitters=[f"[req {i}] " for i in range(32)],
        label="UNIQUE",
    )

    for r in (a, b):
        print(f"\n=== {r['label']} ({r['n_requests']} requests) ===")
        print(f"wall: {r['wall']:.2f}s")
        print(f"per-req: {r['wall']/r['n_requests']*1000:.1f} ms")
        print(f"preview: {r['preview']!r}")
        print(f"stats: {r['cache_stats']}")

    # Expected outcome:
    #   SHARED.wall is much smaller than UNIQUE.wall. The delta is roughly
    #   (N - 1) * prefill_cost_for_shared_prefix. Confirm with the cache
    #   stats: SHARED should report many cached tokens, UNIQUE far fewer.

    # ----- Bonus: control comparison vs vLLM -----
    # Take the same workload, run it on a vLLM engine with
    # enable_prefix_caching=True and =False. You should see that vLLM with
    # caching ON matches SGLang's SHARED case closely. SGLang's edge over
    # vLLM is NOT prefix dedup per se — vLLM has that too — but rather:
    #   (a) program-level expressiveness (fork/select/branching) and
    #   (b) cross-request sharing that's automatic at program structure
    #       level, not at byte-level hashing time.


if __name__ == "__main__":
    main()

# ===== Nuances to absorb =====
#
# Don't conflate "prefix caching" with "RadixAttention":
#   - Prefix caching (vLLM-style): hash each new prompt's prefix blocks,
#     look up in a hash table, reuse on match.
#   - RadixAttention (SGLang): maintain an explicit radix tree of all live
#     KV sequences. Any two sequences that share a path in the tree share
#     KV blocks for that path.
#   Both achieve the same end (dedup shared prefix tokens). The radix-tree
#   model gives SGLang first-class support for program-level branching
#   (fork()) because branches are literally tree children.
#
# When SGLang's radix wins:
#   - Many concurrent agentic / tool-use programs that branch from a
#     shared prefix.
#   - Best-of-N or self-consistency sampling at high concurrency.
#   - Pipelines that interleave gen + branching + gen.
#
# When vLLM is more practical:
#   - Simple single-prompt → single-response serving where the request
#     mix doesn't have strong prefix overlap.
#   - You need bleeding-edge feature coverage (latest models, exotic
#     attention variants) — vLLM tends to land them first.
#   - Operational simplicity: vLLM's OpenAI-compatible server has wider
#     client/library support.
#
# Reproducibility:
#   `sgl.flush_cache()` is essential between A/B experiments. If you skip
#   it, experiment B benefits from experiment A's lingering KV state and
#   your numbers are gibberish.

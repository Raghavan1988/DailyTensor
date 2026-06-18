"""
Problem 2 — SGLang fork() for Best-of-N Tutor Response Selection
================================================================

Capstone tie-in (InferTutor Arena):
    For a given student question, generate N candidate tutor responses with
    diverse sampling, then have the model self-judge which is best. This is
    the "best-of-N" / "self-consistency" pattern used inside arena pipelines
    to harvest high-quality data and to score candidate models.

Goal:
    Use SGLang's @sgl.function + fork(N) to express the whole pipeline as
    one server-side program, leveraging RadixAttention so the shared prefix
    is prefilled exactly once for all N branches.

Concepts you must internalize:
    1. @sgl.function turns a Python function into an SGLang *program*. The
       function body doesn't run a tensor op — it builds a token-level
       program that the SGLang interpreter executes on the runtime.
    2. RadixAttention stores KV blocks in a radix tree keyed by token
       sequence. Identical prefixes (across forks AND across requests) are
       deduplicated automatically. You don't configure this.
    3. s.fork(n) returns n child states that all share the parent's prefix.
       The runtime prefills the shared prefix ONCE, then runs n parallel
       continuations. Compare with vLLM, where you'd typically issue n
       separate prompts and rely on prefix caching to dedup them.
    4. sgl.select(name, choices=[...]) does NOT spawn a classifier head. It
       constrains decoding to one of the provided strings via per-step
       logit masking, then exposes the chosen string in state[name].
    5. The Python `for` loop over forks runs in the SGLang interpreter; the
       underlying gen() calls are batched together on the GPU. You're
       writing what looks like serial code but you get batched execution.

Run:
    # Start the SGLang server in another terminal:
    # python -m sglang.launch_server \
    #     --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
    pip install "sglang[all]"
    python p2_sglang_fork_best_of_n.py
"""
from __future__ import annotations

import sglang as sgl


@sgl.function
def best_of_n_tutor(s, question: str, n: int = 4):
    """SGLang program: generate N candidates, then pick the best.

    Note `s` is the program state — a token buffer plus a dict of
    materialized variables. Operations like `s +=` append to the buffer;
    sgl.gen() / sgl.select() append sampled tokens AND record them in
    the state dict under the given name.
    """
    # Step 1: build the shared prefix. Everything appended before fork() will
    # have its KV cache computed ONCE for all branches. Putting the long
    # system prompt + question here is the whole point.
    s += sgl.system(
        "You are a patient math tutor. Explain step-by-step. "
        "Prefer worked examples over abstract definitions."
    )
    s += sgl.user(question)

    # Step 2: fork into N branches. Each branch is an independent program
    # state that inherits the prefix. RadixAttention guarantees the prefix
    # KV blocks are physically shared in GPU memory — no duplication.
    forks = s.fork(n)

    # Step 3: in each branch, sample a candidate response with high temperature.
    # The Python `for` loop is sequential in your client thread, but SGLang
    # batches the underlying decode steps on the server. Internally this is
    # equivalent to issuing n parallel `gen` calls that share a prefix.
    for i, f in enumerate(forks):
        f += sgl.assistant(
            sgl.gen(
                "response",
                max_tokens=256,
                # High temperature = diversity. Best-of-N only works if the
                # samples are actually different; greedy decoding gives N
                # identical outputs.
                temperature=0.9,
                # top_p reins in long-tail garbage tokens without killing
                # diversity. 0.95 is a common arena setting.
                top_p=0.95,
            )
        )

    # Step 4: join the forks back into the parent state and ask the parent
    # to pick a winner. Note we read each fork's "response" via f["response"];
    # this is the materialized text the fork's gen() produced.
    s += sgl.user(
        "Here are candidate explanations from 4 tutors. "
        "Reply with ONLY the number of the best one.\n\n"
        + "\n\n".join(f"[{i+1}] {f['response']}" for i, f in enumerate(forks))
    )

    # sgl.select constrains the next sample to one of the listed choices.
    # Mechanism: SGLang scores each candidate string by its joint log-prob
    # under the model and picks the highest. This is NOT the same as
    # "sample a token and check if it matches" — it's exact-mode selection.
    s += sgl.assistant(sgl.select("winner", choices=["1", "2", "3", "4"]))


def main() -> None:
    # Two backend options:
    #  (a) RuntimeEndpoint("http://localhost:30000")
    #       — the SGLang HTTP server. Best for multi-process / multi-host.
    #  (b) sgl.Engine(model_path=...)
    #       — in-process Python engine. Lower latency, no HTTP overhead, but
    #         pins the model into your process. Use this in notebooks / scripts.
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

    # Single run.
    state = best_of_n_tutor.run(
        question="Why does softmax saturate gradients for large inputs?"
    )
    winner_idx = int(state["winner"]) - 1
    print(f"Winner: candidate #{winner_idx + 1}")
    # state.get_var picks a variable from a specific fork.
    print(f"Text:\n{state.get_var(f'response[{winner_idx}]', default=None)}")

    # Batched run — this is where SGLang's cross-request prefix sharing shines.
    questions = [
        "Why is attention O(n^2)?",
        "Why is attention O(n^2)?",  # exact duplicate — full prefix hit
        "Why does ReLU avoid vanishing gradients?",  # different question
    ]
    # The radix tree will dedupe:
    #   - The system prompt across ALL three (longest shared prefix).
    #   - The user question across the first two (full prefix until the fork).
    # That sharing applies BOTH to the per-request prefill AND to the
    # shared prefix among the four forks of each request.
    states = best_of_n_tutor.run_batch(
        [{"question": q} for q in questions],
        # num_threads controls client-side concurrency. The server is
        # already batching; this controls how many programs you submit
        # at once so the server has work to batch.
        num_threads=8,
        progress_bar=True,
    )
    for q, st in zip(questions, states):
        print(f"Q: {q}\n  winner={st['winner']}")


if __name__ == "__main__":
    main()

# ===== Nuances to absorb =====
#
# Why fork() beats "issue N separate vLLM calls with prefix caching":
#   In vLLM, you submit N prompts, the scheduler hashes their prefixes, and
#   reuses KV blocks where it can. That works — but you've paid the cost of
#   N tokenization passes, N scheduler entries, and N response handling
#   passes. With SGLang fork(), the runtime knows from program structure
#   that N branches share a prefix; there's no hashing dance, just direct
#   shared pointers in the radix tree.
#
# When fork() is the WRONG tool:
#   If your "branches" have substantially different prefixes (e.g. different
#   system prompts), fork() doesn't help — the shared region is empty.
#   You'd just use separate runs.
#
# Determinism warning:
#   sgl.select() uses logprob scoring under FlashAttention. Token-level
#   batching can produce tiny non-determinism in tied cases. If you need
#   bit-exact repro, set temperature=0 on gen() and use single-request mode.


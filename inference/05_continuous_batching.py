"""
IDEA 5: Continuous (in-flight) batching for a serving system
============================================================
A GPU is most efficient when it processes many requests at once (a batch).
Naive "static" batching waits for EVERY request in the batch to finish before
starting new ones, so short requests are stuck waiting for the longest one.

"Continuous batching" (used by vLLM/TGI) instead runs one decode STEP across all
active requests, and as soon as a request emits its end token it leaves the
batch and a queued request immediately takes its slot -> much higher throughput.

We simulate both schedulers (no real model needed) and compare how many step-
slots are wasted. Run:  python 05_continuous_batching.py
"""

from collections import deque


def static_batching(requests, batch_size):
    """Process requests in fixed groups; a group ends only when its LONGEST
    request ends. Slots of finished-but-waiting requests are wasted."""
    steps = wasted = 0
    for start in range(0, len(requests), batch_size):
        group = requests[start : start + batch_size]
        group_len = max(group)                  # batch runs until the longest one
        steps += group_len
        for r in group:
            wasted += group_len - r             # idle slots after r finished
    return steps, wasted


def continuous_batching(requests, batch_size):
    """Keep `batch_size` slots always full: when a request finishes, pull the
    next queued one into its slot on the very next step."""
    queue = deque(requests)
    active = [queue.popleft() for _ in range(min(batch_size, len(queue)))]
    steps = wasted = 0
    while active:
        steps += 1                              # run one decode step for all slots
        for i in range(len(active)):
            active[i] -= 1                      # each request emits one token
        finished = [i for i, r in enumerate(active) if r == 0]
        # Refill finished slots immediately from the queue (the key difference).
        for i in finished:
            if queue:
                active[i] = queue.popleft()
            else:
                active[i] = None
        active = [r for r in active if r is not None]
    return steps, wasted


if __name__ == "__main__":
    # Each number = how many tokens that request needs to generate.
    requests = [2, 9, 3, 1, 8, 2, 5, 1]
    bs = 4
    s_steps, s_wasted = static_batching(requests, bs)
    c_steps, _ = continuous_batching(requests, bs)
    print(f"requests (token lengths): {requests}, batch size {bs}")
    print(f"static batching     : {s_steps} steps, {s_wasted} wasted slot-steps")
    print(f"continuous batching : {c_steps} steps (no waiting on the longest)")
    print(f"speedup: {s_steps / c_steps:.2f}x fewer steps")

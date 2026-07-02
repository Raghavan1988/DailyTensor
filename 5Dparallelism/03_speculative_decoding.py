"""
IDEA 3: Speculative decoding (make generation faster)
=====================================================
Big models are slow per token. Idea: a small/cheap "draft" model guesses the
next few tokens quickly, then the big/accurate "target" model checks them all
in ONE batched pass. Tokens the target agrees with are accepted for free; at the
first disagreement we resample from the target and stop. This produces EXACTLY
the same distribution as the target model alone, but often several tokens per
expensive target step.

We simulate this with two mock models (different seed_offset => different
behaviour). Run:  python 03_speculative_decoding.py
"""

import numpy as np
from mock_model import get_logits, softmax


def draft_model(seq):   # cheap, slightly different distribution
    return softmax(get_logits(seq, seed_offset=7))


def target_model(seq):  # the accurate model we actually want to sample from
    return softmax(get_logits(seq, seed_offset=0))


def speculative_step(sequence, k=4):
    """Propose k tokens with the draft, then verify with the target.
    Returns the list of accepted tokens (length 1..k+1)."""
    proposed, draft_probs = [], []
    seq = list(sequence)
    for _ in range(k):                       # 1) draft proposes k tokens cheaply
        p = draft_model(seq)
        tok = int(np.random.choice(len(p), p=p))
        proposed.append(tok)
        draft_probs.append(p[tok])
        seq.append(tok)

    accepted = []
    seq = list(sequence)
    for i, tok in enumerate(proposed):       # 2) target verifies each proposal
        t = target_model(seq)
        # Accept with probability min(1, target_prob / draft_prob).
        ratio = min(1.0, t[tok] / draft_probs[i])
        if np.random.random() < ratio:
            accepted.append(tok)             # accepted -> token is "free"
            seq.append(tok)
        else:
            # Rejected: resample one token from the target and stop this round.
            accepted.append(int(np.random.choice(len(t), p=t)))
            return accepted
    # All k accepted: take one bonus token straight from the target.
    accepted.append(int(np.random.choice(len(target_model(seq)), p=target_model(seq))))
    return accepted


if __name__ == "__main__":
    np.random.seed(0)
    sequence, target_calls = [0], 0
    while len(sequence) < 20:
        new = speculative_step(sequence, k=4)
        sequence.extend(new)
        target_calls += 1                    # one expensive verify round
    generated = sequence[1:]
    print("generated tokens :", generated)
    print(f"tokens produced  : {len(generated)}")
    print(f"target verify rounds (expensive calls): {target_calls}")
    print(f"avg tokens per expensive call: {len(generated)/target_calls:.2f}")

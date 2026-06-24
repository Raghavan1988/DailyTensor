"""
IDEA 4: Beam search (search for a high-probability whole sequence)
==================================================================
Greedy decoding picks the best token at each step, but a locally-best token can
lead to a globally-worse sentence. Beam search keeps the `beam_width` most
promising partial sequences ("beams") at every step and expands all of them,
trading compute for better overall sequence probability.

We score sequences by the SUM of log-probabilities (log avoids underflow from
multiplying many small numbers). Run:  python 04_beam_search.py
"""

import numpy as np
from mock_model import get_logits, softmax


def beam_search(n_tokens=8, beam_width=3):
    # Each beam is (sequence, cumulative_log_prob). Start with one empty-ish beam.
    beams = [([0], 0.0)]
    for _ in range(n_tokens):
        candidates = []
        for seq, score in beams:
            probs = softmax(get_logits(seq))
            logprobs = np.log(probs + 1e-12)        # log-prob of every next token
            for token in range(len(probs)):
                # Expand this beam by one token; accumulate log-prob.
                candidates.append((seq + [token], score + logprobs[token]))
        # Keep only the beam_width best candidates by total log-prob.
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    return beams


def greedy(n_tokens=8):
    """For comparison: the single greedy path and its log-prob."""
    seq, score = [0], 0.0
    for _ in range(n_tokens):
        probs = softmax(get_logits(seq))
        token = int(np.argmax(probs))
        score += np.log(probs[token] + 1e-12)
        seq.append(token)
    return seq, score


if __name__ == "__main__":
    g_seq, g_score = greedy()
    print(f"greedy : {g_seq[1:]}  logprob={g_score:.3f}")
    print("beam search results (best first):")
    for seq, score in beam_search(beam_width=3):
        print(f"  {seq[1:]}  logprob={score:.3f}")
    # The top beam's log-prob should be >= greedy's: search found a better path.

"""
A tiny "mock language model" shared by all the demos.

Real LLMs map a sequence of token IDs -> a probability distribution over the
next token. They are huge neural networks. To keep these demos under 100 lines
and dependency-free, we fake that behaviour with a small, *deterministic*
function. The point of these files is to teach INFERENCE concepts (how we turn
a model's next-token scores into text), NOT how the model itself is trained.

The mock model:
  - has a vocabulary of `vocab_size` tokens (just integers 0..vocab_size-1)
  - given the current sequence, returns "logits": one raw score per token
  - logits depend on the last token, so the model has a (fake) sense of context
"""

import numpy as np

VOCAB_SIZE = 16  # small vocabulary so output is easy to read


def softmax(logits, temperature=1.0):
    """Convert raw logits into a probability distribution.

    temperature > 1 flattens the distribution (more random),
    temperature < 1 sharpens it (more confident/greedy-like).
    """
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits = logits - logits.max()          # subtract max for numerical stability
    exp = np.exp(logits)
    return exp / exp.sum()


def get_logits(sequence, vocab_size=VOCAB_SIZE, seed_offset=0):
    """Return fake next-token logits given the sequence so far.

    We seed a random generator with the last token id so that:
      - the same context always produces the same logits (deterministic), and
      - different contexts produce different distributions (context matters).
    `seed_offset` lets a "draft" model behave slightly differently from a
    "target" model (used in the speculative decoding demo).
    """
    last_token = sequence[-1] if len(sequence) else 0
    rng = np.random.default_rng(last_token + seed_offset)
    logits = rng.normal(size=vocab_size)
    # Make the model mildly prefer the "next" token id, so sequences tend to
    # count upward -- this just makes the demo output look less random.
    logits[(last_token + 1) % vocab_size] += 2.0
    return logits

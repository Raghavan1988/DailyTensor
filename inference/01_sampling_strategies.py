"""
IDEA 1: Decoding / sampling strategies
======================================
Given the next-token probabilities, HOW do we pick the next token?
This is the single most important inference-time choice. We implement four
classic strategies and show how each changes the generated text.

Run:  python 01_sampling_strategies.py
"""

import numpy as np
from mock_model import get_logits, softmax, VOCAB_SIZE


def greedy(probs):
    """Always take the single most likely token. Deterministic, can be dull."""
    return int(np.argmax(probs))


def temperature_sample(probs):
    """Sample randomly in proportion to probability (temperature already
    applied when computing probs). Adds diversity/creativity."""
    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(probs, k=3):
    """Keep only the k highest-probability tokens, renormalize, then sample.
    Prevents picking from the long tail of unlikely (often nonsense) tokens."""
    top = np.argsort(probs)[-k:]            # indices of the k best tokens
    masked = np.zeros_like(probs)
    masked[top] = probs[top]               # zero out everything else
    masked /= masked.sum()                 # renormalize to a valid distribution
    return int(np.random.choice(len(probs), p=masked))


def top_p_sample(probs, p=0.9):
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability >= p, then sample from them. Adapts how many tokens to consider
    based on how confident the model is."""
    order = np.argsort(probs)[::-1]        # tokens from most to least likely
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, p) + 1   # how many to keep
    keep = order[:cutoff]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked /= masked.sum()
    return int(np.random.choice(len(probs), p=masked))


def generate(strategy, n_tokens=12, temperature=1.0):
    """Autoregressive loop: feed sequence -> get logits -> pick token -> append."""
    sequence = [0]                          # start-of-sequence token
    for _ in range(n_tokens):
        probs = softmax(get_logits(sequence), temperature=temperature)
        sequence.append(strategy(probs))
    return sequence[1:]                     # drop the start token


if __name__ == "__main__":
    np.random.seed(42)                      # reproducible randomness for the demo
    print("vocab size:", VOCAB_SIZE)
    print("greedy        :", generate(greedy))
    print("temperature   :", generate(temperature_sample, temperature=1.2))
    print("top-k (k=3)   :", generate(lambda p: top_k_sample(p, k=3)))
    print("top-p (p=0.9) :", generate(lambda p: top_p_sample(p, p=0.9)))

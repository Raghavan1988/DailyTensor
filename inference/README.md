# 5 Simple LLM Inference Ideas

Five small, self-contained Python demos of core LLM *inference* concepts. Each
demo is under 100 lines (excluding comments), depends only on `numpy`, and runs
in under a second using a tiny **mock model** (`mock_model.py`) — no real LLM or
GPU required. The goal is to understand the *idea*, not to train a model.

| File | Idea | What it shows |
|------|------|---------------|
| `01_sampling_strategies.py` | Decoding strategies | greedy vs temperature vs top-k vs top-p, and how each changes output |
| `02_kv_cache.py` | KV cache | caching keys/values turns O(N²) attention work into O(N), same result |
| `03_speculative_decoding.py` | Speculative decoding | a cheap draft model proposes tokens, big model verifies a batch → more tokens per expensive call |
| `04_beam_search.py` | Beam search | keeping the top-k partial sequences finds a higher-probability sentence than greedy |
| `05_continuous_batching.py` | Continuous batching | refilling finished slots immediately beats static batching for serving throughput |

## Run

```bash
pip install numpy
cd llm_inference_ideas
python 01_sampling_strategies.py   # or any of the five files
```

Start with `mock_model.py` — it explains the fake "model" the others share.
Suggested reading order is 01 → 05.

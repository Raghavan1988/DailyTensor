from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class Rubric(BaseModel):
    """Pydantic schema for the rubric. vLLM accepts either:
       - a Pydantic class directly,
       - a JSON Schema dict,
       - a regex (for simpler outputs),
       - or a context-free grammar (advanced).

    Field constraints (ge, le, max_length, Literal[...]) translate to
    JSON Schema constraints which the FSM enforces. Be aware that the
    FSM enforces them at the GRAMMAR level — integer fields are restricted
    to digits 0-9 and a minus sign, but the ge/le bounds may or may not
    be enforced depending on the backend version. Always re-validate the
    output with Pydantic to catch any drift.
    """
    clarity:      int = Field(ge=1, le=5)
    correctness:  int = Field(ge=1, le=5)
    depth:        int = Field(ge=1, le=5)
    friendliness: int = Field(ge=1, le=5)
    overall_grade: Literal["A", "B", "C", "D", "F"]
    rationale: str = Field(min_length=10, max_length=200)


# Build the guided-decoding spec ONCE and pass it as a reused object.
# Recreating GuidedDecodingParams per request makes vLLM re-look-up the
# cached FSM by schema hash — cheap in absolute terms, but at high QPS it
# becomes a measurable contention point on the schema cache lock.
RUBRIC_GUIDE = GuidedDecodingParams(json=Rubric.model_json_schema())


def grade_tutor_responses(model: str, items: list[tuple[str, str]]) -> list[Rubric]:
    """items is a list of (question, response) pairs."""
    llm = LLM(
        model=model,
        # xgrammar is significantly faster than outlines for JSON, because
        # the FSM mask runs in a CUDA kernel fused with the sampler. Outlines
        # uses CPU+numpy for the mask, which adds ~1-3 ms per step. At
        # 1000 tok/s that ratchets your throughput down by ~30%.
        guided_decoding_backend="xgrammar",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    prompts = [
        # The model needs hints that the output is JSON. Even with guided
        # decoding, you'll get better results if you tell the model the
        # SHAPE you want; the FSM corrects errors but the model produces
        # better content when not fighting the mask.
        (
            "You are grading a tutor response on a rubric. Return JSON only "
            "with keys clarity, correctness, depth, friendliness (each 1-5), "
            "overall_grade (A-F), and rationale.\n\n"
            f"Question: {q}\nResponse: {r}\n\nRubric JSON: "
        )
        for q, r in items
    ]

    sp = SamplingParams(
        # Low but non-zero temperature: gives variation across many
        # scorings (so you can ensemble), without veering into nonsense.
        temperature=0.2,
        # max_tokens must comfortably fit your maximum JSON serialization.
        # If you cut off mid-string the FSM will happily emit a closing
        # brace at the boundary but you'll have lost rationale content.
        # Budget generously; the FSM will stop early once the JSON is done.
        max_tokens=300,
        guided_decoding=RUBRIC_GUIDE,
    )

    outputs = llm.generate(prompts, sp)

    # Note we still validate with Pydantic. Two reasons:
    #  - Defense in depth against backend bugs / unusual schemas.
    #  - Pydantic enforces ge/le bounds that grammar-level FSMs may not.
    return [Rubric.model_validate_json(o.outputs[0].text) for o in outputs]


def main() -> None:
    items = [
        ("Why is attention O(n^2)?",
         "Because each query attends to every key — n queries × n keys = n^2."),
        ("Explain layer norm.",
         "It normalizes across the feature dimension, with learnable affine."),
        ("What is dropout?",
         "Randomly zero some activations during training to prevent co-adaptation."),
    ]
    rubrics = grade_tutor_responses("Qwen/Qwen2.5-7B-Instruct", items)
    for (q, _), r in zip(items, rubrics):
        print(f"Q: {q}\n  {r.model_dump()}\n")


if __name__ == "__main__":
    main()

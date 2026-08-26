"""
Day 1, Step 3 — Advanced Vanilla SFT in RAW PyTorch.

Still NO Trainer, Accelerate, DeepSpeed, PEFT, etc.

You explicitly own:
    1. Dataset loading
    2. Train / validation split
    3. Chat formatting
    4. Prompt masking
    5. Truncation
    6. Dynamic padding
    7. DataLoader
    8. Deterministic shuffling
    9. Forward pass
   10. Completion-only loss
   11. Token-weighted gradient accumulation
   12. Gradient clipping
   13. Optimizer parameter groups
   14. LR scheduling
   15. Validation
   16. Checkpointing
   17. Resume-from-checkpoint
   18. Final model saving

Smoke:
    python 03_sft_raw_advanced.py --smoke

Train:
    python 03_sft_raw_advanced.py

Resume:
    python 03_sft_raw_advanced.py --resume ./checkpoints/step-50
"""

import os

# Must be configured before importing torch.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import math
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from common import BASE_MODEL, MAX_LEN, load_chat_dataset


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run a tiny end-to-end test.",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume from a checkpoint directory.",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=1,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=2,
)

parser.add_argument(
    "--grad-accum",
    type=int,
    default=8,
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
)

parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.1,
)

parser.add_argument(
    "--eval-every",
    type=int,
    default=25,
)

parser.add_argument(
    "--save-every",
    type=int,
    default=50,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

args = parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"

if not torch.cuda.is_available():
    raise RuntimeError("This script currently expects a CUDA GPU.")

EPOCHS = args.epochs

BATCH = args.batch_size
GRAD_ACCUM = args.grad_accum

LR = args.lr
WEIGHT_DECAY = args.weight_decay

SEED = args.seed

N_TRAIN = 64 if args.smoke else 2000

OUT_DIR = Path("./out-raw-advanced")
CHECKPOINT_DIR = Path("./checkpoints")

OUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    """
    Seed the major random-number generators.

    This does not guarantee bit-for-bit identical GPU training,
    but it removes many common sources of randomness.
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =============================================================================
# Tokenizer
# =============================================================================

tokenizer_source = args.resume if args.resume else BASE_MODEL

tok = AutoTokenizer.from_pretrained(tokenizer_source)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# =============================================================================
# Encoding
# =============================================================================

def encode(example):
    """
    Convert:

        system
        user
        assistant

    into:

        input_ids
        labels

    while masking everything before the assistant completion.

    Example:

        <system> You are helpful.
        <user> What is 2 + 2?
        <assistant> 4

    input_ids:

        [system tokens]
        [user tokens]
        [assistant header]
        [4]
        [EOS]

    labels:

        -100
        -100
        -100
        ...
        token("4")
        token(EOS)

    Cross entropy ignores positions whose target is -100.
    """

    messages = example["messages"]

    if len(messages) < 2:
        raise ValueError("Expected at least user + assistant messages.")

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Last message must be assistant, got {messages[-1]['role']}"
        )

    # -------------------------------------------------------------------------
    # Full conversation INCLUDING assistant answer.
    # -------------------------------------------------------------------------

    full_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # -------------------------------------------------------------------------
    # Everything BEFORE the assistant answer.
    #
    # Unlike:
    #
    #     messages[:1]
    #
    # this works even when examples contain:
    #
    #     system
    #     user
    #     assistant
    #
    # or longer conversation histories.
    # -------------------------------------------------------------------------

    prompt_messages = messages[:-1]

    prompt_text = tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # -------------------------------------------------------------------------
    # Tokenize independently.
    # -------------------------------------------------------------------------

    full_ids = tok(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    prompt_ids = tok(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    # -------------------------------------------------------------------------
    # Truncate.
    # -------------------------------------------------------------------------

    full_ids = full_ids[:MAX_LEN]

    n_prompt = min(
        len(prompt_ids),
        len(full_ids),
    )

    # -------------------------------------------------------------------------
    # Completion-only labels.
    # -------------------------------------------------------------------------

    labels = (
        [-100] * n_prompt
        + full_ids[n_prompt:]
    )

    completion_tokens = len(full_ids) - n_prompt

    return {
        "input_ids": full_ids,
        "labels": labels,
        "completion_tokens": completion_tokens,
    }


# =============================================================================
# Dataset
# =============================================================================

raw_ds = load_chat_dataset(n=N_TRAIN)

ds = raw_ds.map(
    encode,
    remove_columns=["messages"],
)


# -----------------------------------------------------------------------------
# Remove examples whose assistant answer disappeared entirely because of
# truncation.
# -----------------------------------------------------------------------------

before = len(ds)

ds = ds.filter(
    lambda x: x["completion_tokens"] > 0
)

removed = before - len(ds)

if removed:
    print(
        f"Warning: removed {removed} examples "
        "whose completion was entirely truncated."
    )


# -----------------------------------------------------------------------------
# Train / validation split.
# -----------------------------------------------------------------------------

split = ds.train_test_split(
    test_size=0.10,
    seed=SEED,
)

train_ds = split["train"]
val_ds = split["test"]


print(
    f"dataset: train={len(train_ds)} "
    f"validation={len(val_ds)}"
)


# =============================================================================
# Collation
# =============================================================================

def collate(batch):
    """
    Dynamically pad only to the longest sequence in THIS batch.

    Suppose lengths are:

        example 1 = 100
        example 2 = 150

    Instead of padding both to MAX_LEN=1024, we create:

        [2, 150]

    This saves substantial compute.
    """

    max_length = max(
        len(example["input_ids"])
        for example in batch
    )

    input_ids = []
    labels = []
    attention_masks = []

    for example in batch:

        seq_len = len(example["input_ids"])

        padding = max_length - seq_len

        input_ids.append(
            example["input_ids"]
            + [tok.pad_token_id] * padding
        )

        labels.append(
            example["labels"]
            + [-100] * padding
        )

        attention_masks.append(
            [1] * seq_len
            + [0] * padding
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
        ),
    }


# =============================================================================
# DataLoaders
# =============================================================================

def make_train_loader(epoch):
    """
    Create deterministic-but-different shuffling every epoch.

    Epoch 0:
        seed = 42

    Epoch 1:
        seed = 43

    etc.

    This also lets us recreate the exact ordering when resuming.
    """

    generator = torch.Generator()

    generator.manual_seed(
        SEED + epoch
    )

    return DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        pin_memory=True,
        num_workers=0,
    )


val_loader = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
    collate_fn=collate,
    pin_memory=True,
    num_workers=0,
)


# =============================================================================
# Model
# =============================================================================

model_source = args.resume if args.resume else BASE_MODEL

print(
    f"Loading model from: {model_source}"
)

model = AutoModelForCausalLM.from_pretrained(
    model_source,
    dtype=torch.bfloat16,
)

model.to(DEVICE)


# -----------------------------------------------------------------------------
# Gradient checkpointing
# -----------------------------------------------------------------------------

model.gradient_checkpointing_enable()


# KV caching is useful for generation but not training.
#
# Gradient checkpointing and use_cache=True also conflict in many transformer
# implementations.
model.config.use_cache = False

model.train()


# =============================================================================
# Trainable parameter statistics
# =============================================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"parameters: "
    f"{trainable_parameters / 1e6:.1f}M trainable / "
    f"{total_parameters / 1e6:.1f}M total"
)


# =============================================================================
# AdamW parameter groups
# =============================================================================
#
# Weight decay should generally NOT be applied to:
#
#     bias
#     LayerNorm weights
#
# We therefore build two groups manually.
#
# This is usually hidden from you by Trainer.
# =============================================================================

decay_params = []
no_decay_params = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    lower_name = name.lower()

    if (
        name.endswith(".bias")
        or "norm" in lower_name
    ):
        no_decay_params.append(parameter)

    else:
        decay_params.append(parameter)


optimizer_groups = [
    {
        "params": decay_params,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": no_decay_params,
        "weight_decay": 0.0,
    },
]


# -----------------------------------------------------------------------------
# Use fused AdamW when CUDA supports it.
# -----------------------------------------------------------------------------

try:

    opt = torch.optim.AdamW(
        optimizer_groups,
        lr=LR,
        fused=True,
    )

    print("Using fused AdamW")

except TypeError:

    opt = torch.optim.AdamW(
        optimizer_groups,
        lr=LR,
    )

    print("Using standard AdamW")


# =============================================================================
# Training-step calculation
# =============================================================================

micro_batches_per_epoch = math.ceil(
    len(train_ds) / BATCH
)

optimizer_steps_per_epoch = math.ceil(
    micro_batches_per_epoch / GRAD_ACCUM
)

total_optimizer_steps = (
    optimizer_steps_per_epoch
    * EPOCHS
)

warmup_steps = max(
    1,
    int(total_optimizer_steps * 0.03),
)


sched = get_cosine_schedule_with_warmup(
    opt,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps,
)


print()
print("Training configuration")
print("----------------------")
print(f"epochs             : {EPOCHS}")
print(f"micro batch        : {BATCH}")
print(f"gradient accumulation: {GRAD_ACCUM}")
print(f"effective batch    : {BATCH * GRAD_ACCUM}")
print(f"optimizer steps    : {total_optimizer_steps}")
print(f"warmup steps       : {warmup_steps}")
print(f"learning rate      : {LR:.2e}")
print()


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate():
    """
    Calculate validation loss.

    Important subtlety:

    Different batches contain different numbers of completion tokens.

    Therefore:

        average(batch losses)

    is NOT exactly equivalent to:

        average(all completion-token losses)

    We weight each batch by its number of valid target tokens.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:

        batch = {
            key: value.to(
                DEVICE,
                non_blocking=True,
            )
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            outputs = model(**batch)

        # Causal LM internally shifts:
        #
        # logits[:, :-1]
        # labels[:, 1:]
        #
        # so count target tokens after the same shift.
        valid_tokens = (
            batch["labels"][:, 1:] != -100
        ).sum().item()

        total_loss += (
            outputs.loss.item()
            * valid_tokens
        )

        total_tokens += valid_tokens

    average_loss = (
        total_loss / max(total_tokens, 1)
    )

    # Perplexity can overflow for terrible models,
    # so cap before exponentiating.
    perplexity = math.exp(
        min(average_loss, 20)
    )

    model.train()

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
    }


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    global_step,
    epoch,
    next_micro_batch,
):
    """
    Save BOTH:

        HuggingFace model
        PyTorch training state

    Saving only model weights is enough for inference,
    but not enough for true training resumption.
    """

    path = (
        CHECKPOINT_DIR
        / f"step-{global_step}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model in standard HF format.
    old_use_cache = model.config.use_cache

    model.config.use_cache = True

    model.save_pretrained(path)
    tok.save_pretrained(path)

    model.config.use_cache = old_use_cache

    # Save optimizer / scheduler / progress state.
    training_state = {
        "global_step": global_step,
        "epoch": epoch,
        "next_micro_batch": next_micro_batch,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),

        # RNG state makes resume more reproducible.
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }

    torch.save(
        training_state,
        path / "training_state.pt",
    )

    print(
        f"\ncheckpoint saved -> {path}"
    )


# =============================================================================
# Resume state
# =============================================================================

global_step = 0

start_epoch = 0
start_micro_batch = 0


if args.resume:

    state_path = (
        Path(args.resume)
        / "training_state.pt"
    )

    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing training state: {state_path}"
        )

    print(
        f"Loading training state: {state_path}"
    )

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    global_step = state["global_step"]

    start_epoch = state["epoch"]

    start_micro_batch = state[
        "next_micro_batch"
    ]

    opt.load_state_dict(
        state["optimizer"]
    )

    sched.load_state_dict(
        state["scheduler"]
    )

    torch.set_rng_state(
        state["torch_rng"]
    )

    torch.cuda.set_rng_state_all(
        state["cuda_rng"]
    )

    random.setstate(
        state["python_rng"]
    )

    print(
        f"Resuming at "
        f"epoch={start_epoch}, "
        f"micro_batch={start_micro_batch}, "
        f"global_step={global_step}"
    )


# =============================================================================
# Gradient utility
# =============================================================================

def divide_gradients(divisor):
    """
    Divide every accumulated gradient by a scalar.

    Why are we doing this ourselves?

    Standard code usually does:

        loss = loss / GRAD_ACCUM
        loss.backward()

    But that weights each MICRO-BATCH equally.

    Suppose:

        batch A contains 400 completion tokens
        batch B contains 40 completion tokens

    They should not necessarily contribute equally.

    Instead we accumulate SUMMED token loss and then normalize by the actual
    number of completion tokens in the accumulation window.

    This is closer to pretending all accumulated examples were processed as one
    giant batch.
    """

    if divisor <= 0:
        return

    scale = 1.0 / divisor

    for parameter in model.parameters():

        if parameter.grad is not None:
            parameter.grad.mul_(scale)


# =============================================================================
# Initial validation
# =============================================================================

initial_metrics = evaluate()

print(
    f"initial validation: "
    f"loss={initial_metrics['loss']:.4f} "
    f"ppl={initial_metrics['perplexity']:.2f}"
)


# =============================================================================
# Training loop
# =============================================================================

opt.zero_grad(
    set_to_none=True
)

running_loss_sum = 0.0
running_tokens = 0

accumulated_tokens = 0
micro_batches_accumulated = 0


for epoch in range(
    start_epoch,
    EPOCHS,
):

    train_loader = make_train_loader(
        epoch
    )

    print(
        f"\n===== Epoch {epoch + 1}/{EPOCHS} ====="
    )

    for micro_step, batch in enumerate(
        train_loader
    ):

        # -------------------------------------------------------------
        # Resume in the middle of an epoch.
        # -------------------------------------------------------------

        if (
            epoch == start_epoch
            and micro_step < start_micro_batch
        ):
            continue

        batch = {
            key: value.to(
                DEVICE,
                non_blocking=True,
            )
            for key, value in batch.items()
        }

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            outputs = model(**batch)

            mean_loss = outputs.loss

        # -------------------------------------------------------------
        # Number of assistant target tokens.
        # -------------------------------------------------------------

        valid_tokens = (
            batch["labels"][:, 1:] != -100
        ).sum()

        token_count = valid_tokens.item()

        if token_count == 0:
            continue

        # -------------------------------------------------------------
        # Convert mean token loss into SUM of token losses.
        #
        # HuggingFace gives:
        #
        #     loss = total_loss / N
        #
        # We reconstruct:
        #
        #     total_loss = loss * N
        #
        # so multiple micro-batches can be combined correctly.
        # -------------------------------------------------------------

        token_loss_sum = (
            mean_loss
            * valid_tokens
        )

        token_loss_sum.backward()

        accumulated_tokens += token_count

        micro_batches_accumulated += 1

        running_loss_sum += (
            mean_loss.item()
            * token_count
        )

        running_tokens += token_count

        # -------------------------------------------------------------
        # Should optimizer step now?
        # -------------------------------------------------------------

        reached_accumulation_target = (
            micro_batches_accumulated
            == GRAD_ACCUM
        )

        reached_end_of_epoch = (
            micro_step + 1
            == len(train_loader)
        )

        should_step = (
            reached_accumulation_target
            or reached_end_of_epoch
        )

        if not should_step:
            continue

        # -------------------------------------------------------------
        # Normalize accumulated gradients by ACTUAL token count.
        #
        # This also correctly handles the final partial accumulation
        # window.
        # -------------------------------------------------------------

        divide_gradients(
            accumulated_tokens
        )

        # -------------------------------------------------------------
        # Gradient clipping.
        #
        # Returns the norm BEFORE clipping.
        # -------------------------------------------------------------

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
        )

        # -------------------------------------------------------------
        # Real parameter update.
        # -------------------------------------------------------------

        opt.step()

        sched.step()

        opt.zero_grad(
            set_to_none=True
        )

        global_step += 1

        # -------------------------------------------------------------
        # Reset accumulation state.
        # -------------------------------------------------------------

        accumulated_tokens = 0
        micro_batches_accumulated = 0

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        if args.smoke:
            log_every = 1
        else:
            log_every = 10

        if global_step % log_every == 0:

            train_loss = (
                running_loss_sum
                / max(running_tokens, 1)
            )

            lr = sched.get_last_lr()[0]

            gpu_memory = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            print(
                f"step={global_step:4d}/{total_optimizer_steps} | "
                f"loss={train_loss:.4f} | "
                f"grad={float(grad_norm):.3f} | "
                f"lr={lr:.2e} | "
                f"tokens={running_tokens:,} | "
                f"gpu={gpu_memory:.2f}GB"
            )

            running_loss_sum = 0.0
            running_tokens = 0

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        if (
            not args.smoke
            and global_step % args.eval_every == 0
        ):

            metrics = evaluate()

            print(
                f"  validation | "
                f"loss={metrics['loss']:.4f} | "
                f"ppl={metrics['perplexity']:.2f} | "
                f"tokens={metrics['tokens']:,}"
            )

        # -------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------

        if (
            not args.smoke
            and global_step % args.save_every == 0
        ):

            save_checkpoint(
                global_step=global_step,
                epoch=epoch,
                next_micro_batch=micro_step + 1,
            )

    # After completing an epoch, resumption skipping should stop.
    start_micro_batch = 0


# =============================================================================
# Final validation
# =============================================================================

final_metrics = evaluate()

print()
print("Final validation")
print("----------------")
print(
    f"loss       : {final_metrics['loss']:.4f}"
)
print(
    f"perplexity : {final_metrics['perplexity']:.2f}"
)


# =============================================================================
# Save final model
# =============================================================================

if args.smoke:

    print(
        "\nSmoke test OK."
    )

else:

    # Re-enable KV cache for inference.
    model.config.use_cache = True

    model.save_pretrained(
        OUT_DIR
    )

    tok.save_pretrained(
        OUT_DIR
    )

    print(
        f"\nSaved final model -> {OUT_DIR}"
    )

    print(
        f"Evaluate with:\n"
        f"python 04_chat_eval.py --model {OUT_DIR}"
    )

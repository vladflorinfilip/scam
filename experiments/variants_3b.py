"""Variant definitions for the Qwen2.5-3B experiments."""

VARIANTS = {
    "lexical_e6": {
        "rule": "lexical",
        "training": {"epochs": 6},
        "adapter": "checkpoints/qwen3b-cot-sft-lexical-e6",
    }
}

for seed in (1, 2):
    for rule in ("s1", "voice", "clause", "lexical"):
        VARIANTS[f"{rule}_seed{seed}"] = {
            "rule": rule,
            "training": {"training_seed": seed},
            "adapter": f"checkpoints/qwen3b-cot-sft-{rule}-seed{seed}",
        }
    VARIANTS[f"lexical_e6_seed{seed}"] = {
        "rule": "lexical",
        "training": {"epochs": 6, "training_seed": seed},
        "adapter": f"checkpoints/qwen3b-cot-sft-lexical-e6-seed{seed}",
    }

for rule in ("s1", "voice", "clause", "lexical"):
    VARIANTS[f"{rule}_seed0"] = {
        "rule": rule,
        "training": {},
        "adapter": f"checkpoints/qwen3b-cot-sft-{rule}",
    }

VARIANTS["lexical_e6_seed0"] = {
    "rule": "lexical",
    "training": {},
    "adapter": "checkpoints/qwen3b-cot-sft-lexical-e6",
}

SEED0_ALIASES = tuple(name for name in VARIANTS if name.endswith("_seed0"))

# Label-flipped adapters (same data, splits, recipe; final_answer = 1 - cue label). Used to tell a shared
# cue code from the adapter's own answer code (kernel-exploration).
for rule in ("clause", "voice"):
    VARIANTS[f"{rule}_flip"] = {
        "rule": rule,
        "training": {},
        "train": f"data/inputs/training_data/kernel_flip/{rule}_flipped_train.jsonl",
        "adapter": f"checkpoints/qwen3b-cot-sft-{rule}-flip",
    }

VARIANTS["s1_flip"] = {
    "rule": "s1",
    "training": {},
    "train": "data/inputs/training_data/kernel_flip/s1_flipped_train.jsonl",
    "adapter": "checkpoints/qwen3b-cot-sft-s1-flip",
}

VARIANTS["lexical_flip"] = {
    "rule": "lexical",
    "training": {"epochs": 6},
    "train": "data/inputs/training_data/kernel_flip/lexical_flipped_train.jsonl",
    "adapter": "checkpoints/qwen3b-cot-sft-lexical-e6-flip",
}

VARIANTS["lexical_flip_e10"] = {
    "rule": "lexical",
    "training": {"epochs": 10},
    "train": "data/inputs/training_data/kernel_flip/lexical_flipped_train.jsonl",
    "adapter": "checkpoints/qwen3b-cot-sft-lexical-e10-flip",
}

"""Modal runner for the Qwen2.5-3B four-rule replication."""
from copy import deepcopy
import os
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/scam")
CONFIG_PATH = "configs/stegano_experiments_3b.yaml"
RULES = ["s1", "voice", "clause", "lexical"]
VARIANTS = {
    "lexical_e6": {
        "rule": "lexical",
        "training": {"epochs": 6},
        "adapter": "checkpoints/qwen3b-cot-sft-lexical-e6",
    }
}
GPU = "A100-80GB"
TIMEOUT = 2 * 60 * 60


def _transfer_out_dir(rules: list[str], variant: str = "") -> str:
    if variant:
        name = f"four_rule_3b_l30_{variant}"
    elif rules == RULES:
        name = "four_rule_3b_l30"
    else:
        name = f"rules_{'-'.join(rules)}_3b_l30"
    return f"/vol/outputs/transfer/{name}"


DATA_FILES = [
    "data/inputs/training_data/synthetic_ethics_cot_training_v2.jsonl",
    "data/inputs/training_data/synthetic_ethics_voice_paired_train.jsonl",
    "data/inputs/training_data/synthetic_ethics_clause_order_paired_train.jsonl",
    "data/inputs/training_data/synthetic_ethics_lexical_paired_train.jsonl",
    "data/inputs/validation_data/synthetic_ethics_cot_val_v2.jsonl",
    "data/inputs/validation_data/synthetic_ethics_voice_paired_val.jsonl",
    "data/inputs/validation_data/synthetic_ethics_clause_order_paired_eval.jsonl",
    "data/inputs/validation_data/synthetic_ethics_lexical_paired_eval.jsonl",
    "data/experiments/residuals/ablations/l18/third_rule_l18/splits.json",
    "data/experiments/baselines/clause_order/base_ethics.jsonl",
]


def build_image():
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.5.1",
            "transformers==4.49.0",
            "peft==0.14.0",
            "datasets<4",
            "sympy==1.13.1",
            "accelerate",
            "pyyaml",
            "tqdm",
            "matplotlib",
            "plotly",
            "numpy<2",
        )
        .env({"MPLBACKEND": "Agg", "PLOTLY_RENDERER": "json"})
    )
    image = image.run_commands("mkdir -p /root/scam")

    python_dirs = (
        "experiments",
        "training",
        "sparse_autoencoders",
        "intervention",
        "evaluation",
        "stencils",
    )
    for name in python_dirs:
        source = ROOT / name
        if name == "sparse_autoencoders":
            ignore = lambda path: (
                "artifacts" in path.parts or (path.is_file() and path.suffix != ".py")
            )
        else:
            ignore = lambda path: path.is_file() and path.suffix != ".py"
        image = image.add_local_dir(
            source,
            remote_path=str(REMOTE_ROOT / name),
            ignore=ignore,
        )

    image = image.add_local_dir(
        ROOT / "configs",
        remote_path=str(REMOTE_ROOT / "configs"),
        ignore=lambda path: path.is_file() and path.suffix not in {".yaml", ".yml"},
    )
    for relative in DATA_FILES:
        image = image.add_local_file(
            ROOT / relative,
            remote_path=str(REMOTE_ROOT / relative),
        )
    return image


app = modal.App("scam-qwen-3b")
image = build_image()
scam_volume = modal.Volume.from_name("scam-3b", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface")


def _map_huggingface_secret():
    names = sorted(
        key
        for key in os.environ
        if "TOKEN" in key.upper()
        or "HUGGINGFACE" in key.upper()
        or key.upper().startswith("HF_")
    )
    preferred = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    key = next((candidate for candidate in preferred if os.environ.get(candidate)), None)
    if key is None:
        key = next(
            (
                candidate
                for candidate in names
                if "HUGGINGFACE" in candidate.upper() or candidate.upper().startswith("HF_")
            ),
            None,
        )
    if key is None:
        key = next((candidate for candidate in names if "TOKEN" in candidate.upper()), None)
    if key:
        token = os.environ[key]
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    return names


def _prepare():
    _map_huggingface_secret()
    os.chdir(REMOTE_ROOT)
    if str(REMOTE_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT))
    checkpoints = REMOTE_ROOT / "checkpoints"
    volume_checkpoints = Path("/vol/checkpoints")
    volume_checkpoints.mkdir(parents=True, exist_ok=True)
    if checkpoints.is_symlink():
        if checkpoints.resolve() != volume_checkpoints.resolve():
            checkpoints.unlink()
    elif checkpoints.exists():
        if not checkpoints.is_dir() or any(checkpoints.iterdir()):
            raise RuntimeError(f"Refusing to replace non-empty checkpoints path: {checkpoints}")
        checkpoints.rmdir()
    if not checkpoints.exists():
        checkpoints.symlink_to(volume_checkpoints, target_is_directory=True)
    Path("/vol/outputs").mkdir(parents=True, exist_ok=True)

    import plotly.io as pio

    pio.renderers.default = "json"


def _config(variant: str = ""):
    from experiments.data import config

    cfg = deepcopy(config(CONFIG_PATH))
    if variant:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        spec = VARIANTS[variant]
        cfg["training"].update(spec["training"])
        cfg["rules"][spec["rule"]]["adapter"] = spec["adapter"]
        cfg["variant"] = variant
    return cfg


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=[huggingface_secret],
    volumes={
        "/vol": scam_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def train_rule(rule: str, variant: str = ""):
    _prepare()
    cfg = _config(variant)
    if variant and rule != VARIANTS[variant]["rule"]:
        raise ValueError(f"Variant {variant} must use rule {VARIANTS[variant]['rule']}")
    adapter = Path(cfg["rules"][rule]["adapter"])
    if (adapter / "adapter_model.safetensors").exists():
        print(f"Training already complete for {variant or rule}; skipping.")
    else:
        from experiments.colab import train

        train(cfg, rule)
        scam_volume.commit()

    baseline_dir = Path("/vol/outputs/baselines") / (variant or rule)
    if (baseline_dir / "experiment.json").exists():
        print(f"Baseline already complete for {variant or rule}; skipping.")
    else:
        from experiments.colab import baselines

        baselines(cfg, rule, baseline_dir)
        scam_volume.commit()
    return variant or rule


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=[huggingface_secret],
    volumes={
        "/vol": scam_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def scan_rule(
    rule: str,
    layers: list[int],
    out_dir: str,
    run_free: bool,
    variant: str = "",
):
    _prepare()
    cfg = _config(variant)
    if variant and rule != VARIANTS[variant]["rule"]:
        raise ValueError(f"Variant {variant} must use rule {VARIANTS[variant]['rule']}")
    adapter = Path(cfg["rules"][rule]["adapter"]) / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(f"Train {variant or rule} before scanning; adapter missing: {adapter}")
    out = Path(out_dir)
    required_scan_outputs = (
        out / "experiment.json",
        out / "selected_subspace.pt",
        out / "layer_scan.json",
        out / "summary.json",
    )
    if (out / "experiment.json").exists():
        if not all(path.exists() for path in required_scan_outputs):
            raise RuntimeError(f"Scan output is incomplete and cannot be safely resumed: {out}")
        print(f"Scan already complete for {variant or rule}; skipping.")
        import torch

        selected = torch.load(
            out / "selected_subspace.pt",
            map_location="cpu",
            weights_only=False,
        )
    else:
        from experiments.residual import scan

        selected = scan(cfg, rule, layers, out, rank=1, method="mean")
        scam_volume.commit()

    if run_free and not (out / "free" / "free_summary.json").exists():
        from experiments.residual import free_generation

        free_generation(
            cfg,
            {
                key: selected[key]
                for key in ("rule", "layer", "basis", "center", "base_center")
            },
            out / "free",
        )
        scam_volume.commit()
    result = {"rule": rule, "layer": selected["layer"]}
    if variant:
        result["variant"] = variant
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=[huggingface_secret],
    volumes={
        "/vol": scam_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def transfer_all(
    layer: int,
    out_dir: str,
    rules: list[str] = RULES,
    variant: str = "",
):
    _prepare()
    cfg = _config(variant)
    out = Path(out_dir)
    if (out / "experiment.json").exists():
        required = ("subspaces.pt", "selection.json", "summary.json", "direction_geometry.json")
        if not all((out / name).exists() for name in required):
            raise RuntimeError(f"Transfer output is incomplete and cannot be safely resumed: {out}")
        print("Transfer already complete; skipping.")
        return str(out)
    missing = [
        rule
        for rule in rules
        if not (Path(cfg["rules"][rule]["adapter"]) / "adapter_model.safetensors").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Train these rules before transfer: {missing}")

    from experiments.residual import transfer

    transfer(cfg, rules, rules, layer, out, ranks=(1, 2, 4, 8), method="consensus")
    scam_volume.commit()
    return str(out)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    secrets=[huggingface_secret],
    volumes={
        "/vol": scam_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def smoke():
    secret_keys = _map_huggingface_secret()
    _prepare()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.data import config, read
    from sparse_autoencoders.run_sae import transformer_layers
    from training.train import format_pair

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Hugging Face secret key names: {secret_keys}")
    cfg = config(CONFIG_PATH)
    paths = sorted(
        {
            spec[key]
            for spec in cfg["rules"].values()
            for key in ("train", "eval", "saved_splits")
            if key in spec
        }
        | {"data/experiments/baselines/clause_order/base_ethics.jsonl"}
    )
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing configured data paths: {missing}")
    print(f"Container data paths: {len(paths)} present")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    for rule, spec in cfg["rules"].items():
        longest = max(
            sum(
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in format_pair(record)
            )
            + 1
            for record in read(spec["train"])
        )
        print(f"Longest training sequence {rule}: {longest} tokens")
        assert longest <= cfg["training"]["max_length"], (rule, longest)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16,
        token=token,
    ).to("cuda")
    layer_count = len(transformer_layers(model))
    print(f"Transformer layers: {layer_count}")
    assert layer_count == 36, layer_count
    del model
    torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(0), "layers": layer_count}


def _transfer_rules(rules: list[str], variant: str = "") -> list[str]:
    return RULES if variant else rules


@app.local_entrypoint()
def main(stage: str, rules: str = "s1,voice,clause,lexical", variant: str = ""):
    if variant:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        selected_rules = [VARIANTS[variant]["rule"]]
    else:
        selected_rules = [rule.strip() for rule in rules.split(",") if rule.strip()]
        unknown = set(selected_rules) - set(RULES)
        if unknown:
            raise ValueError(f"Unknown rules: {sorted(unknown)}")

    if stage == "smoke":
        smoke.remote()
    elif stage == "train":
        variants = [variant] * len(selected_rules)
        for rule in train_rule.map(selected_rules, variants):
            print(f"Training/baseline finished: {rule}")
    elif stage == "scan_full":
        layers = list(range(36))
        output_names = [variant or rule for rule in selected_rules]
        out_dirs = [
            f"/vol/outputs/scans/3b_full/{name}_residual_scan_3b_full"
            for name in output_names
        ]
        for result in scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            [True] * len(selected_rules),
            [variant] * len(selected_rules),
        ):
            print(f"Full scan finished: {result}")
    elif stage == "scan_late":
        layers = [30, 31, 32, 33]
        output_names = [variant or rule for rule in selected_rules]
        out_dirs = [
            f"/vol/outputs/scans/3b_late_l30plus/{name}_residual_scan_l30plus"
            for name in output_names
        ]
        for result in scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            [False] * len(selected_rules),
            [variant] * len(selected_rules),
        ):
            print(f"Late scan finished: {result}")
    elif stage == "transfer":
        transfer_rules = _transfer_rules(selected_rules, variant)
        out_dir = _transfer_out_dir(transfer_rules, variant)
        print(f"Transfer output: {out_dir}")
        print(transfer_all.remote(30, out_dir, transfer_rules, variant))
    else:
        raise ValueError(f"Unknown stage: {stage}")

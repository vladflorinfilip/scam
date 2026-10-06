"""Modal runner for the Qwen2.5-3B four-rule replication."""
from copy import deepcopy
import json
import os
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "/root/scam" not in sys.path:
    sys.path.append("/root/scam")

from experiments.variants_3b import SEED0_ALIASES, VARIANTS


REMOTE_ROOT = Path("/root/scam")
CONFIG_PATH = "configs/stegano_experiments_3b.yaml"
RULES = ["s1", "voice", "clause", "lexical"]
GPU = "A100-80GB"
TIMEOUT = 2 * 60 * 60


def _transfer_out_dir(rules: list[str], variant: str = "", layer: int = 30) -> str:
    if variant:
        name = f"four_rule_3b_l{layer}_{variant}"
    elif rules == RULES:
        name = f"four_rule_3b_l{layer}"
    else:
        name = f"rules_{'-'.join(rules)}_3b_l{layer}"
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
    if variant in SEED0_ALIASES:
        raise ValueError(f"Seed-0 aliases are scan-only: {variant}")
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
def free_strong(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    _prepare()
    rule = VARIANTS[variant]["rule"]
    cfg = _config(variant)
    out = Path("/vol/outputs/scans/3b_seeds") / f"{variant}_residual_scan_l24plus"
    free_out = out / "free_strongest"
    free_summary = free_out / "free_summary.json"
    strongest_path = free_out / "strongest.json"

    if free_summary.exists():
        print(f"Strongest-layer generation already complete for {variant}; skipping.")
        return {"variant": variant, "skipped": True}
    if strongest_path.exists():
        existing = json.loads(strongest_path.read_text())
        if existing.get("same_as_selected"):
            print(f"Strongest layer matches selected layer for {variant}; skipping.")
            return existing

    experiment_path = out / "experiment.json"
    layer_scan_path = out / "layer_scan.json"
    if not experiment_path.is_file() or not layer_scan_path.is_file():
        raise FileNotFoundError(f"Complete scan_seed output required for {variant}: {out}")
    experiment = json.loads(experiment_path.read_text())
    scores = json.loads(layer_scan_path.read_text())
    layer, selection_metrics = _strongest_layer(scores)
    selected_layer = int(experiment["selected_layer"])
    record = {
        "layer": layer,
        "same_as_selected": layer == selected_layer,
        "selection_metrics": selection_metrics,
    }
    free_out.mkdir(parents=True, exist_ok=True)

    if record["same_as_selected"]:
        from experiments.data import save

        save(strongest_path, record)
        scam_volume.commit()
        return record

    import torch

    from experiments.data import save
    from experiments.residual import basis_for, free_generation, geometry

    scan_activations = torch.load(
        out / "activations.pt",
        map_location="cpu",
        weights_only=False,
    )
    layer_index = scan_activations["layers"].index(layer)
    acts = scan_activations["activations"]
    contrast, center, base_center = geometry(acts, rule, layer_index)
    basis = basis_for([contrast], 1, "mean")
    free_generation(
        cfg,
        {
            "rule": rule,
            "layer": layer,
            "basis": basis,
            "center": center,
            "base_center": base_center,
        },
        free_out,
    )
    save(strongest_path, record)
    scam_volume.commit()
    return record


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


def _strongest_layer(scores: dict) -> tuple[int, dict]:
    layer = max(
        scores,
        key=lambda value: (
            scores[value]["follow_drop"],
            scores[value]["gap_recovery"],
        ),
    )
    return int(layer), scores[layer]


def _print_map_results(stage: str, names: list[str], results):
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            print(f"{stage} failed for {name}: {result}")
        else:
            print(f"{stage} finished for {name}: {result}")


@app.local_entrypoint()
def main(stage: str, rules: str = "s1,voice,clause,lexical", variant: str = "", layer: int = 30):
    selected_variants = [
        name.strip() for name in variant.split(",") if name.strip()
    ]
    unknown_variants = set(selected_variants) - set(VARIANTS)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    if stage == "transfer" and len(selected_variants) > 1:
        raise ValueError("Transfer accepts at most one variant")

    if selected_variants:
        selected_rules = [VARIANTS[name]["rule"] for name in selected_variants]
    else:
        selected_rules = [rule.strip() for rule in rules.split(",") if rule.strip()]
        unknown = set(selected_rules) - set(RULES)
        if unknown:
            raise ValueError(f"Unknown rules: {sorted(unknown)}")
    job_variants = selected_variants or [""] * len(selected_rules)
    output_names = selected_variants or selected_rules

    if stage == "smoke":
        smoke.remote()
    elif stage == "train":
        seed0 = set(selected_variants) & set(SEED0_ALIASES)
        if seed0:
            raise ValueError(f"Seed-0 aliases are scan-only: {sorted(seed0)}")
        results = train_rule.map(
            selected_rules,
            job_variants,
            return_exceptions=True,
        )
        _print_map_results("Training/baseline", output_names, results)
    elif stage == "scan_full":
        layers = list(range(36))
        out_dirs = [
            f"/vol/outputs/scans/3b_full/{name}_residual_scan_3b_full"
            for name in output_names
        ]
        results = scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            [True] * len(selected_rules),
            job_variants,
            return_exceptions=True,
        )
        _print_map_results("Full scan", output_names, results)
    elif stage == "scan_late":
        layers = [30, 31, 32, 33]
        out_dirs = [
            f"/vol/outputs/scans/3b_late_l30plus/{name}_residual_scan_l30plus"
            for name in output_names
        ]
        results = scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            [False] * len(selected_rules),
            job_variants,
            return_exceptions=True,
        )
        _print_map_results("Late scan", output_names, results)
    elif stage == "scan_seed":
        if not selected_variants:
            raise ValueError("scan_seed requires --variant")
        layers = list(range(24, 36))
        out_dirs = [
            f"/vol/outputs/scans/3b_seeds/{name}_residual_scan_l24plus"
            for name in selected_variants
        ]
        results = scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            [True] * len(selected_rules),
            selected_variants,
            return_exceptions=True,
        )
        _print_map_results("Seed scan", selected_variants, results)
    elif stage == "free_strong":
        if not selected_variants:
            raise ValueError("free_strong requires --variant")
        results = free_strong.map(
            selected_variants,
            return_exceptions=True,
        )
        _print_map_results("Strongest-layer free generation", selected_variants, results)
    elif stage == "transfer":
        transfer_variant = selected_variants[0] if selected_variants else ""
        transfer_rules = _transfer_rules(selected_rules, transfer_variant)
        out_dir = _transfer_out_dir(transfer_rules, transfer_variant, layer)
        print(f"Transfer output: {out_dir}")
        print(
            transfer_all.remote(
                layer,
                out_dir,
                transfer_rules,
                transfer_variant,
            )
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

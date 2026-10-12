"""Modal runner for the GPT-OSS-20B four-rule replication."""

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "/root/scam" not in sys.path:
    sys.path.append("/root/scam")

REMOTE_ROOT = Path("/root/scam")
CONFIG_PATH = "configs/stegano_experiments_gptoss20b.yaml"
RULES = ["s1", "voice", "clause", "lexical"]
GPU = "H100"
TIMEOUT = 4 * 60 * 60
VOLUME_NAME = "scam-gptoss"
PROFILE = "vf278"
GPU_HOURLY_RATE_USD = 3.95

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
    "data/inputs/training_data/kernel_flip/clause_flipped_train.jsonl",
    "data/inputs/training_data/kernel_flip/voice_flipped_train.jsonl",
]


def build_image():
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.8.0",
            "transformers==4.56.2",
            "peft==0.17.1",
            "accelerate",
            "datasets<4",
            "pyarrow==19.0.1",
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


app = modal.App("scam-gptoss-20b")
image = build_image()
scam_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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


def _config():
    from experiments.data import config

    return config(CONFIG_PATH)


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
def train_rule(rule: str):
    _prepare()
    cfg = _config()
    adapter = Path(cfg["rules"][rule]["adapter"])
    if (adapter / "adapter_model.safetensors").exists():
        print(f"Training already complete for {rule}; skipping.")
    else:
        from experiments.colab import train

        train(cfg, rule)
        scam_volume.commit()

    baseline_dir = Path("/vol/outputs/baselines") / rule
    if (baseline_dir / "experiment.json").exists():
        print(f"Baseline already complete for {rule}; skipping.")
    else:
        from experiments.data import save, splits
        from experiments.residual import export, forward, loaded, metrics, record_run, texts

        pairs = splits(cfg, rule)["eval"]
        originals = {}
        for name, path in [
            ("base", cfg["base_model"]),
            ("adapter", cfg["rules"][rule]["adapter"]),
        ]:
            with loaded(path) as (tok, model):
                originals[name] = forward(model, tok, texts(pairs), cfg)["margin"]

        baseline_dir.mkdir(parents=True, exist_ok=True)
        assert not (baseline_dir / "experiment.json").exists(), "Use a fresh output directory"
        export(baseline_dir, rule, pairs, {"adapter": {}, "base": {}}, originals)
        adapter_margin, base_margin = originals["adapter"], originals["base"]
        save(
            baseline_dir / "paired_summary.json",
            metrics(adapter_margin, adapter_margin, base_margin, base_margin),
        )
        record_run(
            cfg,
            baseline_dir,
            {"rules": [rule], "stage": "post-training base versus final-epoch adapter baselines"},
        )
        scam_volume.commit()
    return rule


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
def scan_rule(rule: str, layers: list[int], out_dir: str):
    _prepare()
    cfg = _config()
    adapter = Path(cfg["rules"][rule]["adapter"]) / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(f"Train {rule} before scanning; adapter missing: {adapter}")
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
        print(f"Scan already complete for {rule}; skipping.")
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
    return {"rule": rule, "layer": selected["layer"]}


def _run(*args):
    command = [sys.executable, "-u", *map(str, args)]
    lines = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    return "".join(lines)


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
    smoke_started = time.monotonic()
    _prepare()

    import torch

    from experiments.data import read, splits
    from experiments.residual import forward, loaded, texts
    from sparse_autoencoders.run_sae import transformer_layers

    cfg = _config()
    spec = cfg["rules"]["s1"]
    pairs = splits(cfg, "s1")["eval"][:8]
    eval_texts = texts(pairs)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with loaded(cfg["base_model"]) as (tok, model):
        label_tokens = {
            label: tok.encode(label, add_special_tokens=False) for label in ("0", "1")
        }
        print(f"Label token IDs: {label_tokens}")
        assert all(len(ids) == 1 for ids in label_tokens.values()), label_tokens

        layer_count = len(transformer_layers(model))
        hidden_size = model.config.hidden_size
        print(f"Transformer layers: {layer_count}")
        print(f"Hidden size: {hidden_size}")
        assert layer_count == 24, layer_count
        assert hidden_size == 2880, hidden_size

        forward_started = time.monotonic()
        result = forward(model, tok, eval_texts, cfg, layers=(0, 12, 23))
        forward_seconds = time.monotonic() - forward_started
        assert result["h"].shape == (16, 3, 2880), tuple(result["h"].shape)
        assert torch.isfinite(result["margin"]).all()
        valid_fraction = float((result["greedy_label"] >= 0).float().mean())
        print(f"Hidden state shape: {tuple(result['h'].shape)}")
        print(f"Greedy-label-valid fraction: {valid_fraction:.4f}")
        print(f"Forward elapsed seconds: {forward_seconds:.2f}")
        print(f"Margins sample: {result['margin'][:8].tolist()}")

        basis = torch.randn((2880, 1), generator=torch.Generator().manual_seed(0))
        basis /= basis.norm()
        center = torch.zeros(2880)
        intervened = forward(
            model,
            tok,
            eval_texts,
            cfg,
            intervention=(12, basis, center),
        )
        margin_delta = float((intervened["margin"] - result["margin"]).abs().max())
        print(f"Intervention max absolute margin change: {margin_delta:.6f}")
        assert margin_delta > 0, margin_delta
        margins_sample = result["margin"][:8].tolist()
        base_peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    model_seconds = time.monotonic() - started
    print(f"Base-model peak allocated GPU memory: {base_peak_gib:.3f} GiB")
    print(f"Model smoke elapsed seconds: {model_seconds:.2f}")
    del tok, model, result, intervened
    torch.cuda.empty_cache()

    timing_dir = Path("/vol/tmp_timing")
    if timing_dir.exists():
        raise FileExistsError(f"Refusing to overwrite timing output: {timing_dir}")
    train_rows_by_rule = {
        rule: len(read(cfg["rules"][rule]["train"])) for rule in RULES
    }
    train_rows = train_rows_by_rule["s1"]
    options = [
        arg
        for key, value in cfg["training"].items()
        for arg in ("--" + key.replace("_", "-"), str(value))
    ]
    command = [
        "training/train.py",
        "--model",
        cfg["base_model"],
        "--data",
        spec["train"],
        "--output-dir",
        timing_dir,
        "--val-data",
        "",
        "--val-fraction",
        "0",
        "--lora",
        *options,
        "--max-steps",
        "20",
    ]
    try:
        train_output = _run(*command)
        training_log = json.loads((timing_dir / "training_log.json").read_text())
        train_summary = next(
            entry for entry in reversed(training_log) if "train_runtime" in entry
        )
        training_seconds = float(train_summary["train_runtime"])
        seconds_per_step = training_seconds / 20
        batch_size = 1
        gradient_accumulation = 8
        steps_per_epoch = math.ceil(
            math.ceil(train_rows / batch_size) / gradient_accumulation
        )
        projected_steps = steps_per_epoch * cfg["training"]["epochs"]
        projected_steps_by_rule = {
            rule: (
                math.ceil(
                    math.ceil(rows / batch_size) / gradient_accumulation
                )
                * cfg["training"]["epochs"]
            )
            for rule, rows in train_rows_by_rule.items()
        }
        projected_hours_by_rule = {
            rule: seconds_per_step * steps / 3600
            for rule, steps in projected_steps_by_rule.items()
        }
        memory_match = re.search(r"peak_cuda_memory_gib:\s*([0-9.]+)", train_output)
        if memory_match is None:
            raise RuntimeError("Training output did not report peak CUDA memory")
        training_peak_gib = float(memory_match.group(1))
        print(f"Timing run optimizer steps: 20")
        print(f"Training runtime seconds: {training_seconds:.2f}")
        print(f"Training seconds per step: {seconds_per_step:.3f}")
        print(f"Three-epoch projected s1 steps: {projected_steps}")
        print(f"Three-epoch projected steps by rule: {projected_steps_by_rule}")
        print(
            "Projected training hours per rule at the s1 measured step time: "
            f"{projected_hours_by_rule}"
        )
        print(f"Timing-run peak allocated GPU memory: {training_peak_gib:.3f} GiB")
        smoke_seconds = time.monotonic() - smoke_started
        estimated_spend = smoke_seconds / 3600 * GPU_HOURLY_RATE_USD
        print(f"Smoke GPU time: {smoke_seconds:.1f} seconds")
        print(f"Approximate H100 spend: ${estimated_spend:.2f}")
        return {
            "gpu": torch.cuda.get_device_name(0),
            "layers": layer_count,
            "hidden_size": hidden_size,
            "label_tokens": label_tokens,
            "greedy_label_valid_fraction": valid_fraction,
            "margins_sample": margins_sample,
            "intervention_max_abs_margin_change": margin_delta,
            "forward_seconds": forward_seconds,
            "base_peak_allocated_gib": base_peak_gib,
            "training_seconds_per_step": seconds_per_step,
            "three_epoch_steps": projected_steps,
            "three_epoch_steps_by_rule": projected_steps_by_rule,
            "projected_training_hours_per_rule": projected_hours_by_rule,
            "training_peak_allocated_gib": training_peak_gib,
            "smoke_gpu_seconds": smoke_seconds,
            "approximate_spend_usd": estimated_spend,
        }
    finally:
        if timing_dir.exists():
            shutil.rmtree(timing_dir)
            scam_volume.commit()


def _print_map_results(stage: str, names: list[str], results):
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            print(f"{stage} failed for {name}: {result}")
        else:
            print(f"{stage} finished for {name}: {result}")


def _run_modal(*args):
    command = ["modal", *map(str, args[:2]), "--profile", PROFILE, *map(str, args[2:])]
    subprocess.run(command, cwd=ROOT, check=True)


def _pull_directory(remote: str, destination: Path):
    with tempfile.TemporaryDirectory(prefix="pull-gptoss-") as temporary:
        _run_modal("volume", "get", VOLUME_NAME, remote, temporary)
        source = Path(temporary) / PurePosixPath(remote).name
        if not source.is_dir():
            raise RuntimeError(f"Modal did not download the expected directory: {remote}")
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            target = destination / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def _pull():
    _pull_directory("outputs", ROOT / "data/experiments/gptoss20b")
    for rule in RULES:
        adapter = f"gptoss20b-cot-sft-{rule}"
        _pull_directory(f"checkpoints/{adapter}", ROOT / "checkpoints" / adapter)


@app.local_entrypoint()
def main(stage: str, rules: str = "s1,voice,clause,lexical"):
    selected_rules = [rule.strip() for rule in rules.split(",") if rule.strip()]
    unknown = set(selected_rules) - set(RULES)
    if unknown:
        raise ValueError(f"Unknown rules: {sorted(unknown)}")

    if stage == "smoke":
        print(smoke.remote())
    elif stage == "train":
        results = train_rule.map(selected_rules, return_exceptions=True)
        _print_map_results("Training/baseline", selected_rules, results)
    elif stage == "scan_full":
        layers = list(range(24))
        out_dirs = [f"/vol/outputs/scans/full/{rule}" for rule in selected_rules]
        results = scan_rule.map(
            selected_rules,
            [layers] * len(selected_rules),
            out_dirs,
            return_exceptions=True,
        )
        _print_map_results("Full scan", selected_rules, results)
    elif stage == "pull":
        _pull()
    else:
        raise ValueError(f"Unknown stage: {stage}")

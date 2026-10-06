"""Pull Modal 3B artifacts while avoiding GitHub's large-file limit."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.variants_3b import SEED0_ALIASES, VARIANTS


VOLUME = "scam-3b"
RULES = ("s1", "voice", "clause", "lexical")
PULL_RULES = (*RULES, *VARIANTS)
STAGES = ("adapter", "baseline", "scan_full", "scan_late", "scan_seed", "transfer")
LIMIT = 95 * 1024 * 1024
SKIP_TRAINING_FILES = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
LOCAL_CRITIC_FILES = {
    "free_critic_summary.json",
    "critic_metadata.json",
    "clause_critic_sentence_cache.json",
}


def run_modal(*args, capture=False):
    command = ["modal", *map(str, args)]
    if capture:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result.stdout
    subprocess.run(command, check=True)
    return None


def volume_entries(path):
    try:
        output = run_modal("volume", "ls", "--json", VOLUME, path, capture=True)
    except subprocess.CalledProcessError as error:
        if "No such file or directory" in (error.stderr or ""):
            return []
        raise
    return json.loads(output)


def volume_file_exists(path):
    marker = PurePosixPath(path)
    return any(
        entry["type"] == "file"
        and PurePosixPath(entry["filename"]).name == marker.name
        for entry in volume_entries(marker.parent.as_posix())
    )


def files_under(path):
    entries = volume_entries(path)
    files = []
    for entry in entries:
        candidate = PurePosixPath(entry["filename"])
        if not candidate.is_absolute() and not candidate.as_posix().startswith(path.rstrip("/") + "/"):
            candidate = PurePosixPath(path) / candidate
        name = candidate.name.lower()
        if entry["type"] == "dir":
            if not candidate.name.startswith("checkpoint-"):
                files.extend(files_under(candidate.as_posix()))
        elif entry["type"] == "file":
            if name not in SKIP_TRAINING_FILES and not name.startswith(
                ("optimizer.", "scheduler.", "rng_state")
            ):
                files.append(candidate.as_posix().lstrip("/"))
    return files


def pull_file(remote, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_modal(
        "volume",
        "get",
        "--force",
        VOLUME,
        remote,
        destination,
    )


def pull_directory(remote, destination):
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix="pull-3b-") as temporary:
        staging = Path(temporary)
        run_modal("volume", "get", "--force", VOLUME, remote, staging)
        source = staging / PurePosixPath(remote).name
        if not source.is_dir():
            raise RuntimeError(f"Modal did not download the expected directory: {remote}")
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not is_local_critic_output(relative):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def is_local_critic_output(path):
    return path.name in LOCAL_CRITIC_FILES or (
        path.name.startswith("free_") and path.name.endswith("_critic.jsonl")
    )


def tensor_shapes(value, prefix=""):
    import torch

    result = {}
    if isinstance(value, torch.Tensor):
        result[prefix or "$"] = list(value.shape)
    elif isinstance(value, dict):
        for key, child in value.items():
            result.update(tensor_shapes(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(tensor_shapes(child, f"{prefix}[{index}]"))
    return result


def to_fp16(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.half() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: to_fp16(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_fp16(child) for child in value]
    if isinstance(value, tuple):
        return tuple(to_fp16(child) for child in value)
    return value


def slice_examples(value, start, end, total):
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == total:
            return value[start:end].clone()
        return value
    if isinstance(value, dict):
        return {key: slice_examples(child, start, end, total) for key, child in value.items()}
    if isinstance(value, list):
        return [slice_examples(child, start, end, total) for child in value]
    if isinstance(value, tuple):
        return tuple(slice_examples(child, start, end, total) for child in value)
    return value


def save_torch(path, value):
    import torch

    torch.save(value, path)
    return path.stat().st_size


def split_oversized_piece(path, wrapper, split_name, split_data):
    first_tensor = next(
        (
            tensor
            for tensor in _tensors(split_data)
            if tensor.ndim and tensor.shape[0] > 1
        ),
        None,
    )
    if first_tensor is None:
        raise RuntimeError(f"Activation piece remains over 95 MiB and cannot be split: {path}")
    total = first_tensor.shape[0]
    chunk_size = total
    while True:
        outputs = []
        for index, start in enumerate(range(0, total, chunk_size)):
            end = min(start + chunk_size, total)
            piece = {
                "layers": wrapper["layers"],
                "activations": {
                    rule: {
                        arm: {
                            f"{split_name}_part_{index:03d}": slice_examples(
                                data, start, end, total
                            )
                        }
                    }
                    for rule, arms in wrapper["activations"].items()
                    for arm, splits in arms.items()
                    for name, data in splits.items()
                    if name == split_name
                },
            }
            candidate = path.with_name(f"{path.stem}_part_{index:03d}{path.suffix}")
            if save_torch(candidate, piece) > LIMIT:
                candidate.unlink()
                for output in outputs:
                    (path.parent / output).unlink(missing_ok=True)
                chunk_size //= 2
                if chunk_size == 0:
                    raise RuntimeError(f"Cannot split activation piece below 95 MiB: {path}")
                break
            outputs.append(candidate.name)
        else:
            return outputs


def _tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensors(child)


def convert_activations(path):
    import torch

    path = Path(path)
    if path.stat().st_size <= LIMIT:
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("activations"), dict):
        return

    manifest = {
        "source_file": path.name,
        "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_shapes": tensor_shapes(payload),
        "outputs": [],
    }
    layers = payload.get("layers")
    activations = payload["activations"]
    arms = sorted(
        {
            arm
            for rule_arms in activations.values()
            for arm in rule_arms
        }
    )
    for arm in arms:
        arm_data = {
            rule: {arm: to_fp16(rule_arms[arm])}
            for rule, rule_arms in activations.items()
            if arm in rule_arms
        }
        wrapper = {"layers": layers, "activations": arm_data}
        output_path = path.with_name(f"activations_{arm}_fp16.pt")
        if save_torch(output_path, wrapper) > LIMIT:
            output_path.unlink()
            split_names = sorted(
                {
                    split
                    for rule_arms in arm_data.values()
                    for splits in rule_arms.values()
                    for split in splits
                }
            )
            for split in split_names:
                split_data = {}
                for rule, rule_arms in arm_data.items():
                    if split in rule_arms[arm]:
                        split_data[rule] = {arm: {split: rule_arms[arm][split]}}
                split_wrapper = {"layers": layers, "activations": split_data}
                split_path = path.with_name(f"activations_{arm}_{split}_fp16.pt")
                if save_torch(split_path, split_wrapper) > LIMIT:
                    manifest["outputs"].extend(
                        {
                            "file": name,
                            "shapes": tensor_shapes(torch.load(
                                path.parent / name,
                                map_location="cpu",
                                weights_only=False,
                            )),
                        }
                        for name in split_oversized_piece(
                            split_path, split_wrapper, split, next(
                                splits[split]
                                for rule_arms in split_wrapper["activations"].values()
                                for splits in rule_arms.values()
                            )
                        )
                    )
                    split_path.unlink(missing_ok=True)
                else:
                    manifest["outputs"].append(
                        {"file": split_path.name, "shapes": tensor_shapes(split_wrapper)}
                    )
        else:
            manifest["outputs"].append(
                {"file": output_path.name, "shapes": tensor_shapes(wrapper)}
            )

    manifest_path = path.with_name("activations_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    path.unlink()


def sanitize_downloads(roots):
    for root in roots:
        if root.exists():
            for path in root.rglob("activations.pt"):
                convert_activations(path)
            oversized = [
                (path, path.stat().st_size)
                for path in root.rglob("*")
                if path.is_file() and path.stat().st_size > LIMIT
            ]
            if oversized:
                raise RuntimeError(f"Artifacts still exceed 95 MiB: {oversized}")


def main():
    parser = argparse.ArgumentParser()
    global VOLUME
    parser.add_argument("--volume", default=VOLUME)
    parser.add_argument(
        "--rules",
        default=",".join(RULES),
        help="Comma-separated rules/variants for per-rule stages (default: standard rules)",
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Comma-separated stages to pull (default: all stages)",
    )
    parser.add_argument(
        "--transfer-name",
        default="four_rule_3b_l30",
        help="Transfer artifact name (default: four_rule_3b_l30)",
    )
    args = parser.parse_args()
    VOLUME = args.volume
    if (
        not args.transfer_name
        or "/" in args.transfer_name
        or "\\" in args.transfer_name
        or args.transfer_name in {".", ".."}
    ):
        parser.error("--transfer-name must be a single directory name")

    def parse_filter(value, choices, name):
        parts = (part.strip() for part in value.split(","))
        selected = tuple(dict.fromkeys(part for part in parts if part))
        invalid = sorted(set(selected) - set(choices))
        if not selected or invalid:
            raise ValueError(
                f"--{name} must be a comma-separated list of {', '.join(choices)}"
                + (f"; unknown: {', '.join(invalid)}" if invalid else "")
            )
        return selected

    try:
        selected_rules = parse_filter(args.rules, PULL_RULES, "rules")
        selected_stages = set(parse_filter(args.stages, STAGES, "stages"))
    except ValueError as error:
        parser.error(str(error))

    print(f"Rules: {','.join(selected_rules)}")
    print(f"Stages: {','.join(stage for stage in STAGES if stage in selected_stages)}")
    sanitize_roots = []

    for rule in selected_rules:
        if "adapter" in selected_stages:
            remote = f"checkpoints/qwen3b-cot-sft-{rule.replace('_', '-')}"
            marker = f"{remote}/adapter_model.safetensors"
            if rule in SEED0_ALIASES:
                print(f"SKIPPED stage=adapter rule={rule}: seed-0 scan-only alias", flush=True)
            elif volume_file_exists(marker):
                files = files_under(remote)
                for file in files:
                    pull_file(file, ROOT / file)
                print(f"PULLED stage=adapter rule={rule} files={len(files)}", flush=True)
            else:
                print(f"SKIPPED stage=adapter rule={rule}: missing marker {marker}", flush=True)

        if "baseline" in selected_stages:
            remote = f"outputs/baselines/{rule}"
            marker = f"{remote}/experiment.json"
            if rule in SEED0_ALIASES:
                print(f"SKIPPED stage=baseline rule={rule}: seed-0 scan-only alias", flush=True)
            elif volume_file_exists(marker):
                pull_directory(remote, ROOT / "data/experiments/baselines/3b" / rule)
                print(f"PULLED stage=baseline rule={rule}", flush=True)
            else:
                print(f"SKIPPED stage=baseline rule={rule}: missing marker {marker}", flush=True)

        if "scan_full" in selected_stages:
            scan_name = f"{rule}_residual_scan_3b_full"
            remote = f"outputs/scans/3b_full/{scan_name}"
            marker = f"{remote}/free/free_summary.json"
            if volume_file_exists(marker):
                destination = ROOT / "data/experiments/residuals/scans/3b_full" / scan_name
                pull_directory(remote, destination)
                sanitize_roots.append(destination)
                print(f"PULLED stage=scan_full rule={rule}", flush=True)
            else:
                print(f"SKIPPED stage=scan_full rule={rule}: missing marker {marker}", flush=True)

        if "scan_late" in selected_stages:
            scan_name = f"{rule}_residual_scan_l30plus"
            remote = f"outputs/scans/3b_late_l30plus/{scan_name}"
            marker = f"{remote}/summary.json"
            if volume_file_exists(marker):
                destination = (
                    ROOT / "data/experiments/residuals/scans/3b_late_l30plus" / scan_name
                )
                pull_directory(remote, destination)
                sanitize_roots.append(destination)
                print(f"PULLED stage=scan_late rule={rule}", flush=True)
            else:
                print(f"SKIPPED stage=scan_late rule={rule}: missing marker {marker}", flush=True)

        if "scan_seed" in selected_stages:
            scan_name = f"{rule}_residual_scan_l24plus"
            remote = f"outputs/scans/3b_seeds/{scan_name}"
            marker = f"{remote}/free/free_summary.json"
            if volume_file_exists(marker):
                destination = (
                    ROOT / "data/experiments/residuals/scans/3b_seeds" / scan_name
                )
                pull_directory(remote, destination)
                sanitize_roots.append(destination)
                print(f"PULLED stage=scan_seed rule={rule}", flush=True)
            else:
                print(f"SKIPPED stage=scan_seed rule={rule}: missing marker {marker}", flush=True)

    if "transfer" in selected_stages:
        remote = f"outputs/transfer/{args.transfer_name}"
        marker = f"{remote}/summary.json"
        if volume_file_exists(marker):
            destination = ROOT / "data/experiments/residuals/transfer" / args.transfer_name
            pull_directory(remote, destination)
            sanitize_roots.append(destination)
            print(f"PULLED stage=transfer name={args.transfer_name}", flush=True)
        else:
            print(
                f"SKIPPED stage=transfer name={args.transfer_name}: missing marker {marker}",
                flush=True,
            )

    sanitize_downloads(sanitize_roots)


if __name__ == "__main__":
    main()

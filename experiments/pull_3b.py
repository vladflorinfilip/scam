"""Pull Modal 3B artifacts while avoiding GitHub's large-file limit."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
VOLUME = "scam-3b"
RULES = ("s1", "voice", "clause", "lexical")
LIMIT = 95 * 1024 * 1024
SKIP_TRAINING_FILES = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}


def run_modal(*args, capture=False):
    command = ["modal", *map(str, args)]
    if capture:
        return subprocess.check_output(command, text=True)
    subprocess.run(command, check=True)
    return None


def volume_entries(path):
    output = run_modal("volume", "ls", "--json", VOLUME, path, capture=True)
    return json.loads(output)


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
    destination.mkdir(parents=True, exist_ok=True)
    run_modal(
        "volume",
        "get",
        "--force",
        VOLUME,
        remote,
        destination,
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


def sanitize_downloads():
    roots = (
        ROOT / "data/experiments/residuals/scans/3b_full",
        ROOT / "data/experiments/residuals/scans/3b_late_l30plus",
        ROOT / "data/experiments/residuals/transfer/four_rule_3b_l30",
    )
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
    args = parser.parse_args()
    VOLUME = args.volume

    for rule in RULES:
        remote = f"checkpoints/qwen3b-cot-sft-{rule}"
        for file in files_under(remote):
            pull_file(file, ROOT / file)
        baseline = f"outputs/baselines/{rule}"
        pull_directory(baseline, ROOT / f"data/experiments/baselines/3b/{rule}")

    pull_directory(
        "outputs/scans/3b_full",
        ROOT / "data/experiments/residuals/scans/3b_full",
    )
    pull_directory(
        "outputs/scans/3b_late_l30plus",
        ROOT / "data/experiments/residuals/scans/3b_late_l30plus",
    )
    pull_directory(
        "outputs/transfer/four_rule_3b_l30",
        ROOT / "data/experiments/residuals/transfer/four_rule_3b_l30",
    )
    sanitize_downloads()


if __name__ == "__main__":
    main()

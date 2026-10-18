"""Collection and causal-ablation helpers for the close-gap Qwen 3B analyses."""
import json
from copy import deepcopy
from pathlib import Path

import torch

from experiments.data import config, save, splits
from experiments.residual import (
    basis_for,
    follow,
    forward,
    forward_cached_arms,
    geometry,
    loaded,
    random_basis,
    texts,
)
from experiments.variants_3b import VARIANTS


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "select", "eval")
LIMIT_BYTES = 95 * 1024 * 1024


def base_config(variant):
    path = VARIANTS.get(variant, {}).get("config", "configs/stegano_experiments_3b.yaml")
    return deepcopy(config(path))


def variant_config(variant):
    cfg = base_config(variant)
    spec = VARIANTS[variant]
    rule = spec["rule"]
    cfg["training"].update(spec["training"])
    cfg["rules"][rule]["adapter"] = spec["adapter"]
    if "train" in spec:
        cfg["rules"][rule]["train"] = spec["train"]
    cfg["variant"] = variant
    return cfg


def model_key(cfg):
    return "qwen05b" if "0.5B" in cfg["base_model"] else "qwen3b"


def _limit_data(data, pair_limit):
    if pair_limit:
        return {name: pairs[:pair_limit] for name, pairs in data.items()}
    return data


def _stored(output):
    return {
        key: (value.to(torch.bfloat16) if key == "h" else value)
        for key, value in output.items()
    }


def _follow_summary(acts):
    return {
        arm: {
            split: {
                "follow_original_cue": follow(value["margin"]),
                "label_tokens": float((value["greedy_label"] >= 0).float().mean()),
            }
            for split, value in splits_.items()
        }
        for arm, splits_ in acts.items()
    }


def collect_base(variant, rule, root, limit_layers=0, pair_limit=0):
    cfg = base_config(variant)
    data = _limit_data(splits(cfg, rule), pair_limit)
    cache = Path(root) / f"base_{model_key(cfg)}_{rule}"
    target = cache / "activations.pt"
    if target.is_file():
        print(f"Base cache exists: {target}", flush=True)
        return str(target)
    cache.mkdir(parents=True, exist_ok=True)
    with loaded(cfg["base_model"]) as (tok, model):
        from sparse_autoencoders.run_sae import transformer_layers

        count = len(transformer_layers(model))
        layers = list(range(min(limit_layers, count))) if limit_layers else list(range(count))
        outputs = {name: forward(model, tok, texts(pairs), cfg, layers) for name, pairs in data.items()}
    stored = {name: _stored(value) for name, value in outputs.items()}
    torch.save({"layers": layers, "activations": {rule: {"base": stored}}}, target)
    save(cache / "splits.json", {rule: data})
    save(cache / "summary.json", _follow_summary({"base": outputs}))
    if target.stat().st_size >= LIMIT_BYTES:
        raise RuntimeError(f"Base activation cache exceeds 95 MiB: {target}")
    return str(target)


def collect_variant(variant, root, limit_pairs=0, smoke=False):
    cfg = variant_config(variant)
    rule = VARIANTS[variant]["rule"]
    adapter = Path(cfg["rules"][rule]["adapter"])
    if not (adapter / "adapter_model.safetensors").is_file():
        if smoke:
            raise FileNotFoundError(f"Smoke collection never trains: {adapter}")
        from experiments.colab import train

        train(cfg, rule)
    base_path = Path(root) / f"base_{model_key(cfg)}_{rule}"
    cache_file = base_path / "activations.pt"
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing base cache; run collect base first: {cache_file}")
    base_bundle = torch.load(cache_file, map_location="cpu", weights_only=False)
    layers = base_bundle["layers"]
    data = _limit_data(splits(base_config(variant), rule), limit_pairs)
    out = Path(root) / variant
    out.mkdir(parents=True, exist_ok=True)
    save(out / "splits.json", {rule: data})
    summary = {"base": {}, "adapter": {}}
    for split in SPLITS:
        split_path = out / f"activations_{split}.pt"
        if split_path.is_file():
            saved = torch.load(split_path, map_location="cpu", weights_only=False)
            for arm in ("base", "adapter"):
                summary[arm][split] = {
                    "follow_original_cue": follow(saved["activations"][rule][arm][split]["margin"]),
                    "label_tokens": float(
                        (saved["activations"][rule][arm][split]["greedy_label"] >= 0).float().mean()
                    ),
                }
            continue
        with loaded(str(adapter)) as (tok, model):
            output = forward(model, tok, texts(data[split]), cfg, layers)
        base_values = base_bundle["activations"][rule]["base"][split]
        base_values = {k: v for k, v in base_values.items()}
        if limit_pairs and len(base_values["margin"]) != 2 * len(data[split]):
            raise ValueError("Base cache was created with different smoke limits")
        split_acts = {
            "base": {split: base_values},
            "adapter": {split: _stored(output)},
        }
        torch.save({"layers": layers, "activations": {rule: split_acts}}, split_path)
        if split_path.stat().st_size >= LIMIT_BYTES:
            raise RuntimeError(f"Activation shard exceeds 95 MiB: {split_path}")
        for arm, value in (("base", base_values), ("adapter", output)):
            summary[arm][split] = {
                "follow_original_cue": follow(value["margin"]),
                "label_tokens": float((value["greedy_label"] >= 0).float().mean()),
            }
    save(out / "summary.json", summary)
    return str(out)


def load_collection(path, rule):
    path = Path(path)
    data = json.loads((path / "splits.json").read_text())[rule]
    acts = {rule: {arm: {} for arm in ("base", "adapter")}}
    layers = None
    for split in SPLITS:
        bundle = torch.load(path / f"activations_{split}.pt", map_location="cpu", weights_only=False)
        layers = bundle["layers"]
        for arm in ("base", "adapter"):
            value = dict(bundle["activations"][rule][arm][split])
            value["h"] = value["h"].float()
            acts[rule][arm][split] = value
    return layers, data, acts


def _write_rows(path, pairs, margins):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for side_index, side in enumerate(("pos", "neg")):
        for pair_index, pair in enumerate(pairs):
            rows.append({
                "pair_index": int(pair["pair_index"]),
                "side": side,
                "cue_label": int(side == "pos"),
                "logit_margin": float(margins[side_index * len(pairs) + pair_index]),
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _batch_forward(model, tok, cfg, pairs, layer, arms, center, use_prefix_cache=True):
    """Run many direction arms in one forward by assigning one basis per example."""
    flat_texts = texts(pairs)
    by_rank = {}
    for name, basis in arms.items():
        by_rank.setdefault(basis.shape[1], {})[name] = basis
    results = {}
    for group in by_rank.values():
        if use_prefix_cache:
            cached = forward_cached_arms(
                model,
                tok,
                flat_texts,
                cfg,
                {name: (layer, basis, center) for name, basis in group.items()},
            )
            results.update({name: value["margin"] for name, value in cached.items()})
            continue
        arm_names = list(group)
        batched_texts = flat_texts * len(arm_names)
        qs = torch.cat([group[name].unsqueeze(0).expand(len(flat_texts), -1, -1) for name in arm_names])
        margins = forward(
            model, tok, batched_texts, cfg, intervention=(layer, qs, center)
        )["margin"].reshape(len(arm_names), len(flat_texts))
        results.update(zip(arm_names, margins))
    return results


def layer_check(
    variant, collection_root, out_root, limit_layers=0, limit_pairs=0,
    random_arms=3, use_prefix_cache=True,
):
    cfg = variant_config(variant)
    rule = VARIANTS[variant]["rule"]
    layers, data, acts = load_collection(Path(collection_root) / variant, rule)
    pairs = data["eval"][:limit_pairs] if limit_pairs else data["eval"]
    if limit_layers:
        layers = layers[:limit_layers]
    out = Path(out_root) / variant
    out.mkdir(parents=True, exist_ok=True)
    eval_acts = acts[rule]
    if not (out / "unablated.jsonl").is_file():
        _write_rows(out / "unablated.jsonl", pairs, eval_acts["adapter"]["eval"]["margin"][:2 * len(pairs)])
        _write_rows(out / "base.jsonl", pairs, eval_acts["base"]["eval"]["margin"][:2 * len(pairs)])
    with loaded(str(cfg["rules"][rule]["adapter"])) as (tok, model):
        for i, layer in enumerate(layers):
            contrast, center, base_center = geometry(acts, rule, i)
            q = basis_for([contrast], 1, "mean")
            adapter_arms = {f"own_L{layer}": q}
            for k in range(random_arms):
                adapter_arms[f"random_L{layer}_{k}"] = random_basis(q.shape[0], 1, layer * 1000 + k)
            pending = {
                name: value for name, value in adapter_arms.items()
                if not (out / f"{name}.jsonl").is_file()
            }
            if pending:
                for name, margin in _batch_forward(
                    model, tok, cfg, pairs, layer, pending, center, use_prefix_cache
                ).items():
                    _write_rows(out / f"{name}.jsonl", pairs, margin)
    with loaded(cfg["base_model"]) as (tok, model):
        for i, layer in enumerate(layers):
            contrast, _, base_center = geometry(acts, rule, i)
            q = basis_for([contrast], 1, "mean")
            name = f"base_own_L{layer}"
            if (out / f"{name}.jsonl").is_file():
                continue
            margin = forward(
                model, tok, texts(pairs), cfg,
                intervention=(layer, q, base_center),
                use_prefix_cache=use_prefix_cache,
            )["margin"]
            _write_rows(out / f"{name}.jsonl", pairs, margin)
    return str(out)


def cue_ablate(
    variant, direction_file, out_root, limit_layers=0, limit_pairs=0,
    random_arms=200, use_prefix_cache=True,
):
    cfg = variant_config(variant)
    rule = VARIANTS[variant]["rule"]
    directions = torch.load(direction_file, map_location="cpu", weights_only=False)
    layers = sorted(directions["directions"])
    if limit_layers:
        layers = layers[:limit_layers]
    data = splits(base_config(variant), rule)
    pairs = data["eval"][:limit_pairs] if limit_pairs else data["eval"]
    target = Path(out_root)
    with loaded(str(cfg["rules"][rule]["adapter"])) as (tok, model):
        original = forward(model, tok, texts(pairs), cfg, use_prefix_cache=use_prefix_cache)["margin"]
        for layer in layers:
            arms = directions["directions"][layer][rule]
            means = directions["means"][layer][rule]
            folder = target / f"{rule}_L{layer}"
            if not (folder / "unablated.jsonl").is_file():
                _write_rows(folder / "unablated.jsonl", pairs, original)
            selected = {
                name: q for name, q in arms.items()
                if name in {"shared_loo", "own_cue", "own_answer"}
                or (name.startswith("random_") and int(name.split("_")[1]) < random_arms)
            }
            selected = {
                name: q for name, q in selected.items()
                if not (folder / f"{name}.jsonl").is_file()
            }
            if selected:
                for name, margin in _batch_forward(
                    model, tok, cfg, pairs, layer, selected, means, use_prefix_cache
                ).items():
                    _write_rows(folder / f"{name}.jsonl", pairs, margin)
    base_arms = ("shared_loo", "own_cue", "own_answer")
    with loaded(cfg["base_model"]) as (tok, model):
        base_original = forward(model, tok, texts(pairs), cfg, use_prefix_cache=use_prefix_cache)["margin"]
        for layer in layers:
            folder = target / f"{rule}_L{layer}"
            if not (folder / "base.jsonl").is_file():
                _write_rows(folder / "base.jsonl", pairs, base_original)
            basis = directions["directions"][layer][rule]
            means = directions["means"][layer][rule]
            selected = {name: basis[name] for name in base_arms}
            selected = {
                name: q for name, q in selected.items()
                if not (folder / f"base_{name}.jsonl").is_file()
            }
            if selected:
                for name, margin in _batch_forward(
                    model, tok, cfg, pairs, layer, selected, means, use_prefix_cache
                ).items():
                    _write_rows(folder / f"base_{name}.jsonl", pairs, margin)
    return str(target)


def verify_layer_cache(variant, collection_root, limit_pairs=8, layer=30):
    cfg = variant_config(variant)
    rule = VARIANTS[variant]["rule"]
    layers, data, acts = load_collection(Path(collection_root) / variant, rule)
    if layer not in layers:
        raise ValueError(f"Layer {layer} is missing from {variant} collection")
    pairs = data["eval"][:limit_pairs]
    contrast, center, _ = geometry(acts, rule, layers.index(layer))
    arms = {
        "unablated": None,
        f"own_L{layer}": (layer, basis_for([contrast], 1, "mean"), center),
        f"random_L{layer}_0": (layer, random_basis(contrast.shape[-1], 1, layer * 1000), center),
    }
    differences = {}
    with loaded(str(cfg["rules"][rule]["adapter"])) as (tok, model):
        for name, intervention in arms.items():
            sample_texts = texts(pairs)
            full = forward(model, tok, sample_texts, cfg, intervention=intervention)["margin"]
            cached = forward(
                model, tok, sample_texts, cfg, intervention=intervention,
                use_prefix_cache=True,
            )["margin"]
            differences[name] = float((full - cached).abs().max())
    return differences


def verify_cue_cache(variant, direction_file, limit_pairs=8, layer=28):
    cfg = variant_config(variant)
    rule = VARIANTS[variant]["rule"]
    directions = torch.load(direction_file, map_location="cpu", weights_only=False)
    if layer not in directions["directions"]:
        raise ValueError(f"Layer {layer} is missing from shared-cue directions")
    pairs = splits(base_config(variant), rule)["eval"][:limit_pairs]
    arms = directions["directions"][layer][rule]
    center = directions["means"][layer][rule]
    differences = {}
    with loaded(str(cfg["rules"][rule]["adapter"])) as (tok, model):
        for name in ("shared_loo", "random_0"):
            intervention = (layer, arms[name], center)
            sample_texts = texts(pairs)
            full = forward(model, tok, sample_texts, cfg, intervention=intervention)["margin"]
            cached = forward(
                model, tok, sample_texts, cfg, intervention=intervention,
                use_prefix_cache=True,
            )["margin"]
            differences[name] = float((full - cached).abs().max())
    return differences

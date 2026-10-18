"""Build shared-cue causal directions from normal and label-flipped adapters."""
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.kernel.cue_geometry import FLIPS, NORMAL
from experiments.kernel.opposite import LAYERS, ROOT, acts, pairs


LAYERS_TO_SAVE = (27, 28, 29, 30)
OUT = ROOT / "shared_cue" / "directions.pt"


def _unit(value):
    return F.normalize(value, dim=0)


def build(out=OUT):
    width = 2048
    bundles = {}
    for rule, normal_name in NORMAL.items():
        normal = acts(
            ROOT / f"residuals/scans/3b_seeds/{normal_name}_residual_scan_l24plus/activations.pt",
            rule,
        )
        flip_name = FLIPS[rule][0]
        flipped = acts(ROOT / f"kernel_exploration/flip/{flip_name}/activations.pt", rule)
        for split in ("fit", "select", "eval"):
            assert torch.equal(normal["base"][split], flipped["base"][split]), (rule, split)
        x_normal = torch.cat([pairs(normal, split) for split in ("fit", "select")])
        x_flipped = torch.cat([pairs(flipped, split) for split in ("fit", "select")])
        bundles[rule] = {
            "cue": ((x_normal + x_flipped) / 2).mean(0),
            "answer": ((x_normal - x_flipped) / 2).mean(0),
            "mean": normal["adapter"]["fit"].mean(0),
        }

    generator = torch.Generator().manual_seed(0)
    directions, means = {}, {}
    for layer in LAYERS_TO_SAVE:
        index = LAYERS.index(layer)
        answer_vectors = torch.stack([bundles[rule]["answer"][index] for rule in NORMAL], dim=1)
        answer_basis = torch.linalg.qr(answer_vectors, mode="reduced").Q

        def remove_answers(vector):
            return vector - answer_basis @ (answer_basis.T @ vector)

        layer_directions, layer_means = {}, {}
        for target in NORMAL:
            shared = torch.stack(
                [remove_answers(bundles[rule]["cue"][index]) for rule in NORMAL if rule != target],
                dim=1,
            )
            shared_basis, singular, _ = torch.linalg.svd(shared, full_matrices=False)
            if singular[:3].min() <= 1e-6:
                raise ValueError(f"Shared cue span is rank deficient for {target} at layer {layer}")
            arms = {
                "shared_loo": shared_basis[:, :3].contiguous(),
                "own_cue": _unit(remove_answers(bundles[target]["cue"][index])).unsqueeze(1),
                "own_answer": _unit(bundles[target]["answer"][index]).unsqueeze(1),
            }
            for random_index in range(200):
                gaussian = torch.randn(width, 3, generator=generator)
                projected = remove_answers(gaussian)
                basis, singular, _ = torch.linalg.svd(projected, full_matrices=False)
                if singular[:3].min() <= 1e-6:
                    raise ValueError(f"Random subspace rank deficient at layer {layer}")
                arms[f"random_{random_index}"] = basis[:, :3].contiguous()
            assert all(torch.allclose(q.T @ q, torch.eye(q.shape[1]), atol=1e-4) for q in arms.values())
            layer_directions[target] = {name: value.float() for name, value in arms.items()}
            layer_means[target] = bundles[target]["mean"][index].float()
        directions[layer] = layer_directions
        means[layer] = layer_means

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": directions, "means": means, "layers": list(LAYERS_TO_SAVE)}, out)
    return out


if __name__ == "__main__":
    print(build())

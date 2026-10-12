# Rebuilding full activations
Each rule's `activations.pt` is split into BF16 shards `activations_bf16_l00-11.pt` and `activations_bf16_l12-23.pt`.
Load both shards as `a` and `b`; their `activations[rule][arm][split]` dictionaries have matching metadata.
For every arm/split, rebuild `h` as `torch.cat([a['activations'][rule][arm][split]['h'], b['activations'][rule][arm][split]['h']], dim=1).float()`.
The full `layers` list is `a['layers'] + b['layers']`; margin and greedy-label fields are unchanged.

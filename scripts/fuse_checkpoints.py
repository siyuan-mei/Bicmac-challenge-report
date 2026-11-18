#!/usr/bin/env python3
import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    for key in ("state_dict", "model"):
        if isinstance(checkpoint.get(key), Mapping):
            return dict(checkpoint[key])
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return dict(checkpoint)
    raise TypeError(f"No state dict found in {path}")


def fuse_checkpoints(paths, weights):
    states = [checkpoint_state(path) for path in paths]
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("All checkpoints must have identical ordered keys.")

    fused = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Shape mismatch for {key}.")
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            value = sum(
                tensor.to(torch.float32) * weight
                for tensor, weight in zip(tensors, weights)
            )
            fused[key] = value.to(tensors[0].dtype)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating tensor differs across folds: {key}")
            fused[key] = tensors[0].clone()
    return fused


def main():
    parser = argparse.ArgumentParser(description="Create a weighted model soup.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [Path(path).resolve() for path in args.checkpoints]
    weights = args.weights or [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("Provide one weight per checkpoint.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Checkpoint weights must have a positive sum.")
    weights = [float(weight) / total for weight in weights]

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fuse_checkpoints(paths, weights)}, output)
    metadata = {
        "output": str(output),
        "checkpoints": [str(path) for path in paths],
        "weights": weights,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

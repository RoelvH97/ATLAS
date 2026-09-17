"""Aggregate Table 1 diagnostics by training seed."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("runs/table1.md"))
    args = parser.parse_args()
    groups = {}
    for path in args.reports:
        report = json.loads(path.read_text())
        model = report["model"]
        label = model["type"] + (f" SA{model['n_self_attn']}" if model["type"] == "atlas" else "")
        key = (label, report["gradient_rule"])
        seeds = groups.setdefault(key, {})
        if report["seed"] in seeds:
            raise ValueError(f"Duplicate seed for {key}")
        seeds[report["seed"]] = report["mean"]
    lines = [
        "| Model | Gradient rule | Seeds | PSNR | Effective rank | Tangent fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (model, rule), seeds in sorted(groups.items()):
        values = []
        for metric in ("psnr", "effective_rank", "tangent_fraction"):
            sample = np.array([row[metric] for row in seeds.values()])
            spread = f"{np.std(sample, ddof=1):.3f}" if len(sample) > 1 else "n/a"
            values.append(f"{np.mean(sample):.3f} ({spread})")
        lines.append(f"| {model} | {rule} | {len(seeds)} | " + " | ".join(values) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

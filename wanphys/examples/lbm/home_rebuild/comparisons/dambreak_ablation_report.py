# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare matched Dam-break ablation artifacts and select viable candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_variant(label: str, artifact: Path) -> dict[str, object]:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    steps = sorted(int(step) for step in manifest["metrics"])
    initial_step = steps[0]
    final_step = steps[-1]
    initial = np.load(artifact / "snapshots" / f"step-{initial_step:06d}.npz")
    final = np.load(artifact / "snapshots" / f"step-{final_step:06d}.npz")
    initial_mass = float(
        np.sum(initial["mass"], dtype=np.float64)
        + np.sum(initial["excess"], dtype=np.float64)
    )
    initial_interfaces = int(manifest["metrics"][str(initial_step)]["interface_cells"])
    final_metrics = manifest["metrics"][str(final_step)]
    target_step = int(max(manifest["config"]["sample_steps"]))
    baseline_completed = bool(manifest.get("completed", True))
    return {
        "label": label,
        "artifact": str(artifact),
        "manifest": manifest,
        "initial_fill": initial["fill"],
        "final_fill": final["fill"],
        "completed": baseline_completed and int(manifest.get("completed_step", target_step)) >= target_step,
        "completed_step": int(manifest.get("completed_step", target_step)),
        "target_step": target_step,
        "failure": manifest.get("failure"),
        "steps_per_second": float(manifest["steps_per_second"]),
        "relative_mass_drift": abs(float(final_metrics["total_mass_drift"]))
        / max(initial_mass, 1.0),
        "queued_excess_fraction": abs(float(final_metrics["queued_excess_mass"]))
        / max(initial_mass, 1.0),
        "interface_growth": int(final_metrics["interface_cells"])
        / max(initial_interfaces, 1),
        "front_x": int(final_metrics["front_x"]),
        "volume_drift": float(final_metrics["volume_drift"]),
    }


def build_report(
    variants: list[tuple[str, Path]],
    output_dir: Path,
    *,
    baseline_label: str = "link-wise",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [_load_variant(label, artifact) for label, artifact in variants]
    baseline = next(item for item in loaded if item["label"] == baseline_label)
    baseline_rate = float(baseline["steps_per_second"])
    ranking: list[dict[str, object]] = []
    for item in loaded:
        throughput_ratio = float(item["steps_per_second"]) / baseline_rate
        gates = {
            "completed_target": bool(item["completed"]),
            "mass_drift": float(item["relative_mass_drift"]) <= 5.0e-5,
            "queued_excess": float(item["queued_excess_fraction"]) <= 1.0e-3,
            "interface_growth": float(item["interface_growth"]) <= 4.0,
            "throughput": throughput_ratio >= 0.25,
        }
        ranking.append(
            {
                key: value
                for key, value in item.items()
                if key not in ("manifest", "initial_fill", "final_fill")
            }
            | {
                "throughput_ratio": throughput_ratio,
                "gates": gates,
                "candidate": all(gates.values()),
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(loaded), 3, figsize=(14, 2.7 * len(loaded)), constrained_layout=True
    )
    for row, (item, result) in enumerate(zip(loaded, ranking)):
        center = item["initial_fill"].shape[2] // 2
        for column, key in enumerate(("initial_fill", "final_fill")):
            axes[row, column].imshow(
                item[key][:, :, center].T,
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                interpolation="nearest",
                aspect="auto",
            )
            axes[row, column].set_title(
                "initial" if column == 0 else f"last valid step {item['completed_step']}"
            )
            axes[row, column].set_ylabel(str(item["label"]))
        axes[row, 2].axis("off")
        failure = item["failure"]
        status = "candidate" if result["candidate"] else "rejected"
        failure_text = "none" if failure is None else f"step {failure['step']}: {failure['message']}"
        axes[row, 2].text(
            0.0,
            0.95,
            "\n".join(
                (
                    f"status: {status}",
                    f"failure: {failure_text}",
                    f"throughput: {item['steps_per_second']:.1f} step/s "
                    f"({result['throughput_ratio']:.3f}x)",
                    f"mass drift: {item['relative_mass_drift']:.3e}",
                    f"queued excess: {item['queued_excess_fraction']:.3e}",
                    f"interface growth: {item['interface_growth']:.2f}x",
                    f"front x: {item['front_x']}",
                )
            ),
            va="top",
            family="monospace",
            fontsize=10,
        )
    figure.suptitle("HOME-Free Dam-break staged ablation", fontsize=16)
    overview = output_dir / "dambreak-ablation-overview.png"
    figure.savefig(overview, dpi=170)
    plt.close(figure)

    selected = [item["label"] for item in ranking if item["candidate"]]
    report = {
        "baseline": baseline_label,
        "selected_candidates": selected,
        "ranking": ranking,
        "overview": overview.name,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Dam-break staged ablation",
        "",
        f"Selected candidates: {', '.join(selected) if selected else 'none'}",
        "",
        "| Variant | Last step | Relative speed | Mass drift | Excess | Interface growth | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in ranking:
        lines.append(
            f"| {item['label']} | {item['completed_step']} | "
            f"{item['throughput_ratio']:.3f}x | {item['relative_mass_drift']:.3e} | "
            f"{item['queued_excess_fraction']:.3e} | {item['interface_growth']:.2f}x | "
            f"{'candidate' if item['candidate'] else 'rejected'} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="label=/path/to/artifact; repeat for every variant",
    )
    args = parser.parse_args()
    variants: list[tuple[str, Path]] = []
    for value in args.variant:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("--variant must use label=/path syntax")
        variants.append((label, Path(path)))
    print(json.dumps(build_report(variants, args.output), indent=2))


if __name__ == "__main__":
    main()

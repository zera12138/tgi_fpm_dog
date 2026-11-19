"""Plot the closed-loop A/B/C comparison: forward progress against simulated time."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "perception_out/closed_loop"
STYLE = {"A": ("#1a1a1a", "A truth map (ceiling)"),
         "B": ("#d62728", "B single frame"),
         "C": ("#1f77b4", "C fused history")}


def main():
    scenarios = [p.stem for p in sorted(OUT.glob("*.json"))]
    if not scenarios:
        raise SystemExit("run closed_loop.py first")
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 3.9), squeeze=False)
    for ax, name in zip(axes[0], scenarios):
        data = json.loads((OUT / f"{name}.json").read_text())
        for run in data["runs"]:
            col, label = STYLE[run["condition"]]
            t = [p["t"] for p in run["trace"]]
            x = [p["x"] for p in run["trace"]]
            ax.plot(t, x, color=col, lw=1.6,
                    ls="--" if run["fallen"] else "-", label=label)
            if run["fallen"]:
                ax.axvline(run["fallen_at_s"], color=col, ls=":", lw=1)
                ax.annotate("fall", (run["fallen_at_s"], min(x)), color=col,
                            fontsize=7, rotation=90, va="bottom", ha="right")
        ax.axhline(2.0, color="0.6", lw=0.8, ls="-.")
        ax.text(0.15, 2.03, "stairs start", fontsize=6.5, color="0.45")
        ax.set_title(f"{name}  (pitch {data['pitch_deg']:g}, depth every {data['depth_every']} steps)",
                     fontsize=9)
        ax.set_xlabel("simulated time (s)", fontsize=8)
        ax.set_ylabel("forward x (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "closed_loop.png", dpi=130)
    print(OUT / "closed_loop.png")


if __name__ == "__main__":
    main()

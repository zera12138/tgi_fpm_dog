"""Summarise paired A/B/C runs into outcome distributions and the three error metrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "perception_out/paired"
CONDS = ("A", "B", "C")


def main():
    files = sorted(OUT.glob("*.json"))
    if not files:
        raise SystemExit("no paired results")
    by_scenario = defaultdict(list)
    for f in files:
        data = json.loads(f.read_text())
        by_scenario[data["scenario"]].append(data)

    summary = {}
    for scenario, rows in sorted(by_scenario.items()):
        print(f"\n=== {scenario}  ({len(rows)} conditions) ===")
        print(f"{'':>3} {'pass':>5} {'stall':>6} {'fall':>5} | {'progress/A':>11} {'coverage':>9} "
              f"{'age s':>6} | {'err valid':>10} {'err all':>8} {'unknown':>8} | {'step contacts':>13}")
        per = {}
        for c in CONDS:
            runs = [r["runs"][c] for r in rows]
            counts = Counter(r["outcome"] for r in runs)
            ratios = [r["progress_ratio_vs_A"] for r in runs if r["progress_ratio_vs_A"] is not None]
            cov = [r["mean_coverage"] for r in runs if r["mean_coverage"] is not None]
            age = [r["median_age_s"] for r in runs if r["median_age_s"] is not None]
            ev = [r["err_valid_mm"] for r in runs if r["err_valid_mm"] is not None]
            ea = [r["err_all_patch_mm"] for r in runs if r["err_all_patch_mm"] is not None]
            un = [r["unknown_fraction"] for r in runs if r["unknown_fraction"] is not None]
            stepped = sum(r["stepped_contacts"] for r in runs)
            total_c = sum(r["contacts"] for r in runs)
            print(f"{c:>3} {counts['pass']:>5} {counts['stall']:>6} {counts['fall']:>5} | "
                  f"{np.mean(ratios) if ratios else float('nan'):>11.2f} "
                  f"{np.mean(cov)*100 if cov else float('nan'):>8.1f}% "
                  f"{np.mean(age) if age else float('nan'):>6.2f} | "
                  f"{np.mean(ev) if ev else float('nan'):>9.1f}mm "
                  f"{np.mean(ea) if ea else float('nan'):>7.1f}mm "
                  f"{np.mean(un)*100 if un else float('nan'):>7.1f}% | "
                  f"{stepped}/{total_c}")
            per[c] = {"pass": counts["pass"], "stall": counts["stall"], "fall": counts["fall"],
                      "mean_progress_ratio": round(float(np.mean(ratios)), 3) if ratios else None,
                      "mean_coverage": round(float(np.mean(cov)), 4) if cov else None,
                      "mean_age_s": round(float(np.mean(age)), 3) if age else None,
                      "err_valid_mm": round(float(np.mean(ev)), 1) if ev else None,
                      "err_all_patch_mm": round(float(np.mean(ea)), 1) if ea else None,
                      "unknown_fraction": round(float(np.mean(un)), 4) if un else None,
                      "stepped_contacts": stepped, "contacts": total_c}
        summary[scenario] = {"conditions": len(rows), "per_condition": per}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()

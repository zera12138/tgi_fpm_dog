"""D-group single-factor sweep on the historical-map condition (C).

Each run replays one frozen paired condition with exactly one degradation applied, and reports the
outcome against that condition's truth-map (A) progress. Factors and levels follow the agreed list:

  latency  0 / 50 / 100 / 200 ms of publish delay   (frames are queued, map ages)
  dx,dz    2 / 5 / 10 cm map-relative horizontal or height offset
  pitch    -2 / -1 / +1 / +2 deg extrinsic pitch error
  drop     0.2 / 0.5 / 1.0 s observation gaps (history retained)

  python d_sweep.py --scenario down --factor latency --value 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_eval import PASS_FRACTION, SCEN, make_conditions, run_condition
from perception import CameraSpec
from verify_perception import HERE, Runner, Truth

FACTORS = {
    "latency": ("latency_s", [0.0, 0.05, 0.10, 0.20]),
    "dx": ("dx", [0.0, 0.02, 0.05, 0.10]),
    "dz": ("dz", [0.0, 0.02, 0.05, 0.10]),
    "pitch": ("pitch_deg", [-2.0, -1.0, 0.0, 1.0, 2.0]),
    "drop": ("drop_s", [0.0, 0.2, 0.5, 1.0]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=sorted(SCEN))
    ap.add_argument("--factor", required=True, choices=sorted(FACTORS))
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitch-mount", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--paired-ref", type=Path, default=HERE / "perception_out/paired")
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/d_sweep")
    a = ap.parse_args()

    key, _ = FACTORS[a.factor]
    degrade = {key: a.value} if a.value else {}
    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label=f"assumed pitch{a.pitch_mount:g}", pitch_deg=a.pitch_mount, vfov_deg=a.vfov)
    base = Runner(cam, truth, SCEN[a.scenario][0] - 1, [0.0, 0.0, 0.0])
    conds = make_conditions(a.scenario, a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, init in enumerate(conds):
        ref_path = a.paired_ref / f"{a.scenario}_{a.seed}_{i:03d}.json"
        if not ref_path.exists():
            continue
        a_progress = json.loads(ref_path.read_text())["runs"]["A"]["progress"]
        r = run_condition(base, a.scenario, "C", init, a.depth_every, cam, truth, degrade)
        ratio = r["progress"] / a_progress if a_progress > 0.05 else None
        r["progress_ratio_vs_A"] = None if ratio is None else round(ratio, 3)
        r["outcome"] = ("fall" if r["fallen"] else
                        "pass" if (ratio is not None and ratio >= PASS_FRACTION) else "stall")
        rows.append(r)

    passed = sum(r["outcome"] == "pass" for r in rows)
    stalled = sum(r["outcome"] == "stall" for r in rows)
    fell = sum(r["outcome"] == "fall" for r in rows)
    payload = {"scenario": a.scenario, "factor": a.factor, "value": a.value,
               "conditions": len(rows),
               "pass": passed, "stall": stalled, "fall": fell,
               "mean_progress_ratio": round(float(np.mean([r["progress_ratio_vs_A"] for r in rows
                                                           if r["progress_ratio_vs_A"] is not None])), 3),
               "mean_coverage": round(float(np.mean([r["mean_coverage"] for r in rows])), 4),
               "mean_err_all_mm": round(float(np.mean([r["err_all_patch_mm"] for r in rows])), 1),
               "mean_unknown": round(float(np.mean([r["unknown_fraction"] for r in rows])), 4),
               "runs": rows}
    tag = f"{a.scenario}_{a.factor}_{a.value:+.3f}".replace(".", "p")
    (a.out / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"{a.scenario:<5} {a.factor:<8} {a.value:>+7.3f}  pass {passed:2d} stall {stalled:2d} "
          f"fall {fell:2d}  ratio {payload['mean_progress_ratio']:.2f}  "
          f"cov {payload['mean_coverage']*100:5.1f}%  err {payload['mean_err_all_mm']:6.1f}mm", flush=True)
    base.cam_rig.close()


if __name__ == "__main__":
    main()

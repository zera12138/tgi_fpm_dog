"""Compare fill rules on the paired conditions: plane / low (-0.35 m) / local neighbourhood min."""
import argparse, json
from pathlib import Path
import numpy as np
from paired_eval import PASS_FRACTION, SCEN, make_conditions, run_condition
from perception import CameraSpec
from verify_perception import HERE, Runner, Truth

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, choices=sorted(SCEN))
ap.add_argument("--rule", required=True, choices=["plane", "low", "local", "relative"])
ap.add_argument("--margin", type=float, default=0.35)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--shards", type=int, default=1)
ap.add_argument("--out", type=Path, default=HERE/"perception_out/fill")
a = ap.parse_args()

truth = Truth(HERE/"scene/heightmap.npz")
cam = CameraSpec(label="assumed", pitch_deg=30.0, vfov_deg=66.0)
base = Runner(cam, truth, SCEN[a.scenario][0]-1, [0.0, 0.0, 0.0])
ref = json.loads((a.out/f"ref_{a.scenario}.json").read_text())["conditions"]
rows = []
for i in range(a.shard, a.n, a.shards):
    init = make_conditions(a.scenario, a.n, a.seed)[i]
    r = run_condition(base, a.scenario, "C", init, 3, cam, truth,
                      fill_rule=a.rule, fill_margin=a.margin)
    ap_ = ref[i]["progress"]
    ratio = r["progress"]/ap_ if ap_ > 0.05 else None
    r["progress_ratio_vs_A"] = None if ratio is None else round(ratio, 3)
    r["outcome"] = ("fall" if r["fallen"] else
                    "pass" if (ratio is not None and ratio >= PASS_FRACTION) else "stall")
    rows.append(r)
    print(f"{a.scenario} {a.rule} {i:02d} {r['outcome']} ratio {r['progress_ratio_vs_A']}", flush=True)
a.out.mkdir(parents=True, exist_ok=True)
(a.out/f"{a.scenario}_{a.rule}_{a.shard}.json").write_text(
    json.dumps({"scenario": a.scenario, "rule": a.rule, "margin": a.margin, "runs": rows}, indent=1))
base.cam_rig.close()

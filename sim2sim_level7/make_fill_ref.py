"""Truth-map reference progress for every condition, per scenario (needed by fill_eval)."""
import argparse, json
from pathlib import Path
import numpy as np
from paired_eval import SCEN, make_conditions, run_condition
from perception import CameraSpec
from verify_perception import HERE, Runner, Truth

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True, choices=sorted(SCEN))
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=HERE/"perception_out/fill")
a = ap.parse_args()
truth = Truth(HERE/"scene/heightmap.npz")
cam = CameraSpec(label="assumed", pitch_deg=30.0, vfov_deg=66.0)
base = Runner(cam, truth, SCEN[a.scenario][0]-1, [0.0, 0.0, 0.0])
rows = []
for i, init in enumerate(make_conditions(a.scenario, a.n, a.seed)):
    r = run_condition(base, a.scenario, "A", init, 3, cam, truth)
    rows.append({"index": i, "progress": r["progress"], "fallen": r["fallen"]})
    print(f"{a.scenario} {i:02d} progress {r['progress']:+.2f} fallen {r['fallen']}", flush=True)
a.out.mkdir(parents=True, exist_ok=True)
(a.out/f"ref_{a.scenario}.json").write_text(json.dumps({"scenario": a.scenario, "conditions": rows}, indent=1))
base.cam_rig.close()

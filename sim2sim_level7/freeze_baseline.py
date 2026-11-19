"""Freeze the current baseline: assets, camera config, paired conditions, and result snapshots.

Writes ``baseline_v1/manifest.json`` with hashes of everything that determines a result, copies the
current paired/d-sweep/lead-time outputs, and emits ``reproduce.sh`` so later changes can be diffed
against this exact state.

  python freeze_baseline.py --tag v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def sha(path: Path, limit=None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", type=Path, default=None, help="default: baseline_<tag>/")
    a = ap.parse_args()
    if a.out is None:
        a.out = HERE / f"baseline_{a.tag}"
    a.out.mkdir(parents=True, exist_ok=True)

    assets = {
        "policy_onnx": HERE / "exported/policy.onnx",
        "foot_model_onnx": HERE / "exported/foot_model.onnx",
        "deploy_json": HERE / "exported/deploy.json",
        "scene_xml": HERE / "scene/level7.xml",
        "truth_heightmap": HERE / "scene/heightmap.npz",
        "scene_manifest": HERE / "scene/scene.json",
        "robot_mjcf": Path("/home/odroid/bpx-tgi-fpm/tgi/src/tgi/assets/bpx.xml"),
    }
    manifest = {
        "tag": a.tag,
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "project": str(PROJECT),
        "checkpoint": "upstreamA iter 100000 (frozen, no retraining in this phase)",
        "supersedes": {
            "previous_map_default": "fixed 28.04 x 24.04 m grid (701x601) with fill = start plane",
            "why_changed": "on the stepping stones the start plane equals the stone surface, so every "
                           "unobserved pit read as solid ground: 0/20 pass. Low fill gives 12/20 and "
                           "leaves the stairs at 20/20.",
            "previous_numbers": "perception_out/fill/*_plane_*.json",
        },
        "assets": {k: {"path": str(v), "sha256_16": sha(v), "bytes": v.stat().st_size}
                   for k, v in assets.items() if v.exists()},
        "camera": {
            "model": "assumed pinhole at BPX ZED optical frame -- NOT a calibrated ZED",
            "vfov_deg": 66.0, "hfov_deg": 97.8, "resolution": [424, 240],
            "near_m": 0.25, "far_m": 3.0, "pixel_size_square": True,
            "mount_pitches_tested_deg": [15.0, 30.0],
            "mount_pitch_used": 30.0,
            "note": "physics always uses the XML mount; only the render model carries the pitch",
        },
        "map": {
            "resolution_m": 0.04, "fuse": "highest point per cell", "fuse_stride": 2,
            "depth_period_control_steps": 3,
            "extent": "rolling window 200x160 cells = 8.0 x 6.4 m, re-centred on the robot",
            "fill_rule": "start support plane - 0.35 m; valid=False, never fused",
            "fill_rules_compared": {"plane": "baseline", "low": "start plane - 0.35 m  [DEFAULT]",
                                    "local": "neighbourhood minimum - 0.35 m, falls back to low"},
            "fill_evidence": "perception_out/fill: stones plane 0/20, low 12/20, local 15/20; "
                             "stairs unchanged at 20/20 for all three",
        },
        "paired_conditions": {
            "generator": "paired_eval.make_conditions(scenario, n=20, seed=0)",
            "scenarios": {"up": {"spawn": 1, "seconds": 12.0, "vx": [0.26, 0.32]},
                          "down": {"spawn": 2, "seconds": 10.0, "vx": [0.22, 0.28]}},
            "jitter": {"dx_m": [-0.45, 0.45], "dy_m": [-0.30, 0.30], "dyaw_deg": [-8.0, 8.0]},
            "pass_rule": "progress >= 0.85 x the truth-map run's own progress",
        },
        "result_snapshots": {},
    }

    for name in ("paired", "d_sweep", "lead_time", "lead_descent", "fill", "stones", "stones_plane"):
        src = HERE / "perception_out" / name
        if not src.exists():
            continue
        dst = a.out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        files = sorted(p for p in dst.rglob("*") if p.is_file())
        manifest["result_snapshots"][name] = {
            "files": len(files),
            "sha256_16": sha(dst / files[0].name) if len(files) == 1 else None,
            "copied_from": str(src),
        }

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    reproduce = f"""#!/usr/bin/env bash
# Full reproduction of baseline {a.tag}: map geometry, paired A/B/C, D-group sweeps,
# fill-rule comparison, lead-time analysis, and the async rolling-map closed loop.
set -eu
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl
PY=/home/odroid/anaconda3/envs/rl_lab/bin/python
OUT=./perception_out

echo "== 1/7 geometry and occlusion =="
$PY check_geometry.py --spawn 1 --pitch 30 > /dev/null
$PY check_geometry.py --spawn 2 --pitch 30 > /dev/null
$PY check_occlusion.py --pitch 80 > /dev/null

echo "== 2/7 paired A/B/C (40 conditions, fixed full grid) =="
for sc in down up; do
  for sh in 0 1 2; do
    $PY paired_eval.py --scenario $sc --n 20 --shard $sh --shards 3 --out $OUT/paired &
  done
done
wait
$PY summarise_paired.py
$PY plot_paired.py

echo "== 3/7 truth reference for the fill comparison =="
for sc in down up stones; do $PY make_fill_ref.py --scenario $sc & done
wait

echo "== 4/7 fill rules: plane / low / local on three terrains =="
{{ for sc in down up stones; do for r in plane low local; do for sh in 0 1; do
    echo "$sc $r $sh"; done; done; done; }} \
  | xargs -P 6 -n 3 bash -c "$PY fill_eval.py --scenario \$0 --rule \$1 --shard \$2 --shards 2"

echo "== 5/7 D-group single factors =="
bash run_sweep.sh
bash run_sweep2.sh
bash run_sweep3.sh
bash run_sweep4.sh
$PY summarise_sweep.py

echo "== 6/7 lead time =="
$PY lead_descent.py --n 10
for c in baseline lat500 lat1500 gap1000 gap2000; do $PY lead_time.py --case $c --n 10 & done
wait

echo "== 7/7 async closed loop (rolling 200x160 window, low fill) =="
# Serial on purpose: this stage reports control p95/p99 and overrun counts, and concurrent
# processes contend for cores and inflate the tail. Never parallelise timing measurements.
for sc in down up stones; do
  $PY closed_loop_async.py --scenario $sc --n 10 --out $OUT/async
done

echo "done. Compare against baseline_{a.tag}/ with the same directory names."
"""
    (a.out / "reproduce.sh").write_text(reproduce)
    (a.out / "reproduce.sh").chmod(0o755)
    print(f"frozen -> {a.out}")
    for k, v in manifest["assets"].items():
        print(f"  {k:<20} {v['sha256_16']}  {v['bytes']:>9} B")
    print(f"  snapshots: {list(manifest['result_snapshots'])}")
    print(f"  rerun: {a.out/'reproduce.sh'}")


if __name__ == "__main__":
    main()

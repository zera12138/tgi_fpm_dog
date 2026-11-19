"""Per-component latency of the perception loop, measured over a real rollout.

The 20 ms control period is the budget: a mean under budget means nothing if the tail overruns.
Reports p50/p95/p99/max and how often each stage (and the whole cycle) misses the deadline.

Usage:
  python benchmark_perception.py --pitch 30 --seconds 8 --spawn 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, DepthCamera, HeightMap
from verify_perception import HERE, Runner, Truth  # noqa: F401


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--spawn", type=int, default=1, choices=range(1, 6))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    ap.add_argument("--depth-every", type=int, default=1,
                    help="run the depth pipeline only every N control steps (deployment decimation)")
    ap.add_argument("--out", type=Path, default=HERE / "perception_out")
    a = ap.parse_args()

    import verify_perception as vp
    truth = Truth(HERE / "scene/heightmap.npz")
    vp.TRUTH = truth
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)
    run = Runner(cam, truth, a.spawn - 1, a.command)
    dt = run.ctrl.dt
    hist = HeightMap(truth.origin, truth.h.shape, truth.res, run.plane, stride=2)
    stages = {k: [] for k in ("policy", "physics", "capture", "backproject", "fuse", "cycle")}

    for step in range(round(a.seconds / dt)):
        t0 = time.perf_counter()
        d = run.data
        from tgi.deploy.controller import quat_to_mat

        R = quat_to_mat(d.qpos[3:7])
        vb = R.T @ d.qvel[:3]
        t = time.perf_counter()
        target = run.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                               d.qpos[run.qidx].copy(), d.qvel[run.vidx].copy(), run.command)
        stages["policy"].append(time.perf_counter() - t)
        t = time.perf_counter()
        for _ in range(run.substeps):
            torque = np.clip(run.ctrl.kp * (target - d.qpos[run.qidx]) - run.ctrl.kd * d.qvel[run.vidx],
                             -run.ctrl.eff, run.ctrl.eff)
            d.ctrl[run.actuators] = torque
            mujoco.mj_step(run.model, d)
        run.sync_view()
        stages["physics"].append(time.perf_counter() - t)
        if step % a.depth_every:
            stages["cycle"].append(time.perf_counter() - t0)
            continue
        t = time.perf_counter()
        terrain, with_robot = run.cam_rig.capture(run.vdata)
        stages["capture"].append(time.perf_counter() - t)
        t = time.perf_counter()
        pts, keep = run.cam_rig.points(run.vdata, terrain)
        keep &= ~(with_robot < terrain - 0.02)
        stages["backproject"].append(time.perf_counter() - t)
        t = time.perf_counter()
        hist.fuse(pts, keep, step)
        stages["fuse"].append(time.perf_counter() - t)
        stages["cycle"].append(time.perf_counter() - t0)

    period_ms = dt * 1000
    report = {"pitch_deg": a.pitch, "vfov_deg": a.vfov, "control_period_ms": period_ms,
              "depth_every": a.depth_every, "depth_hz": round(50.0 / a.depth_every, 1),
              "steps": len(stages["cycle"]), "note": "single CPU thread, EGL offscreen 424x240",
              "stages": {}}
    for k, v in stages.items():
        arr = np.asarray(v) * 1000
        report["stages"][k] = {"p50_ms": round(pct(arr, 50), 2), "p95_ms": round(pct(arr, 95), 2),
                               "p99_ms": round(pct(arr, 99), 2), "max_ms": round(float(arr.max()), 2),
                               "mean_ms": round(float(arr.mean()), 2),
                               "over_period": int((arr > period_ms).sum())}
    cyc = np.asarray(stages["cycle"]) * 1000
    report["deadline"] = {"period_ms": period_ms, "misses": int((cyc > period_ms).sum()),
                          "miss_rate": round(float((cyc > period_ms).mean()), 3),
                          "p99_ms": round(pct(cyc, 99), 2),
                          "budget_headroom_p99_ms": round(period_ms - pct(cyc, 99), 2)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"benchmark_pitch{a.pitch:g}_depth{a.depth_every}x.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    run.cam_rig.close()


if __name__ == "__main__":
    main()

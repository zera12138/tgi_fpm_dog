"""Real-time closed loop with asynchronous perception.

Same paired conditions as the frozen baseline, but depth runs on a worker thread and the policy
reads published snapshots at a paced 50 Hz. Measures whether the control step fits its 20 ms budget
and how old the map is when the policy reads it.

  python closed_loop_async.py --scenario down --n 20 --shard 0 --shards 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from async_perception import AsyncPerception
from paired_eval import PASS_FRACTION, SCEN, make_conditions
from perception import CameraSpec, HeightMap, support_plane
from verify_perception import HERE, Runner, Truth

import sys

sys.path.insert(0, str(HERE))
from tgi.deploy.controller import quat_to_mat  # noqa: E402

WINDOW = (200, 160)   # rolling 8.0 x 6.4 m window, 4 cm cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=sorted(SCEN))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--no-pace", action="store_true", help="run as fast as possible instead of 50 Hz")
    ap.add_argument("--ref", type=Path, default=HERE / "baseline_v2/paired")
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/async")
    a = ap.parse_args()

    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)
    base = Runner(cam, truth, SCEN[a.scenario][0] - 1, [0.0, 0.0, 0.0])
    conds = make_conditions(a.scenario, a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(a.shard, a.n, a.shards):
        init = conds[i]
        base.reset(SCEN[a.scenario][0] - 1)
        d = base.data
        d.qpos[0] += init["dx"]
        d.qpos[1] += init["dy"]
        yaw = math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                         1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        yaw += math.radians(init["dyaw"])
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(base.model, d)
        base.sync_view()
        plane = support_plane(base.cam_rig, base.vdata, cam)

        perc = AsyncPerception(HERE / "scene/level7.xml", cam, truth.origin, truth.h.shape,
                               truth.res, plane, stride=2, window=WINDOW,
                               center=(float(d.qpos[0]), float(d.qpos[1]))).start()
        base.ctrl.height_fn = lambda x, y: 0.0
        deadline = time.monotonic() + 5.0
        while perc.latest() is None and time.monotonic() < deadline:
            perc.submit(d.qpos, d.qvel, 0.0)
            time.sleep(0.01)
        base.ctrl.height_fn = perc.latest().height_at
        base.ctrl.reset()
        command = np.array([init["vx"], 0.0, 0.0])
        x_start = float(d.qpos[0])
        direction = -1.0 if a.scenario == "down" else 1.0
        fell = False
        dt = base.ctrl.dt
        steps = round(SCEN[a.scenario][1] / dt)

        compute, period, latency, age, versions = [], [], [], [], []
        queue_wait, submit_to_publish, read_to_use = [], [], []
        submitted = {}
        sim_t = 0.0
        for step in range(steps):
            t0 = time.perf_counter()
            if step % a.depth_every == 0:
                perc.submit(d.qpos, d.qvel, sim_t)
                submitted[round(sim_t, 6)] = time.perf_counter()
            snap = perc.latest()
            read_wall = time.perf_counter()
            if snap is not None:
                base.ctrl.height_fn = snap.height_at
                versions.append(snap.version)

            R = quat_to_mat(d.qpos[3:7])
            vb = R.T @ d.qvel[:3]
            target = base.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                                    d.qpos[base.qidx].copy(), d.qvel[base.vidx].copy(), command)
            for _ in range(base.substeps):
                torque = np.clip(base.ctrl.kp * (target - d.qpos[base.qidx]) - base.ctrl.kd * d.qvel[base.vidx],
                                 -base.ctrl.eff, base.ctrl.eff)
                d.ctrl[base.actuators] = torque
                mujoco.mj_step(base.model, d)
            base.sync_view()
            sim_t += dt

            use_wall = time.perf_counter()
            if snap is not None:
                latency.append(read_wall - snap.capture_wall)          # read vs render start
                age.append(sim_t - snap.state_t)                        # sim-time staleness
                t_sub = submitted.get(round(snap.state_t, 6))
                if t_sub is not None:
                    queue_wait.append(snap.capture_wall - t_sub)        # submit -> render start
                    submit_to_publish.append(snap.publish_wall - t_sub)  # submit -> readable
                read_to_use.append(use_wall - read_wall)
            compute.append(use_wall - t0)   # full step: read + policy + physics
            if not a.no_pace:
                rest = dt - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
            period.append(time.perf_counter() - t0)

            R2 = quat_to_mat(d.qpos[3:7])
            if (not np.isfinite(d.qpos).all()) or R2[2, 2] < math.cos(0.7) \
                    or d.qpos[2] - truth(*d.qpos[:2]) < 0.16:
                fell = True
                break
        perc.stop()

        progress = (float(d.qpos[0]) - x_start) * direction
        ref_path = a.ref / f"{a.scenario}_{a.seed}_{i:03d}.json"
        if ref_path.exists():
            a_prog = json.loads(ref_path.read_text())["runs"]["A"]["progress"]
        else:   # stones has no paired A run; the fill reference holds the same truth runs
            fill_ref = HERE / "perception_out/fill" / f"ref_{a.scenario}.json"
            a_prog = json.loads(fill_ref.read_text())["conditions"][i]["progress"]
        ratio = round(progress / a_prog, 3) if a_prog > 0.05 else None
        rows.append({"index": i, "scenario": a.scenario, "fallen": fell,
                     "progress": round(progress, 3), "progress_ratio_vs_A": ratio,
                     "outcome": ("fall" if fell else
                                 "pass" if (ratio is not None and ratio >= PASS_FRACTION) else "stall"),
                     "final_x": round(float(d.qpos[0]), 3),
                     "compute_ms_p50": round(float(np.percentile(compute, 50)) * 1000, 2),
                     "compute_ms_p95": round(float(np.percentile(compute, 95)) * 1000, 2),
                     "compute_ms_p99": round(float(np.percentile(compute, 99)) * 1000, 2),
                     "compute_over_20ms": int(sum(c > dt for c in compute)),
                     "period_ms_p50": round(float(np.percentile(period, 50)) * 1000, 2),
                     "period_ms_p99": round(float(np.percentile(period, 99)) * 1000, 2),
                     "map_latency_ms_p50": round(float(np.percentile(latency, 50)) * 1000, 1) if latency else None,
                     "map_latency_ms_p95": round(float(np.percentile(latency, 95)) * 1000, 1) if latency else None,
                     "queue_wait_ms_p50": round(float(np.percentile(queue_wait, 50)) * 1000, 1) if queue_wait else None,
                     "submit_to_publish_ms_p50": round(float(np.percentile(submit_to_publish, 50)) * 1000, 1) if submit_to_publish else None,
                     "submit_to_publish_ms_p95": round(float(np.percentile(submit_to_publish, 95)) * 1000, 1) if submit_to_publish else None,
                     "read_to_use_ms_p95": round(float(np.percentile(read_to_use, 95)) * 1000, 2) if read_to_use else None,
                     "map_age_s_p50": round(float(np.percentile(age, 50)), 3) if age else None,
                     "map_age_s_p95": round(float(np.percentile(age, 95)), 3) if age else None,
                     "map_versions": len(set(versions)),
                     "steps": len(compute)})
        print(json.dumps(rows[-1]), flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.scenario}_{a.shard}.json").write_text(json.dumps({"scenario": a.scenario, "runs": rows}, indent=1))
    base.cam_rig.close()


if __name__ == "__main__":
    main()

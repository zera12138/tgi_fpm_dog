"""A/B/C closed loop: does the policy actually walk on what the camera gives it?

  A  truth heightmap          (ceiling: what the trained policy can do)
  B  single-frame depth map   (what this frame sees, nothing else)
  C  fused history map        (empty at t=0, grows as the robot observes)

Configuration agreed for this round: 30 deg mount, fixed start support plane for unknown cells,
depth captured every 3 control steps (60 ms of simulated time), published immediately and read on
the next control step. Scheduling is in **simulated** time, so a slow machine cannot turn a
perception question into a timeout question. Real-time behaviour is measured separately.

  python closed_loop.py --scenario up
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, HeightMap
from verify_perception import HERE, Runner, Truth, foot_metrics

FEET = ["FL", "FR", "HL", "HR"]
CONDITIONS = ("A", "B", "C")
SCENARIOS = {
    # name: (spawn, [(seconds, command)])
    "up": (1, [(12.0, [0.3, 0.0, 0.0])]),
    "down": (2, [(10.0, [0.25, 0.0, 0.0])]),
    "updown_path": (1, [(10.0, [0.3, 0.0, 0.0]), (6.0, [-0.2, 0.0, 0.0])]),
}


def trajectory_identity(records):
    """Same endpoint is not the same trajectory; compare every logged state."""
    out = {}
    for i in range(1, len(records)):
        a, b = records[0], records[i]
        n = min(len(a["states"]), len(b["states"]))
        d = np.abs(np.asarray(a["states"][:n]) - np.asarray(b["states"][:n]))
        out[records[i]["run"]] = {"compared_steps": int(n),
                                  "max_abs_qpos_diff": float(d.max()) if n else None,
                                  "identical": bool(n and d.max() == 0)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="up", choices=sorted(SCENARIOS))
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--depth-every", type=int, default=3, help="capture every N control steps")
    ap.add_argument("--fill", default="plane", choices=["plane", "nearest"])
    ap.add_argument("--nearest-limit", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/closed_loop")
    a = ap.parse_args()

    spawn, phases = SCENARIOS[a.scenario]
    total = sum(p[0] for p in phases)
    truth = Truth(HERE / "scene/heightmap.npz")
    import verify_perception as vp

    vp.TRUTH = truth
    cam = CameraSpec(label=f"assumed pitch{a.pitch:g}", pitch_deg=a.pitch, vfov_deg=a.vfov)
    runs, records = [], []

    for cond in CONDITIONS:
        run = Runner(cam, truth, spawn - 1, phases[0][1])
        fallback = 0.0 if a.fill == "nearest" else None
        hist = HeightMap(truth.origin, truth.h.shape, truth.res, run.plane,
                         nearest_limit=None if fallback is None else a.nearest_limit, stride=2)
        single = HeightMap(truth.origin, truth.h.shape, truth.res, run.plane, stride=2)
        active = hist if cond == "C" else single
        if cond != "A":
            run.ctrl.height_fn = active.height_at
        run.ctrl.reset()

        dt = run.ctrl.dt
        n = round(total / dt)
        phase_ends = np.cumsum([p[0] for p in phases]) / dt
        prev_contact = np.zeros(4, bool)
        blind = np.zeros(4, int)
        contacts = np.zeros(4, int)
        fell, fell_at = False, None
        max_x = -1e9
        states, trace = [], []
        last_capture = -10
        query_err, patch_cov, patch_age = [], [], []

        for step in range(n):
            phase = int(np.searchsorted(phase_ends, step, side="right"))
            run.command = np.asarray(phases[min(phase, len(phases) - 1)][1], float)

            if step % a.depth_every == 0:  # capture before the policy reads, at this instant
                terrain, with_robot = run.cam_rig.capture(run.vdata)
                pts, keep = run.cam_rig.points(run.vdata, terrain)
                keep &= ~(with_robot < terrain - 0.02)
                if cond == "B":
                    single.clear()
                active.fuse(pts, keep, step)
                last_capture = step

            d = run.data
            from tgi.deploy.controller import quat_to_mat

            R = quat_to_mat(d.qpos[3:7])
            vb = R.T @ d.qvel[:3]
            target = run.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                                   d.qpos[run.qidx].copy(), d.qvel[run.vidx].copy(), run.command)
            for _ in range(run.substeps):
                torque = np.clip(run.ctrl.kp * (target - d.qpos[run.qidx]) - run.ctrl.kd * d.qvel[run.vidx],
                                 -run.ctrl.eff, run.ctrl.eff)
                d.ctrl[run.actuators] = torque
                mujoco.mj_step(run.model, d)
            run.sync_view()
            states.append(d.qpos.copy())
            max_x = max(max_x, float(d.qpos[0]))

            # what the policy actually read this step (the active map, fill included)
            if cond != "A":
                fs = run.ctrl.vis["pts"]
                flat = fs.reshape(-1, 2)
                got = np.array([active.height_at(x, y) for x, y in flat])
                want = truth.at(flat[:, 0], flat[:, 1])
                query_err.append(float(np.mean(np.abs(got - want))))
                h_obs, seen, last = active.sample(flat[:, 0], flat[:, 1])
                patch_cov.append(float(seen.mean()))
                if seen.any():
                    patch_age.append(float(np.median((step - last[seen]) * dt)))

            hit = run.contacts()
            for f in range(4):
                if hit[f] and not prev_contact[f]:
                    contacts[f] += 1
                    fp = d.xpos[run.model.body(f"{FEET[f].lower()}_toe_link").id][:2]
                    if cond != "A" and not active.sample(fp[None, 0], fp[None, 1])[1][0]:
                        blind[f] += 1
            prev_contact = hit

            if not np.isfinite(d.qpos).all():
                fell, fell_at = True, step * dt
                break
            R2 = quat_to_mat(d.qpos[3:7])
            if R2[2, 2] < math.cos(0.7) or d.qpos[2] - truth(*d.qpos[:2]) < 0.16:
                fell, fell_at = True, step * dt
                break
            if (step + 1) % 25 == 0:
                trace.append({"t": round((step + 1) * dt, 3), "x": round(float(d.qpos[0]), 3),
                              "y": round(float(d.qpos[1]), 3)})

        steps_run = len(states)
        row = {"condition": cond, "scenario": a.scenario, "pitch_deg": a.pitch,
               "depth_every": a.depth_every, "fill": a.fill,
               "support_plane_m": round(run.plane, 4),
               "fallen": bool(fell), "fallen_at_s": None if fell_at is None else round(fell_at, 2),
               "final_x": round(float(run.data.qpos[0]), 3),
               "max_x": round(max_x, 3),
               "sim_seconds": round(steps_run * dt, 2),
               "mean_patch_coverage": None if not patch_cov else round(float(np.mean(patch_cov)), 4),
               "median_patch_age_s": None if not patch_age else round(float(np.median(patch_age)), 3),
               "mean_policy_height_err_mm": None if not query_err else round(float(np.mean(query_err)) * 1000, 1),
               "blind_contacts": ({f: int(blind[i]) for i, f in enumerate(FEET)} if cond != "A" else None),
               "contacts": {f: int(contacts[i]) for i, f in enumerate(FEET)},
               "trace": trace}
        runs.append(row)
        records.append({"run": f"{cond}", "states": [s.tolist() for s in states[::50]]})
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
        run.cam_rig.close()

    summary = {"scenario": a.scenario, "pitch_deg": a.pitch, "depth_every": a.depth_every,
               "fill": a.fill, "runs": runs}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.scenario}.json").write_text(json.dumps(summary, indent=2))
    print("\n--- closed loop ---")
    for r in runs:
        print(f"{r['condition']}  fallen={r['fallen']!s:<5} at={r['fallen_at_s']}  "
              f"final_x={r['final_x']:+.2f}  max_x={r['max_x']:+.2f}  "
              f"cov={r['mean_patch_coverage']}  age={r['median_patch_age_s']}  "
              f"policy_err={r['mean_policy_height_err_mm']}mm")


if __name__ == "__main__":
    main()

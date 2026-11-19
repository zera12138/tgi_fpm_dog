"""Paired A/B/C evaluation over a fixed set of test conditions.

Every condition fixes one initial state (position, heading, command) and is replayed by A, B and C
from that exact state, so the only thing that changes is what the policy reads. Conditions shard
cleanly across processes: ``--shard i --shards k`` runs condition indices ``i, i+k, ...``.

Outcomes are classified per condition:
  pass   -- not fallen and progress >= PASS_FRACTION of the truth-map run's own progress
  stall  -- not fallen but short of that
  fall   -- termination criterion triggered

  python paired_eval.py --scenario down --n 10 --shard 0 --shards 4 --out out/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, HeightMap
from verify_perception import HERE, Runner, Truth

FEET = ["FL", "FR", "HL", "HR"]
CONDITIONS = ("A", "B", "C")
PASS_FRACTION = 0.85
SCEN = {"up": (1, 12.0, +1.0, (0.26, 0.32)), "down": (2, 10.0, -1.0, (0.22, 0.28)),
        "stones": (4, 24.0, +1.0, (0.22, 0.28))}


def make_conditions(scenario, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({"dx": float(rng.uniform(-0.45, 0.45)),
                    "dy": float(rng.uniform(-0.30, 0.30)),
                    "dyaw": float(rng.uniform(-8.0, 8.0)),
                    "vx": float(rng.uniform(*SCEN[scenario][3]))})
    return out


def _drop_mask(n, dt, gap_s, every_s=3.0, start_s=2.0):
    """Steps whose depth capture is skipped: a ``gap_s`` hole every ``every_s`` seconds."""
    m = np.zeros(n, bool)
    for t0 in np.arange(start_s, n * dt - 1e-9, every_s):
        i0 = int(round(t0 / dt))
        m[i0:i0 + int(round(gap_s / dt))] = True
    return m


def _perturb(pts, cam_pos, cam_rot, degrade):
    """Apply a map-relative error to the fused point cloud (models pose / extrinsic error)."""
    shape = pts.shape
    p = pts.reshape(-1, 3).copy()
    ang = degrade.get("pitch_deg", 0.0)
    if ang:
        axis = cam_rot[:, 1].copy()                       # camera lateral axis, world frame
        axis /= np.linalg.norm(axis)
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        rel = p - cam_pos
        rel = rel * c + np.cross(axis, rel) * s + np.outer(rel @ axis, axis) * (1.0 - c)
        p = rel + cam_pos
    p[:, 0] += degrade.get("dx", 0.0)
    p[:, 1] += degrade.get("dy", 0.0)
    p[:, 2] += degrade.get("dz", 0.0)
    return p.reshape(shape)


def run_condition(base: Runner, scenario, cond, init, depth_every, cam, truth, degrade=None,
                  fill_rule="plane", fill_margin=0.35, stand_height=0.43):
    spawn, seconds, direction, _ = SCEN[scenario]
    base.reset(spawn - 1)
    d = base.data
    d.qpos[0] += init["dx"]
    d.qpos[1] += init["dy"]
    yaw = math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                     1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
    yaw += math.radians(init["dyaw"])
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(base.model, d)
    base.sync_view()
    from perception import support_plane
    base.plane = support_plane(base.cam_rig, base.vdata, cam)  # from the first depth frame only

    hist = HeightMap(truth.origin, truth.h.shape, truth.res, base.plane, stride=2,
                     fill_mode=fill_rule, fill_margin=fill_margin, stand_height=stand_height)
    single = HeightMap(truth.origin, truth.h.shape, truth.res, base.plane, stride=2,
                       fill_mode=fill_rule, fill_margin=fill_margin, stand_height=stand_height)
    active = {"A": None, "B": single, "C": hist}[cond]
    base.ctrl.height_fn = truth if cond == "A" else active.height_at
    base.ctrl.reset()
    command = np.array([init["vx"], 0.0, 0.0])  # body frame: spawn 2 already faces down

    from tgi.deploy.controller import quat_to_mat

    dt = base.ctrl.dt
    n = round(seconds / dt)
    x0 = float(d.qpos[0])
    prev_contact = np.zeros(4, bool)
    blind = np.zeros(4, int)
    contacts = np.zeros(4, int)
    stepped_contacts = 0
    fell, fell_at = False, None
    valid_err, all_err, unknown = [], [], []
    cov, age = [], []
    states = []

    degrade = degrade or {}
    latency_steps = int(round(degrade.get("latency_s", 0.0) / dt))
    drop = _drop_mask(n, dt, degrade["drop_s"]) if degrade.get("drop_s") else np.zeros(n, bool)
    pending = []

    for step in range(n):
        if step % depth_every == 0 and active is not None and not drop[step]:
            terrain, with_robot = base.cam_rig.capture(base.vdata)
            pts, keep = base.cam_rig.points(base.vdata, terrain)
            keep &= ~(with_robot < terrain - 0.02)
            if degrade:
                pts = _perturb(pts, base.vdata.cam_xpos[base.cam_rig.cid],
                               base.vdata.cam_xmat[base.cam_rig.cid].reshape(3, 3), degrade)
            pending.append((step, pts, keep))
        # publish: a frame becomes visible only once the configured latency has elapsed
        while pending and step - pending[0][0] >= latency_steps:
            _, pts, keep = pending.pop(0)
            if cond == "B":
                single.clear()
            active.fuse(pts, keep, step)

        if fill_rule == "relative" and active is not None:   # fill follows the current torso
            active.set_torso(float(d.xpos[base.model.body("torso").id][2]))
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
        states.append(d.qpos.copy())

        if active is not None:
            flat = base.ctrl.vis["pts"].reshape(-1, 2)
            got = np.array([active.height_at(x, y) for x, y in flat])
            want = truth.at(flat[:, 0], flat[:, 1])
            _, seen, last = active.sample(flat[:, 0], flat[:, 1])
            diff = np.abs(got - want)
            if seen.any():
                valid_err.append(float(diff[seen].mean()))
                age.append(float(np.median((step - last[seen]) * dt)))
            all_err.append(float(diff.mean()))
            unknown.append(float(1.0 - seen.mean()))
            cov.append(float(seen.mean()))

        hit = base.contacts()
        for f in range(4):
            if hit[f] and not prev_contact[f]:
                contacts[f] += 1
                fp = d.xpos[base.model.body(f"{FEET[f].lower()}_toe_link").id]
                if active is not None and not active.sample(fp[None, 0], fp[None, 1])[1][0]:
                    blind[f] += 1
                if abs(truth(float(fp[0]), float(fp[1]))) > 0.05:
                    stepped_contacts += 1
        prev_contact = hit

        R2 = quat_to_mat(d.qpos[3:7])
        if (not np.isfinite(d.qpos).all()) or R2[2, 2] < math.cos(0.7) \
                or d.qpos[2] - truth(*d.qpos[:2]) < 0.16:
            fell, fell_at = True, round(step * dt, 2)
            break

    progress = (float(d.qpos[0]) - x0) * direction
    return {"scenario": scenario, "condition": cond, "init": init,
            "fallen": fell, "fallen_at_s": fell_at,
            "progress": round(progress, 3),
            "final_x": round(float(d.qpos[0]), 3), "final_y": round(float(d.qpos[1]), 3),
            "sim_seconds": round(len(states) * dt, 2),
            "mean_coverage": round(float(np.mean(cov)), 4) if cov else None,
            "median_age_s": round(float(np.median(age)), 3) if age else None,
            "err_valid_mm": round(float(np.mean(valid_err)) * 1000, 1) if valid_err else None,
            "err_all_patch_mm": round(float(np.mean(all_err)) * 1000, 1) if all_err else None,
            "unknown_fraction": round(float(np.mean(unknown)), 4) if unknown else None,
            "blind_contacts": int(blind.sum()) if active is not None else None,
            "contacts": int(contacts.sum()),
            "stepped_contacts": int(stepped_contacts),
            "_states50": [s.tolist() for s in states[::50]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="down", choices=sorted(SCEN))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/paired")
    a = ap.parse_args()

    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label=f"assumed pitch{a.pitch:g}", pitch_deg=a.pitch, vfov_deg=a.vfov)
    base = Runner(cam, truth, SCEN[a.scenario][0] - 1, [0.0, 0.0, 0.0])
    conds = make_conditions(a.scenario, a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    for i in range(a.shard, a.n, a.shards):
        rows = {}
        for cond in CONDITIONS:
            rows[cond] = run_condition(base, a.scenario, cond, conds[i], a.depth_every, cam, truth)
        ref = rows["A"]["progress"]
        for cond, r in rows.items():
            r["progress_ratio_vs_A"] = round(r["progress"] / ref, 3) if ref > 0.05 else None
            if r["fallen"]:
                r["outcome"] = "fall"
            elif r["progress_ratio_vs_A"] is not None and r["progress_ratio_vs_A"] >= PASS_FRACTION:
                r["outcome"] = "pass"
            else:
                r["outcome"] = "stall"
        payload = {"scenario": a.scenario, "index": i, "pitch_deg": a.pitch,
                   "depth_every": a.depth_every, "runs": rows}
        (a.out / f"{a.scenario}_{a.seed}_{i:03d}.json").write_text(json.dumps(payload, indent=1))
        brief = {c: (rows[c]["outcome"], rows[c]["progress"], rows[c]["mean_coverage"]) for c in CONDITIONS}
        print(f"[{a.scenario} {i:02d}] {brief}", flush=True)
    base.cam_rig.close()


if __name__ == "__main__":
    main()

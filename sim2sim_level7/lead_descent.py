"""Per-contact information margin while descending the level-7 stairs.

For every foot touchdown this records what the policy could have known about *the ground under that
foot*, not about the step in general:

  t_info     first time the map holds a valid height within ``--radius`` of the contact point
  t_contact  the touchdown frame
  lead       t_contact - t_info   (None = never observed before touchdown, counted separately)

The trajectory is driven by the truth map so every run shares one descent; the map is rebuilt from
rendered depth along it.

  python lead_descent.py --n 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, HeightMap, support_plane
from verify_perception import HERE, Runner, Truth

FEET = ["FL", "FR", "HL", "HR"]
RISE = 0.1275


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--radius", type=float, default=0.10, help="contact neighbourhood radius (m)")
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/lead_descent")
    a = ap.parse_args()

    from paired_eval import make_conditions

    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)
    base = Runner(cam, truth, a.spawn - 1, [0.0, 0.0, 0.0])
    conds = make_conditions("down", a.n, a.seed)
    off = np.arange(-int(a.radius / truth.res) - 1, int(a.radius / truth.res) + 2) * truth.res
    ox, oy = np.meshgrid(off, off)
    disk = ox ** 2 + oy ** 2 <= a.radius ** 2

    events = []
    for c_i, init in enumerate(conds):
        base.reset(a.spawn - 1)
        d = base.data
        d.qpos[0] += init["dx"]
        d.qpos[1] += init["dy"]
        d.qpos[3:7] = [np.cos(math.pi / 2 + init["dyaw"] * math.pi / 360), 0, 0,
                       np.sin(math.pi / 2 + init["dyaw"] * math.pi / 360)]
        mujoco.mj_forward(base.model, d)
        base.sync_view()
        base.plane = support_plane(base.cam_rig, base.vdata, cam)

        hist = HeightMap(truth.origin, truth.h.shape, truth.res, base.plane, stride=2)
        base.ctrl.height_fn = truth   # trajectory fixed by truth, map is only measured
        base.ctrl.reset()
        command = np.array([a.vx, 0.0, 0.0])
        dt = base.ctrl.dt
        n = round(a.seconds / dt)
        first_obs = np.full(truth.h.shape, -1, np.int32)
        prev = np.zeros(4, bool)
        idx = lambda h: int(round(h / RISE))  # noqa: E731

        for step in range(n):
            if step % a.depth_every == 0:
                terrain, with_robot = base.cam_rig.capture(base.vdata)
                pts, keep = base.cam_rig.points(base.vdata, terrain)
                keep &= ~(with_robot < terrain - 0.02)
                before = hist.valid.copy()
                hist.fuse(pts, keep, step)
                first_obs[hist.valid & ~before] = step

            R = __import__("tgi.deploy.controller", fromlist=["quat_to_mat"]).quat_to_mat(d.qpos[3:7])
            vb = R.T @ d.qvel[:3]
            target = base.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                                    d.qpos[base.qidx].copy(), d.qvel[base.vidx].copy(), command)
            for _ in range(base.substeps):
                torque = np.clip(base.ctrl.kp * (target - d.qpos[base.qidx]) - base.ctrl.kd * d.qvel[base.vidx],
                                 -base.ctrl.eff, base.ctrl.eff)
                d.ctrl[base.actuators] = torque
                mujoco.mj_step(base.model, d)
            base.sync_view()

            hit = base.contacts()
            for f in range(4):
                if hit[f] and not prev[f]:
                    fp = d.xpos[base.model.body(f"{FEET[f].lower()}_toe_link").id]
                    h = truth(float(fp[0]), float(fp[1]))
                    if h < -0.5:
                        continue
                    ii = int(round((fp[0] - truth.origin[0]) / truth.res))
                    jj = int(round((fp[1] - truth.origin[1]) / truth.res))
                    ys, xs = np.nonzero(disk)
                    ii_s = np.clip(ii + ys - disk.shape[0] // 2, 0, truth.h.shape[0] - 1)
                    jj_s = np.clip(jj + xs - disk.shape[1] // 2, 0, truth.h.shape[1] - 1)
                    seen = first_obs[ii_s, jj_s]
                    seen = seen[seen >= 0]
                    t_info = float(seen.min()) * dt if seen.size else None
                    t_con = step * dt
                    events.append({"run": c_i, "foot": FEET[f], "step_level": idx(h),
                                   "contact_xy": [round(float(fp[0]), 3), round(float(fp[1]), 3)],
                                   "t_contact_s": round(t_con, 3),
                                   "t_info_s": None if t_info is None else round(t_info, 3),
                                   "lead_s": None if t_info is None else round(t_con - t_info, 3)})
            prev = hit

    leads = [e["lead_s"] for e in events if e["lead_s"] is not None]
    unseen = sum(e["lead_s"] is None for e in events)
    by_level = {}
    for e in events:
        by_level.setdefault(e["step_level"], []).append(e)
    payload = {"n_runs": a.n, "radius_m": a.radius, "events": len(events),
               "unobserved_at_contact": unseen,
               "lead_mean_s": round(float(np.mean(leads)), 3) if leads else None,
               "lead_min_s": round(float(np.min(leads)), 3) if leads else None,
               "lead_p10_s": round(float(np.percentile(leads, 10)), 3) if leads else None,
               "by_level": {int(k): {"n": len(v),
                                     "lead_min_s": round(float(np.min([x["lead_s"] for x in v
                                                                       if x["lead_s"] is not None])), 3)
                                     if any(x["lead_s"] is not None for x in v) else None,
                                     "unobserved": sum(x["lead_s"] is None for x in v)}
                            for k, v in sorted(by_level.items(), reverse=True)},
               "smallest": sorted([e for e in events if e["lead_s"] is not None],
                                  key=lambda e: e["lead_s"])[:12],
               "events_all": events}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "descent.json").write_text(json.dumps(payload, indent=2))
    print(f"events {payload['events']}  unobserved_at_contact {unseen}  "
          f"lead mean {payload['lead_mean_s']}s  min {payload['lead_min_s']}s  p10 {payload['lead_p10_s']}s")
    for k, v in payload["by_level"].items():
        print(f"  step {k:>2}: n={v['n']:>3}  lead_min={v['lead_min_s']}  unobserved={v['unobserved']}")
    print("smallest leads:")
    for e in payload["smallest"][:8]:
        print(f"  {e['foot']} step {e['step_level']}  lead {e['lead_s']:+.2f}s  "
              f"t_info {e['t_info_s']:.2f} t_contact {e['t_contact_s']:.2f}")
    base.cam_rig.close()


if __name__ == "__main__":
    main()

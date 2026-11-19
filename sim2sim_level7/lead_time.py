"""Lead time from "the first step becomes visible" to "a foot touches it".

Earlier sweeps could not tell whether the policy tolerated latency or whether an older map already
covered the step. This measures the two clocks directly, with the trajectory fixed by the truth map
so only the perception pipeline differs:

  t_info     first moment any cell of the *first step* is published into the map the policy reads
  t_contact  first moment a foot contacts the first step
  lead       t_contact - t_info      (negative = the policy stepped on unseen geometry)

Degradations are triggered by the event, not by wall time: the observation gap starts on the frame
that first reveals the step, and publish latency is applied to every frame.

  python lead_time.py --case gap1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, HeightMap, support_plane
from verify_perception import HERE, Runner, Truth

FEET = ["FL", "FR", "HL", "HR"]
CASES = {  # name: (latency_s, gap_s)
    "baseline": (0.0, None), "lat200": (0.2, None), "lat500": (0.5, None),
    "lat1000": (1.0, None), "lat1500": (1.5, None),
    "gap500": (0.0, 0.5), "gap1000": (0.0, 1.0), "gap2000": (0.0, 2.0),
}
STEP_LO, STEP_HI = 0.05, 0.20   # first stair step only (rises are 0.1275 m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--start-x", type=float, default=-3.0,
                    help="start far enough back that the first step is out of view at t=0")
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/lead_time")
    a = ap.parse_args()

    from paired_eval import make_conditions

    latency_s, gap_s = CASES[a.case]
    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)
    base = Runner(cam, truth, 0, [0.0, 0.0, 0.0])
    conds = make_conditions("up", a.n, a.seed)

    region = (truth.h > STEP_LO) & (truth.h < STEP_HI)
    rows = []
    for init in conds:
        base.reset(0)
        d = base.data
        d.qpos[0] = a.start_x + init["dx"]
        d.qpos[1] += init["dy"]
        d.qpos[3:7] = [np.cos(init["dyaw"] * np.pi / 360), 0, 0, np.sin(init["dyaw"] * np.pi / 360)]
        mujoco.mj_forward(base.model, d)
        base.sync_view()
        base.plane = support_plane(base.cam_rig, base.vdata, cam)

        hist = HeightMap(truth.origin, truth.h.shape, truth.res, base.plane, stride=2)
        base.ctrl.height_fn = hist.height_at
        base.ctrl.reset()
        command = np.array([init["vx"], 0.0, 0.0])
        dt = base.ctrl.dt
        lat_steps = int(round(latency_s / dt))
        n = round(a.seconds / dt)

        from tgi.deploy.controller import quat_to_mat

        pending, first_obs = [], np.full(truth.h.shape, -1, np.int32)
        entered, drop_until = False, -1
        t_contact, contact_xy = None, None
        prev = np.zeros(4, bool)
        t_enter_pub = None

        for step in range(n):
            if step % a.depth_every == 0 and step > drop_until:
                terrain, with_robot = base.cam_rig.capture(base.vdata)
                pts, keep = base.cam_rig.points(base.vdata, terrain)
                keep &= ~(with_robot < terrain - 0.02)
                if not entered:
                    p = pts[keep]
                    if len(p):
                        ii = np.rint((p[:, 0] - truth.origin[0]) / truth.res).astype(int)
                        jj = np.rint((p[:, 1] - truth.origin[1]) / truth.res).astype(int)
                        ok = (ii >= 0) & (ii < truth.h.shape[0]) & (jj >= 0) & (jj < truth.h.shape[1])
                        if ok.any() and region[ii[ok], jj[ok]].any():
                            entered = True
                            if gap_s:
                                # the frame that would have revealed the step is the one lost
                                drop_until = step + int(round(gap_s / dt))
                                continue
                pending.append((step, pts, keep))

            while pending and step - pending[0][0] >= lat_steps:
                _, pts, keep = pending.pop(0)
                before = hist.valid.copy()
                hist.fuse(pts, keep, step)
                new = hist.valid & ~before
                first_obs[new] = step
                if t_enter_pub is None and (new & region).any():
                    t_enter_pub = step * dt

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

            hit = base.contacts()
            for f in range(4):
                if hit[f] and not prev[f] and t_contact is None:
                    fp = d.xpos[base.model.body(f"{FEET[f].lower()}_toe_link").id]
                    if STEP_LO < truth(float(fp[0]), float(fp[1])) < STEP_HI:
                        t_contact = step * dt
                        contact_xy = (float(fp[0]), float(fp[1]))
            prev = hit
            if t_contact is not None and t_enter_pub is not None:
                break
        if t_contact is None or t_enter_pub is None:
            break

        ci = int(round((contact_xy[0] - truth.origin[0]) / truth.res))
        cj = int(round((contact_xy[1] - truth.origin[1]) / truth.res))
        cell_known = bool(hist.valid[ci, cj])
        rows.append({"t_info_s": round(t_enter_pub, 2), "t_contact_s": round(t_contact, 2),
                     "lead_s": round(t_contact - t_enter_pub, 2),
                     "contact_cell_known": cell_known,
                     "contact_xy": [round(contact_xy[0], 3), round(contact_xy[1], 3)]})

    payload = {"case": a.case, "latency_s": latency_s, "gap_s": gap_s, "n": len(rows),
               "lead_mean_s": round(float(np.mean([r["lead_s"] for r in rows])), 2) if rows else None,
               "lead_min_s": round(float(np.min([r["lead_s"] for r in rows])), 2) if rows else None,
               "negative_lead": sum(r["lead_s"] < 0 for r in rows),
               "unknown_at_contact": sum(not r["contact_cell_known"] for r in rows),
               "rows": rows}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.case}.json").write_text(json.dumps(payload, indent=2))
    print(f"{a.case:<10} lead mean {payload['lead_mean_s']}s min {payload['lead_min_s']}s  "
          f"negative {payload['negative_lead']}/{len(rows)}  "
          f"unknown@contact {payload['unknown_at_contact']}/{len(rows)}", flush=True)
    base.cam_rig.close()


if __name__ == "__main__":
    main()

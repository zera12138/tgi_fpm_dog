"""Does the fill rule explain the stepping-stone failure? Run C with different fill values."""
import argparse, json, math, sys
from pathlib import Path
import mujoco, numpy as np
from perception import CameraSpec, HeightMap, support_plane
from verify_perception import HERE, Runner, Truth
from tgi.deploy.controller import quat_to_mat

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=14.0)
ap.add_argument("--vx", type=float, default=0.3)
ap.add_argument("--fills", default="plane,-0.35,-0.80")
ap.add_argument("--out", type=Path, default=HERE/"perception_out/stones")
a = ap.parse_args()

truth = Truth(HERE/"scene/heightmap.npz")
cam = CameraSpec(label="assumed", pitch_deg=30.0, vfov_deg=66.0)
base = Runner(cam, truth, 3, [0.0, 0.0, 0.0])
plane = support_plane(base.cam_rig, base.vdata, cam)
rows = []
for name in a.fills.split(","):
    fill = plane if name == "plane" else float(name)
    base.reset(3); plane0 = support_plane(base.cam_rig, base.vdata, cam)
    hist = HeightMap(truth.origin, truth.h.shape, truth.res, fill, stride=2)
    base.ctrl.height_fn = hist.height_at; base.ctrl.reset()
    d = base.data; dt = base.ctrl.dt; n = round(a.seconds/dt)
    command = np.array([a.vx, 0.0, 0.0])
    fell, t_fall, max_x, contacts, false_ground = False, None, -9, 0, 0
    prev = np.zeros(4, bool)
    for step in range(n):
        if step % 3 == 0:
            terr, wr = base.cam_rig.capture(base.vdata)
            pts, keep = base.cam_rig.points(base.vdata, terr); keep &= ~(wr < terr - 0.02)
            hist.fuse(pts, keep, step)
        R = quat_to_mat(d.qpos[3:7]); vb = R.T @ d.qvel[:3]
        target = base.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                                d.qpos[base.qidx].copy(), d.qvel[base.vidx].copy(), command)
        for _ in range(base.substeps):
            d.ctrl[base.actuators] = np.clip(base.ctrl.kp*(target-d.qpos[base.qidx]) - base.ctrl.kd*d.qvel[base.vidx],
                                             -base.ctrl.eff, base.ctrl.eff)
            mujoco.mj_step(base.model, d)
        base.sync_view(); max_x = max(max_x, float(d.qpos[0]))
        pts_all = base.ctrl.vis["pts"]
        h, seen, _ = hist.sample(pts_all[...,0].ravel(), pts_all[...,1].ravel())
        seen = seen.reshape(4,49); tr = truth.at(pts_all[...,0].ravel(), pts_all[...,1].ravel()).reshape(4,49)
        hit = base.contacts()
        for f in range(4):
            if hit[f] and not prev[f]:
                contacts += 1; false_ground += int(((~seen[f]) & (tr[f] < -0.3)).sum())
        prev = hit
        R2 = quat_to_mat(d.qpos[3:7])
        if (not np.isfinite(d.qpos).all()) or R2[2,2] < math.cos(0.7) or d.qpos[2]-truth(*d.qpos[:2]) < 0.16:
            fell, t_fall = True, round(step*dt,2); break
    rows.append({"fill": name, "fill_value": round(fill,3), "fallen": fell, "fallen_at_s": t_fall,
                 "max_x": round(max_x,2), "final_x": round(float(d.qpos[0]),2),
                 "contacts": contacts, "false_ground_cells": false_ground})
    print(json.dumps(rows[-1]), flush=True)
(a.out).mkdir(parents=True, exist_ok=True)
(a.out/"fill_test.json").write_text(json.dumps({"support_plane": round(plane,3), "runs": rows}, indent=2))
base.cam_rig.close()

"""Why does the history map fail on the stepping stones?

Runs the truth-driven policy (so the walk is fixed and correct) while rebuilding the history map,
then compares, for every foot patch, what the map says against the truth:

  observed      cells the depth actually saw
  filled        cells with no observation, showing the fill value to the policy
  false_ground  filled cells whose true height is a pit (< -0.3 m) -- the policy sees a stone

  python diagnose_stones.py --spawn 4 --seconds 14
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from perception import CameraSpec, HeightMap, support_plane
from verify_perception import HERE, Runner, Truth

FEET = ["FL", "FR", "HL", "HR"]
PIT = -0.30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spawn", type=int, default=4, choices=range(1, 6))
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--fill", default="low", choices=["plane", "low", "local"],
                    help="plane reproduces the original failure, low is the current default")
    ap.add_argument("--out", type=Path, default=HERE / "perception_out/stones")
    a = ap.parse_args()

    truth = Truth(HERE / "scene/heightmap.npz")
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)
    base = Runner(cam, truth, a.spawn - 1, [0.0, 0.0, 0.0])
    plane = support_plane(base.cam_rig, base.vdata, cam)
    hist = HeightMap(truth.origin, truth.h.shape, truth.res, plane, stride=2, fill_mode=a.fill)
    base.ctrl.height_fn = truth          # the walk itself is the successful truth run
    base.ctrl.reset()
    command = np.array([a.vx, 0.0, 0.0])
    dt = base.ctrl.dt
    n = round(a.seconds / dt)

    from tgi.deploy.controller import quat_to_mat

    prev = np.zeros(4, bool)
    records, deltas = [], []
    saved = None

    for step in range(n):
        if step % a.depth_every == 0:
            terrain, with_robot = base.cam_rig.capture(base.vdata)
            pts, keep = base.cam_rig.points(base.vdata, terrain)
            keep &= ~(with_robot < terrain - 0.02)
            before = hist.valid.copy()
            hist.fuse(pts, keep, step)

        d = base.data
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

        pts_all = base.ctrl.vis["pts"]
        filled = base.ctrl.height_fn is not None and hist.query(pts_all[..., 0].ravel(),
                                                               pts_all[..., 1].ravel())
        filled = filled.reshape(4, 49)
        h_obs, seen, _ = hist.sample(pts_all[..., 0].ravel(), pts_all[..., 1].ravel())
        seen = seen.reshape(4, 49)
        h_obs = h_obs.reshape(4, 49)
        tr = truth.at(pts_all[..., 0].ravel(), pts_all[..., 1].ravel()).reshape(4, 49)
        deltas.append(float(np.mean(np.abs(filled - tr))))

        hit = base.contacts()
        for f in range(4):
            if hit[f] and not prev[f]:
                # only counts when the fill makes the pit read as solid ground
                false_ground = int(((~seen[f]) & (tr[f] < PIT) & (filled[f] >= PIT)).sum())
                false_pit = int(((~seen[f]) & (tr[f] >= PIT) & (filled[f] < PIT)).sum())
                seen_pit = int((seen[f] & (tr[f] < PIT)).sum())
                records.append({
                    "step": step, "t": round(step * dt, 2), "foot": FEET[f],
                    "xy": [round(float(d.xpos[base.model.body(f"{FEET[f].lower()}_toe_link").id][0]), 3),
                           round(float(d.xpos[base.model.body(f"{FEET[f].lower()}_toe_link").id][1]), 3)],
                    "observed": int(seen[f].sum()), "filled": int((~seen[f]).sum()),
                    "false_ground": false_ground,
                    "false_pit": false_pit,
                    "observed_pit": seen_pit,
                    "height_err_observed_mm": (round(float(np.mean(np.abs(
                        h_obs[f][seen[f]] - tr[f][seen[f]]))) * 1000, 1) if seen[f].any() else None),
                    "height_err_all_mm": round(float(np.mean(np.abs(filled[f] - tr[f]))) * 1000, 1),
                })
            if false_ground > 0 and saved is None:
                # freeze map + patches together so the picture is one instant, not two
                saved = {"step": step, "t": round(step * dt, 2),
                         "height": hist.height.copy(), "valid": hist.valid.copy(),
                         "fill": hist.fill_grid().copy(), "pts": pts_all.copy(),
                         "qx": float(d.qpos[0]), "qy": float(d.qpos[1]),
                         "foot": FEET[f], "false_ground": false_ground}
        prev = hit
        if (not np.isfinite(d.qpos).all()) or quat_to_mat(d.qpos[3:7])[2, 2] < math.cos(0.7):
            break

    fg = [r["false_ground"] for r in records]
    fp = [r["false_pit"] for r in records]
    observed = [r["observed"] for r in records]
    payload = {
        "spawn": a.spawn, "support_plane_m": round(plane, 4), "pit_m": PIT,
        "contacts": len(records),
        "patch_size": 49,
        "mean_observed": round(float(np.mean(observed)), 1) if observed else None,
        "mean_false_ground_cells": round(float(np.mean(fg)), 1) if fg else None,
        "mean_false_pit_cells": round(float(np.mean(fp)), 1) if fp else None,
        "contacts_with_false_ground": int(sum(1 for x in fg if x > 0)),
        "contacts_fully_filled": int(sum(1 for x in observed if x == 0)),
        "mean_height_err_all_mm": round(float(np.mean(deltas)) * 1000, 1),
        "records": records,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stones.json").write_text(json.dumps(payload, indent=2))

    # figure at the first contact where the policy would step on false ground
    if saved is None:
        saved = {"step": n - 1, "t": round((len(deltas) - 1) * dt, 2),
                 "height": hist.height.copy(), "valid": hist.valid.copy(),
                 "fill": hist.fill_grid().copy(), "pts": base.ctrl.vis["pts"].copy(),
                 "qx": float(base.data.qpos[0]), "qy": float(base.data.qpos[1]),
                 "foot": "-", "false_ground": 0}
    pts_all = saved["pts"]
    qx, qy = saved["qx"], saved["qy"]
    lo = np.array([qx - 2.0, qy - 1.6])
    hi = np.array([qx + 3.0, qy + 1.6])
    ext = [truth.origin[1], truth.origin[1] + truth.h.shape[1] * truth.res,
           truth.origin[0], truth.origin[0] + truth.h.shape[0] * truth.res]
    policy_view = np.where(saved["valid"], saved["height"], saved["fill"])
    fg_grid = (~saved["valid"]) & (truth.h < PIT) & (saved["fill"] >= PIT)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    vmin, vmax = -0.9, 0.1
    axes[0].imshow(truth.h, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("truth: stones vs pit")
    axes[1].imshow(policy_view, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].imshow(np.where(saved["valid"], np.nan, 1.0), origin="lower", extent=ext,
                   cmap="Greys", vmin=0, vmax=1, alpha=0.35)
    axes[1].set_title(f"policy view: observed + fill ({a.fill}, {saved['fill'][0, 0]:+.2f} m)")
    axes[2].imshow(truth.h, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].imshow(np.where(fg_grid, 1.0, np.nan), origin="lower", extent=ext,
                   cmap="autumn", vmin=0, vmax=1, alpha=0.9)
    axes[2].set_title("red = pit the policy reads as solid ground")
    colors = ["#e02b20", "#1f6fe0", "#e0a800", "#12b34a"]
    for f in range(4):
        pch = pts_all[f]
        x0, x1 = pch[:, 0].min(), pch[:, 0].max()
        y0, y1 = pch[:, 1].min(), pch[:, 1].max()
        for ax in axes:
            ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec=colors[f], lw=1.2))
            ax.set_xlim(lo[1], hi[1]); ax.set_ylim(lo[0], hi[0])
            ax.set_aspect("equal"); ax.set_xlabel("y (m)"); ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("x (m)")
    fig.suptitle(f"spawn {a.spawn} t={saved['t']}s  fill={a.fill}  "
                 f"({saved['foot']} contact, {saved['false_ground']} pit cells read as ground)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "stones.png", dpi=120)
    plt.close(fig)

    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    print(f"figure -> {a.out/'stones.png'}")
    base.cam_rig.close()


if __name__ == "__main__":
    main()

"""Static geometry checks: is the fused map unbiased, and does the occlusion pass really work?

Run with the robot parked on flat ground and on the stairs. Nothing here depends on the policy.

  python check_geometry.py --spawn 1   # flat apron
  python check_geometry.py --spawn 2   # top of the stairs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, DepthCamera, HeightMap, build_model, robot_geom_ids, support_plane

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spawn", type=int, default=1, choices=range(1, 6))
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()

    truth_arc = np.load(HERE / "scene/heightmap.npz")
    th, torigin, tres = truth_arc["heights"].astype(np.float32), truth_arc["origin"], float(truth_arc["res"])
    info = json.loads((HERE / "scene/scene.json").read_text())
    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=a.vfov)

    model = build_model(HERE / "scene/level7.xml", cam, apply_pitch=False)
    vmodel = build_model(HERE / "scene/level7.xml", cam, apply_pitch=True)
    data, vdata = mujoco.MjData(model), mujoco.MjData(vmodel)
    sp = info["spawns"][a.spawn - 1]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [*sp["xy"], sp["z"] + 0.43]
    data.qpos[3:7] = [np.cos(sp["yaw"] / 2), 0, 0, np.sin(sp["yaw"] / 2)]
    mujoco.mj_forward(model, data)
    vdata.qpos[:], vdata.qvel[:] = data.qpos, data.qvel
    mujoco.mj_forward(vmodel, vdata)

    rig = DepthCamera(vmodel, cam)
    plane = support_plane(rig, vdata, cam)
    hm = HeightMap(torigin, th.shape, tres, plane, stride=1)

    occluded_px = 0
    total_px = 0
    for step in range(a.frames):
        terrain, with_robot = rig.capture(vdata)
        pts, keep = rig.points(vdata, terrain)
        occ = with_robot < terrain - 0.02
        occluded_px += int(occ.sum())
        total_px += occ.size
        hm.fuse(pts, keep & ~occ, step)

    # signed error over every fused cell, and split by distance to a height discontinuity
    valid = hm.valid
    ii, jj = np.nonzero(valid)
    world = hm.origin + np.stack([ii, jj], -1) * hm.res
    ti = np.rint((world[:, 0] - torigin[0]) / tres).astype(int)
    tj = np.rint((world[:, 1] - torigin[1]) / tres).astype(int)
    ref = th[ti, tj].astype(float)
    err = hm.height[valid].astype(float) - ref
    level = np.abs(ref) > 1e-6  # non-zero height = on a step/platform

    grad = np.abs(np.gradient(th, axis=0)) + np.abs(np.gradient(th, axis=1))
    gi = np.clip(ti, 1, th.shape[0] - 2)
    gj = np.clip(tj, 1, th.shape[1] - 2)
    edge = grad[gi, gj] > 0.02

    # does any fused cell sit where the robot body is?  the legs must never be written as terrain
    torso = vmodel.body("torso").id
    legs = [vmodel.body(n).id for n in ("fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link")]
    leg_xyz = [vdata.xpos[b] for b in legs]
    near_leg = np.zeros(len(world), bool)
    for p in leg_xyz:
        near_leg |= np.linalg.norm(world[:, :2] - p[:2], axis=1) < 0.03
    leg_spike = near_leg & (np.abs(err) > 0.05)

    # --- error decomposition: the metric must not blame the fusion for the representation ---
    span = np.zeros(len(world))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ai = np.clip(ti + di, 0, th.shape[0] - 1)
            bj = np.clip(tj + dj, 0, th.shape[1] - 1)
            span = np.maximum(span, th[ai, bj].astype(float))
    span = span - ref  # how far the 3x3 truth neighbourhood spreads above the cell centre
    interior = span < 0.02
    cross_edge = span >= 0.02
    lateral = np.abs(np.abs(world[:, 1]) - 2.0) < 0.05

    def stats(mask):
        if not mask.any():
            return {"count": 0}
        e = err[mask]
        return {"count": int(mask.sum()),
                "median_mm": round(float(np.median(e) * 1000), 2),
                "p95_abs_mm": round(float(np.percentile(np.abs(e), 95) * 1000), 2),
                "bias_mm": round(float(e.mean() * 1000), 2)}

    # edge position: where does the height jump along x at y ~ 0?
    def jump_x(field):
        row = field[:, np.argmin(np.abs(torigin[1] + np.arange(field.shape[1]) * tres))]
        d = np.diff(row)
        return (np.flatnonzero(np.abs(d) > 0.05) + 0.5) * tres + torigin[0]

    map_row = np.full(th.shape, np.nan, np.float32)
    map_row[ii, jj] = hm.height[valid]
    tj0 = np.argmin(np.abs(torigin[1] + np.arange(th.shape[1]) * tres))
    truth_jumps = jump_x(th)
    got = np.diff(map_row[:, tj0].astype(float))
    map_jumps = (np.flatnonzero(np.abs(got) > 0.05) + 0.5) * tres + torigin[0]
    offsets = []
    for x in truth_jumps:
        if len(map_jumps):
            offsets.append(float(map_jumps[np.argmin(np.abs(map_jumps - x))] - x))
    edge_pos = {"truth_jumps": int(len(truth_jumps)), "map_jumps": int(len(map_jumps)),
                "max_offset_mm": round(max((abs(o) for o in offsets), default=0) * 1000, 2),
                "mean_offset_mm": round(float(np.mean([abs(o) for o in offsets])) * 1000, 2) if offsets else None}

    out = {
        "spawn": sp["name"], "pitch_deg": a.pitch, "support_plane_m": round(plane, 4),
        "occlusion": {"pixels_total": total_px, "pixels_occluded": occluded_px,
                      "fraction": round(occluded_px / total_px, 4)},
        "fused_cells": int(valid.sum()),
        "signed_error_m": {
            "mean": round(float(err.mean()), 5),
            "median_mm": round(float(np.median(err) * 1000), 2),
            "p95_abs_mm": round(float(np.percentile(np.abs(err), 95) * 1000), 2),
            "max_abs_mm": round(float(np.abs(err).max() * 1000), 2),
        },
        "flat_cells": {"count": int((~level).sum()),
                       "median_mm": round(float(np.median(err[~level]) * 1000), 2) if (~level).any() else None},
        "stepped_cells": {"count": int(level.sum()),
                          "median_mm": round(float(np.median(err[level]) * 1000), 2) if level.any() else None},
        "cells_near_discontinuity": {"count": int(edge.sum()),
                                     "median_mm": round(float(np.median(err[edge]) * 1000), 2) if edge.any() else None,
                                     "max_mm": round(float(np.abs(err[edge]).max() * 1000), 2) if edge.any() else None},
        "leg_cells_with_5cm_spike": int(leg_spike.sum()),
        "by_class": {"tread_interior": stats(interior & ~lateral),
                     "cross_edge_cell": stats(cross_edge & ~lateral),
                     "stairs_lateral_edge": stats(lateral)},
        "edge_position": edge_pos,
        "note": "error = fused height - raytraced truth height at the same 4 cm cell",
    }
    (HERE / "perception_out").mkdir(exist_ok=True)
    (HERE / "perception_out" / f"geometry_spawn{a.spawn}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    rig.close()


if __name__ == "__main__":
    main()

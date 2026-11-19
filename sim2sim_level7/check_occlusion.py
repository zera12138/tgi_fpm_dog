"""Positive control for the two-pass occlusion handling.

The production mount never puts the legs in frame, so a zero occlusion count proves nothing. This
script uses a deliberately different *test rig* -- a camera on the torso looking steeply down -- to
put robot geometry inside the image, then checks three things:

  1. occluded pixels actually occur,
  2. no fused cell lands on the robot's own surface,
  3. dropping the occlusion mask would write the terrain *behind* the robot (the see-through bug).

It is a code-path test, not a claim about the real mount.

  python check_occlusion.py --pitch 80
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from perception import CameraSpec, DEPTH_CAM, HeightMap, robot_geom_ids

HERE = Path(__file__).resolve().parent


def build_test_rig(scene_xml, cam: CameraSpec):
    """Camera bolted to the torso, pitched steeply down so the legs enter the frame."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    torso = spec.body("torso")
    rig = torso.add_body()
    rig.name = "test_rig"
    rig.pos = [0.05, 0.0, 0.10]
    # A rotation about +y sends +z to (sin t, 0, cos t); to look ``pitch`` degrees below the
    # horizon along +x we need t = 90 + pitch, so that +z is the view direction.
    half = math.radians(90.0 + cam.pitch_deg) / 2.0
    rig.quat = [math.cos(half), 0.0, math.sin(half), 0.0]
    camera = rig.add_camera()
    camera.name = DEPTH_CAM
    camera.quat = [0.0, 1.0, 0.0, 0.0]
    camera.fovy = cam.vfov_deg
    spec.visual.global_.offwidth, spec.visual.global_.offheight = cam.width, cam.height
    return spec.compile()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitch", type=float, default=80.0)
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--spawn", type=int, default=1, choices=range(1, 6))
    a = ap.parse_args()

    arc = np.load(HERE / "scene/heightmap.npz")
    th, torigin, tres = arc["heights"].astype(np.float32), arc["origin"], float(arc["res"])
    info = json.loads((HERE / "scene/scene.json").read_text())
    sp = info["spawns"][a.spawn - 1]
    cam = CameraSpec(label="test-rig", pitch_deg=a.pitch, vfov_deg=a.vfov, far_m=4.0)

    model = build_test_rig(HERE / "scene/level7.xml", cam)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [*sp["xy"], sp["z"] + 0.43]
    data.qpos[3:7] = [math.cos(sp["yaw"] / 2), 0, 0, math.sin(sp["yaw"] / 2)]
    mujoco.mj_forward(model, data)

    from perception import DepthCamera

    rig = DepthCamera(model, cam)
    terrain, with_robot = rig.capture(data)
    occ = with_robot < terrain - 0.02
    print(f"occluded pixels: {int(occ.sum())} / {occ.size} ({occ.mean()*100:.2f}%)")

    pts, keep = rig.points(data, terrain)

    def fuse(use_mask):
        hm = HeightMap(torigin, th.shape, tres, 0.0, stride=1)
        hm.fuse(pts, keep & use_mask, 0)
        return hm

    masked = fuse(~occ)
    unmasked = fuse(np.ones_like(occ))

    # cells the mask keeps out, and what value they would have taken
    dropped = unmasked.valid & ~masked.valid
    changed = masked.valid & unmasked.valid & (np.abs(masked.height - unmasked.height) > 1e-4)

    # does any fused cell sit on the robot body?
    torso_bodies = {model.body("torso").id}
    changed_t = True
    while changed_t:
        changed_t = False
        for i in range(model.nbody):
            if i not in torso_bodies and model.body_parentid[i] in torso_bodies:
                torso_bodies.add(i)
                changed_t = True
    robot_geoms = robot_geom_ids(model)
    geoms = [model.geom(g) for g in robot_geoms]
    ii, jj = np.nonzero(masked.valid)
    world = masked.origin + np.stack([ii, jj], -1) * masked.res
    on_robot = np.zeros(len(world), bool)
    for g in geoms:
        if model.geom_bodyid[g.id] not in torso_bodies:
            continue
        gp = data.geom_xpos[g.id]
        if np.linalg.norm(world[:, :2] - gp[:2], axis=1).min() < 0.06:
            near = np.linalg.norm(world[:, :2] - gp[:2], axis=1) < 0.06
            # a cell is "on the robot" only if its height is well above the local terrain
            ti = np.clip(np.rint((world[near, 0] - torigin[0]) / tres).astype(int), 0, th.shape[0] - 1)
            tj = np.clip(np.rint((world[near, 1] - torigin[1]) / tres).astype(int), 0, th.shape[1] - 1)
            on_robot[near] |= (masked.height[tuple(np.stack([ii, jj], -1)[near].T)] - th[ti, tj]) > 0.05

    out = {
        "test_rig": "camera on torso, pitched down (NOT the production mount)",
        "pitch_deg": a.pitch,
        "occluded_pixels": int(occ.sum()),
        "occluded_fraction": round(float(occ.mean()), 4),
        "cells_kept_out_by_mask": int(dropped.sum()),
        "cells_with_changed_height": int(changed.sum()),
        "cells_fused_on_robot_surface": int(on_robot.sum()),
        "verdict": {
            "occlusion_triggered": bool(occ.sum() > 0),
            "no_robot_surface_written": bool(on_robot.sum() == 0),
            "see_through_prevented": bool((changed.sum() + dropped.sum()) > 0),
        },
        "note": "cells_kept_out/cells_changed > 0 means the mask suppressed geometry the second "
                "pass would otherwise have let us see through the robot.",
    }
    (HERE / "perception_out").mkdir(exist_ok=True)
    (HERE / "perception_out" / "occlusion_positive.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    rig.close()


if __name__ == "__main__":
    main()

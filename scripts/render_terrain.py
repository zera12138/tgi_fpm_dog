"""Render/evaluate the policy on a MuJoCo terrain built to match the training difficulty.

The terrain is a set of boxes (stairs or stepping stones) generated with the same
difficulty mapping as the Isaac training terrains, so the FPM sees a matching
height map and the video shows real terrain rather than flat ground.

Usage:
  python scripts/render_terrain.py --run <log dir> --iter 40000 --terrain stairs \
      --difficulty 0.43 --seconds 12 --out logs/terrain_stairs.mp4
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="/home/odroid/bpx-tgi-fpm")
ap.add_argument("--run", required=True)
ap.add_argument("--iter", type=int, default=40000)
ap.add_argument("--terrain", default="stairs", choices=["stairs", "stones", "rough"])
ap.add_argument("--difficulty", type=float, default=0.43, help="training difficulty in [0,1] (row/(rows-1))")
ap.add_argument("--command", type=float, default=0.3)
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--out", default=None)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--fps", type=int, default=25)
a = ap.parse_args()

root = Path(a.project)
import sys
sys.path[:0] = [str(root / "tgi/src"), str(root / "fpm/src")]

from tgi.robot import *  # noqa: F401,F403
from tgi.policy_io import load_policy_bundle, bootstrap_footholds, perceive, build_mdp
from tgi.mdp.mdp import RewardCfg, unflatten_state
from tgi.mdp.history import append_history
from tgi.mdp.memory import init_memory, update_memory
from tgi.mdp.heightmap import HeightMap

torch.set_num_threads(2)
START_X, WIDTH_Y = 0.0, 2.0
GAP_FLOOR = -0.8


def terrain_rects(kind: str, d: float):
    """(x0, x1, y0, y1, top) boxes; matches the Isaac sub-terrain parameterisation."""
    rects = [(START_X - 1.2, START_X + 0.8, -WIDTH_Y / 2, WIDTH_Y / 2, 0.0)]  # start platform
    if kind == "stairs":
        rise = 0.03 + d * (0.16 - 0.03)          # pyramid_stairs step_height_range
        depth, n = 0.35, 7
        x = START_X + 0.8
        for i in range(n):
            rects.append((x, x + depth, -WIDTH_Y / 2, WIDTH_Y / 2, rise * (i + 1)))
            x += depth
    elif kind == "stones":
        d_eff = d ** 1.5                          # isaac_stepping_stones progression
        w = 0.5 + d_eff * (0.3 - 0.5)
        g = 0.1 + d_eff * (0.2 - 0.1)
        hmax = 0.05 * d_eff
        rng = np.random.default_rng(7)
        x = START_X + 0.8
        row = 0
        while x < START_X + 4.5:
            y = -WIDTH_Y / 2
            while y < WIDTH_Y / 2 - 1e-6:
                rects.append((x, min(x + w, START_X + 4.5), y, min(y + w, WIDTH_Y / 2),
                              rng.uniform(-hmax, hmax)))
                y += w + g
            x += w + g
            row += 1
    else:  # rough
        noise = 0.01 + d * (0.06 - 0.01)
        rng = np.random.default_rng(11)
        cell = 0.25
        x = START_X + 0.8
        while x < START_X + 4.5:
            y = -WIDTH_Y / 2
            while y < WIDTH_Y / 2 - 1e-6:
                rects.append((x, x + cell, y, y + cell, rng.uniform(-noise, noise)))
                y += cell
            x += cell
    return rects


def height_at(rects, x: float, y: float) -> float:
    for x0, x1, y0, y1, top in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return top
    return GAP_FLOOR


def build_model(rects, width: int, height: int):
    spec = mujoco.MjSpec.from_file(str(root / "tgi/src/tgi/assets/bpx.xml"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max(width, 640), max(height, 480)
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.haze = 0.0
    spec.visual.headlight.active = 1
    spec.visual.headlight.diffuse = [0.40, 0.40, 0.40]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    for pos, direction, power in (([2.5, 2.0, 3.0], [-1.0, -1.0, -1.2], 0.55),
                                  ([-2.0, 1.0, 2.5], [1.0, -0.4, -1.0], 0.35),
                                  ([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.30)):
        light = spec.worldbody.add_light()
        light.pos, light.dir, light.diffuse = pos, direction, [power, power, power]
    sky = spec.add_texture()
    sky.name, sky.type = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin, sky.rgb1, sky.rgb2 = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, [0.35, 0.50, 0.66], [0.72, 0.75, 0.79]
    sky.width, sky.height = 512, 3072
    grid = spec.add_texture()
    grid.name, grid.type = "floor_grid", mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin, grid.rgb1, grid.rgb2 = mujoco.mjtBuiltin.mjBUILTIN_CHECKER, [0.46, 0.48, 0.52], [0.36, 0.38, 0.42]
    grid.width, grid.height = 256, 256
    mat = spec.add_material()
    mat.name, mat.texrepeat, mat.specular, mat.shininess = "floor_mat", [14, 14], 0.05, 0.1
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    dark = spec.add_material()
    dark.name, dark.texrepeat, dark.specular, dark.shininess = "floor_mat_b", [14, 14], 0.05, 0.1
    dark.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    dark.rgba = [0.75, 0.75, 0.80, 1.0]
    ground = spec.geom("ground")
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 0.1]
    ground.pos = [0, 0, GAP_FLOOR]
    ground.material = "floor_mat"
    marker_rgba = [(1, .2, .2, 1), (.2, .5, 1, 1), (1, .75, .1, 1), (.2, 1, .4, 1)]
    for i, rgba in enumerate(marker_rgba):
        body = spec.worldbody.add_body()
        body.name, body.mocap, body.pos = f"fpm_marker_{i}", True, [0, 0, -5]
        geom = body.add_geom()
        geom.name = f"fpm_marker_geom_{i}"
        geom.type, geom.size, geom.rgba = mujoco.mjtGeom.mjGEOM_SPHERE, [0.028, 0, 0], list(rgba)
        geom.contype = geom.conaffinity = 0

    for idx, (x0, x1, y0, y1, top) in enumerate(rects):
        body = spec.worldbody.add_body()
        body.name = f"terrain_{x0:.2f}_{y0:.2f}"
        body.pos = [(x0 + x1) / 2, (y0 + y1) / 2, (top + GAP_FLOOR) / 2]
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [(x1 - x0) / 2, (y1 - y0) / 2, (top - GAP_FLOOR) / 2]
        geom.material = "floor_mat_b" if idx % 2 else "floor_mat"
        geom.friction = [1.0, 0.005, 0.0001]
    return spec.compile()


class Shim:
    """Minimal interface for build_mdp."""

    def __init__(self, heights, origin, res):
        self.joint_names = list(JOINT_NAMES)
        self.default_q = torch.tensor(DEFAULT_Q)
        self.action_scale = torch.full((12,), ACTION_SCALE)
        from tgi.robot import MJCF
        self.kin_model = mujoco.MjModel.from_xml_path(MJCF)
        self._hm = HeightMap(heights, origin, res)

    @property
    def soft_limits(self):
        m, names = self.kin_model, self.joint_names
        lo = torch.tensor([m.jnt_range[m.joint(n).id, 0] for n in names])
        hi = torch.tensor([m.jnt_range[m.joint(n).id, 1] for n in names])
        return lo, hi

    def heightmap(self, device, res=0.04):
        return self._hm


def main() -> int:
    ckpt = Path(a.run) / f"ckpt_{a.iter:06d}.pt"
    out = Path(a.out) if a.out else root / "logs" / f"terrain_{a.terrain}_{a.iter}.mp4"
    rects = terrain_rects(a.terrain, a.difficulty)

    res = 0.04
    xs = np.arange(-2.0, 6.0 + 1e-9, res)
    ys = np.arange(-1.6, 1.6 + 1e-9, res)
    h = np.array([[height_at(rects, x, y) for y in ys] for x in xs], dtype=np.float32)
    shim = Shim(torch.from_numpy(h), (float(xs[0]), float(ys[0])), res)

    model = build_model(rects, a.width, a.height)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0], data.qpos[1], data.qpos[2] = START_X - 0.4, 0.0, NOMINAL_HEIGHT + 0.02
    mujoco.mj_forward(model, data)

    bundle = load_policy_bundle(str(ckpt), "cpu")
    mdp = build_mdp(shim, "cpu", reward_cfg=RewardCfg(**{k: v for k, v in bundle.args.items()
                                                          if k in RewardCfg.__dataclass_fields__}),
                    use_mem=bool(bundle.args.get("mem", 0)))
    use_mem = bool(bundle.args.get("mem", 0))
    mem = init_memory(1, "cpu") if use_mem else None
    hist = None
    cmd = torch.tensor([[a.command, 0.0, 0.0]])
    act_prev = torch.zeros(1, 12)
    xi = torch.zeros(1, 21)
    xi[:, 0], xi[:, -3] = 0.8, 1.0
    bid = model.body("torso").id
    foot_bids = [model.body(n).id for n in FOOT_BODIES]

    ffmpeg = str(root / ".video_deps/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2")
    ff = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-", "-an",
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
                          stdin=subprocess.PIPE)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.6, 142.0, -32.0
    every = max(1, round((1.0 / a.fps) / CONTROL_DT))
    fell, frames, max_x = False, 0, data.qpos[0]
    with torch.no_grad():
        for step in range(round(a.seconds / CONTROL_DT)):
            mujoco.mj_forward(model, data)
            R = data.xmat[bid].reshape(3, 3).copy()
            flat = np.concatenate((data.qpos[:7], data.qvel[:3], R @ data.qvel[3:6],
                                   data.qpos[7:], data.qvel[6:])).astype(np.float32)
            S = unflatten_state(torch.from_numpy(flat)[None])
            K = mdp.context(S)
            if mdp.termination(S, K).item():
                fell = True
                break
            if use_mem and step > 0:
                mem = update_memory(mem, torch.from_numpy(foot_contacts(model, data, foot_bids))[None],
                                    CONTROL_DT, mdp.cfg.gait_period)
            o = mdp.proprio(S, act_prev, cmd, xi, noise=False)
            hist = append_history(hist, o, bundle.pi.hist)
            fh = bundle.fpm(mdp.system_state(S, cmd)) if bundle.fpm else bootstrap_footholds(1, "cpu")
            used = fh if bundle.ck.get("fpm_active", True) else bootstrap_footholds(1, "cpu")
            extra = perceive(mdp, S, K, used, noise=False, mem=mem, cmd=cmd)
            if bool(bundle.ck.get("fpm_active", True)):
                yaw = math.atan2(R[1, 0], R[0, 0])
                c_y, s_y = math.cos(yaw), math.sin(yaw)
                pred = fh[0].numpy()
                data.mocap_pos[:, 0] = data.qpos[0] + c_y * pred[:, 0] - s_y * pred[:, 1]
                data.mocap_pos[:, 1] = data.qpos[1] + s_y * pred[:, 0] + c_y * pred[:, 1]
                wy = data.qpos[1] + s_y * pred[:, 0] + c_y * pred[:, 1]
                wx = data.qpos[0] + c_y * pred[:, 0] - s_y * pred[:, 1]
                data.mocap_pos[:, 2] = [height_at(rects, float(x), float(y)) + 0.04 for x, y in zip(wx, wy)]
            act_prev = bundle.pi(hist, extra).clamp(-10, 10)
            target = np.array(DEFAULT_Q) + ACTION_SCALE * act_prev[0].numpy()
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                data.ctrl[:] = np.clip(KP * (target - data.qpos[7:]) - KD * data.qvel[6:], -EFFORT_LIMIT, EFFORT_LIMIT)
                mujoco.mj_step(model, data)
            max_x = max(max_x, float(data.qpos[0]))
            if step % 50 == 0:
                print(f"  t={step * CONTROL_DT:4.1f}s x={data.qpos[0]:5.2f} z={data.qpos[2]:5.3f} "
                      f"yaw={math.degrees(math.atan2(R[1, 0], R[0, 0])):5.1f}", flush=True)
            if step % every == 0:
                cam.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.05])
                renderer.update_scene(data, camera=cam)
                ff.stdin.write(renderer.render().tobytes())
                frames += 1

    ff.stdin.close()
    rc = ff.wait()
    renderer.close()
    print(f"terrain={a.terrain} difficulty={a.difficulty:.2f} command={a.command} "
          f"fell={fell} distance={max_x - (START_X - 0.4):.2f} m frames={frames} "
          f"ffmpeg_rc={rc} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def foot_contacts(model, data, bids):
    net = np.zeros((4, 3))
    force = np.zeros(6)
    for ci, contact in enumerate(data.contact):
        mujoco.mj_contactForce(model, data, ci, force)
        fw = contact.frame.reshape(3, 3).T @ force[:3]
        for geom, sign in ((contact.geom1, -1), (contact.geom2, 1)):
            body = int(model.geom_bodyid[geom])
            if body in bids:
                net[bids.index(body)] += sign * fw
    return (np.linalg.norm(net, axis=1) > 1).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())

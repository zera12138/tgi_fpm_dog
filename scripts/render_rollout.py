"""Render a frozen BPX-TGI checkpoint rollout to MP4, with FPM predicted footholds as spheres.

MuJoCo CPU rollout (same controller path as scripts/evaluate_pilot.py), offscreen render,
frames piped straight into ffmpeg. Markers are mocap bodies injected via MjSpec.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="/home/odroid/bpx-tgi-fpm")
ap.add_argument("--run", required=True)
ap.add_argument("--iter", type=int, default=10000)
ap.add_argument("--case", default="forward", choices=["forward", "stand", "reverse", "turn"])
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--out", default=None)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--no-fpm", action="store_true", help="draw no markers")
a = ap.parse_args()

root = Path(a.project)
sys.path[:0] = [str(root / "tgi/src"), str(root / "fpm/src")]

from tgi.robot import *  # noqa: F401,F403
from tgi.policy_io import load_policy_bundle, bootstrap_footholds, perceive
from tgi.mdp.mdp import MDP, RewardCfg, unflatten_state
from tgi.mdp.kinematics import RobotKinematics
from tgi.mdp.heightmap import HeightMap
from tgi.mdp.terrain import TerrainFeatures
from tgi.mdp.interface import FootholdScan
from tgi.mdp.memory import init_memory, update_memory
from tgi.mdp.history import append_history

torch.set_num_threads(2)

CMDS = {"forward": [0.3, 0.0, 0.0], "stand": [0.0, 0.0, 0.0],
        "reverse": [-0.2, 0.0, 0.0], "turn": [0.0, 0.0, 0.3]}
MARKER_RGBA = [(1, .2, .2, 1), (.2, .5, 1, 1), (1, .75, .1, 1), (.2, 1, .4, 1)]


def build_model(n_markers: int, width: int, height: int):
    spec = mujoco.MjSpec.from_file(str(root / "tgi/src/tgi/assets/bpx.xml"))
    spec.visual.global_.offwidth = max(width, 640)
    spec.visual.global_.offheight = max(height, 480)
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.haze = 0.0
    spec.visual.headlight.active = 1
    spec.visual.headlight.diffuse = [0.9, 0.9, 0.9]
    spec.visual.headlight.ambient = [0.75, 0.75, 0.75]
    for pos, direction in (([2.5, 2.0, 3.0], [-1.0, -1.0, -1.2]),
                           ([-2.0, 1.0, 2.5], [1.0, -0.4, -1.0]),
                           ([0.0, 0.0, 5.0], [0.0, 0.0, -1.0])):
        light = spec.worldbody.add_light()
        light.pos = pos
        light.dir = direction
        light.diffuse = [0.55, 0.55, 0.55]
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.35, 0.50, 0.66]
    sky.rgb2 = [0.72, 0.75, 0.79]
    sky.width = 512
    sky.height = 3072
    grid = spec.add_texture()
    grid.name = "floor_grid"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.86, 0.88, 0.91]
    grid.rgb2 = [0.74, 0.77, 0.80]
    grid.width = 256
    grid.height = 256
    mat = spec.add_material()
    mat.name = "floor_mat"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    mat.texrepeat = [14, 14]
    mat.specular = 0.05
    mat.shininess = 0.1
    ground = spec.geom("ground")
    ground.material = "floor_mat"

    for i in range(n_markers):
        body = spec.worldbody.add_body()
        body.name = f"fpm_marker_{i}"
        body.mocap = True
        body.pos = [0, 0, -5]
        geom = body.add_geom()
        geom.name = f"fpm_marker_geom_{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [0.028, 0, 0]
        geom.rgba = list(MARKER_RGBA[i % 4])
        geom.contype = 0
        geom.conaffinity = 0
    return spec.compile()


def _foot_contacts(model, data) -> "np.ndarray":
    """Binary per-foot contact exactly like the training collector (toe net force > 1 N)."""
    bids = [model.body(n).id for n in FOOT_BODIES]
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


def main() -> int:
    ckpt = Path(a.run) / f"ckpt_{a.iter:06d}.pt"
    out = Path(a.out) if a.out else root / "logs" / f"rollout_{a.case}_{a.iter}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd_np = np.array(CMDS[a.case], dtype=np.float64)

    show_markers = not a.no_fpm
    model = build_model(4 if show_markers else 0, a.width, a.height)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    bundle = load_policy_bundle(str(ckpt), "cpu")
    kin = RobotKinematics(model, list(JOINT_NAMES), "cpu")
    hm = HeightMap(torch.zeros(101, 101), (-2.0, -2.0), 0.04)
    hm.build_support_field()
    rc = RewardCfg(**{k: v for k, v in bundle.args.items() if k in RewardCfg.__dataclass_fields__})
    use_mem = bool(bundle.args.get("mem", 0))
    mdp = MDP(kin, TerrainFeatures(hm, "cpu"), torch.tensor(DEFAULT_Q), list(JOINT_NAMES),
              kin.q_lo, kin.q_hi, torch.full((12,), ACTION_SCALE), rc, use_mem=use_mem)
    mdp.scan = FootholdScan(hm, "cpu")
    active = bundle.fpm is not None and bool(bundle.ck.get("fpm_active", False))
    bid = model.body("torso").id
    cmd = torch.tensor([CMDS[a.case]], dtype=torch.float32)
    act_prev = torch.zeros(1, 12)
    xi = torch.zeros(1, 21)
    xi[:, 0] = 0.8
    xi[:, -3] = 1.0

    ffmpeg = str(root / ".video_deps/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2")
    ff = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
        stdin=subprocess.PIPE)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 142.0
    cam.elevation = -13.0

    every = max(1, round((1.0 / a.fps) / CONTROL_DT))
    steps = round(a.seconds / CONTROL_DT)
    fell = False
    mem = init_memory(1, "cpu") if use_mem else None
    history = None
    frames = 0
    yaw_tot = 0.0
    with torch.no_grad():
        for step in range(steps):
            mujoco.mj_forward(model, data)
            R = data.xmat[bid].reshape(3, 3).copy()
            flat = np.concatenate((data.qpos[:7], data.qvel[:3], R @ data.qvel[3:6],
                                   data.qpos[7:], data.qvel[6:])).astype(np.float32)
            S = unflatten_state(torch.from_numpy(flat)[None])
            K = mdp.context(S)
            if mdp.termination(S, K).item():
                fell = True
                break
            if use_mem and step > 0:  # training semantics: first observation uses mem_0
                mem = update_memory(mem, torch.from_numpy(_foot_contacts(model, data))[None], CONTROL_DT,
                                    mdp.cfg.gait_period)
            prediction = bundle.fpm(mdp.system_state(S, cmd)) if bundle.fpm else bootstrap_footholds(1, "cpu")
            used = prediction if active else bootstrap_footholds(1, "cpu")
            if show_markers and active:
                yaw = math.atan2(R[1, 0], R[0, 0])
                c, s = math.cos(yaw), math.sin(yaw)
                p = prediction[0].numpy()
                wx = data.qpos[0] + c * p[:, 0] - s * p[:, 1]
                wy = data.qpos[1] + s * p[:, 0] + c * p[:, 1]
                data.mocap_pos[:, 0] = wx
                data.mocap_pos[:, 1] = wy
                data.mocap_pos[:, 2] = 0.03
            obs = mdp.proprio(S, act_prev, cmd, xi, noise=False)
            extra = perceive(mdp, S, K, used, noise=False, mem=mem, cmd=cmd)
            history = append_history(history, obs, bundle.pi.hist)
            act_prev = mdp.jmap.to_mj(bundle.pi(history, extra)).clamp(-ACTION_CLIP, ACTION_CLIP)
            target = np.array(DEFAULT_Q) + ACTION_SCALE * act_prev[0].numpy()
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                data.ctrl[:] = np.clip(KP * (target - data.qpos[7:]) - KD * data.qvel[6:],
                                       -EFFORT_LIMIT, EFFORT_LIMIT)
                mujoco.mj_step(model, data)
            if step % every == 0:
                cam.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.05])
                renderer.update_scene(data, camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                ff.stdin.write(renderer.render().tobytes())
                frames += 1
            yaw_tot = math.atan2(R[1, 0], R[0, 0])

    ff.stdin.close()
    rc_ff = ff.wait()
    renderer.close()
    dist = math.hypot(data.qpos[0], data.qpos[1])
    print(f"wrote {out} frames={frames} fell={fell} dist={dist:.2f} m "
          f"final_xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) yaw={math.degrees(yaw_tot):.0f} deg "
          f"ffmpeg_rc={rc_ff} size={out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

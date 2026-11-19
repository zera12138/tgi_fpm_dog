"""Perception-degradation sweep: how well does the trained policy cope with an imperfect height map?

The policy never sees the terrain directly; it sees foothold patches read from a HeightMap. This
script keeps the physics and the policy identical and only corrupts that map:

  * Gaussian height noise      (depth noise / pose error)
  * forward-camera visibility  (cells behind the robot are "unknown" until visited)
  * unknown cells              filled with the flat default, like the FPM bootstrap

Usage:
  python scripts/degrade_eval.py --run <log dir> --iter 100000 --terrain stairs \
      --noise-cm 0,1,2,3 --fov 0,1 --seconds 15
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="/home/odroid/bpx-tgi-fpm")
ap.add_argument("--run", required=True)
ap.add_argument("--iter", type=int, default=100000)
ap.add_argument("--terrain", default="stairs", choices=["stairs", "stones", "flat"])
ap.add_argument("--difficulty", type=float, default=0.51)
ap.add_argument("--command", type=float, default=0.3)
ap.add_argument("--start-x", type=float, default=-0.4)
ap.add_argument("--seconds", type=float, default=15.0)
ap.add_argument("--noise-cm", default="0,1,2,3")
ap.add_argument("--offset-cm", default="0", help="systematic height bias (map says terrain is this much higher)")
ap.add_argument("--fov", default="0,1")
ap.add_argument("--fov-range", type=float, default=2.5)
ap.add_argument("--fov-half-angle", type=float, default=35.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

root = Path(a.project)
sys.path[:0] = [str(root / "tgi/src"), str(root / "fpm/src")]

from tgi.robot import *  # noqa: F401,F403
from tgi.policy_io import load_policy_bundle, bootstrap_footholds, perceive, build_mdp
from tgi.mdp.mdp import RewardCfg, unflatten_state
from tgi.mdp.history import append_history
from tgi.mdp.memory import init_memory, update_memory
from tgi.mdp.heightmap import HeightMap

torch.set_num_threads(2)
WIDTH_Y, GAP_FLOOR = 2.0, -0.8
RES = 0.04


def terrain_rects(kind: str, d: float):
    rects = [(-1.2, 0.8, -WIDTH_Y / 2, WIDTH_Y / 2, 0.0)]
    if kind == "stairs":
        rise, depth, n, x = 0.03 + d * 0.13, 0.35, 10, 0.8
        for i in range(n):
            rects.append((x, x + depth, -WIDTH_Y / 2, WIDTH_Y / 2, rise * (i + 1)))
            x += depth
    elif kind == "stones":
        d_eff = d ** 1.5
        w, g, hmax = 0.5 + d_eff * -0.2, 0.1 + d_eff * 0.1, 0.05 * d_eff
        rng = np.random.default_rng(7)
        x = 0.8
        while x < 5.0:
            y = -WIDTH_Y / 2
            while y < WIDTH_Y / 2 - 1e-6:
                rects.append((x, min(x + w, 5.0), y, min(y + w, WIDTH_Y / 2), rng.uniform(-hmax, hmax)))
                y += w + g
            x += w + g
    return rects


def height_at(rects, x, y):
    for x0, x1, y0, y1, top in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return top
    return GAP_FLOOR if rects else 0.0


class Shim:
    def __init__(self, heights, origin, res):
        self.joint_names = list(JOINT_NAMES)
        self.default_q = torch.tensor(DEFAULT_Q)
        self.action_scale = torch.full((12,), ACTION_SCALE)
        self.kin_model = mujoco.MjModel.from_xml_path(MJCF)
        self._hm = HeightMap(heights, origin, res)

    @property
    def soft_limits(self):
        m = self.kin_model
        return (torch.tensor([m.jnt_range[m.joint(n).id, 0] for n in self.joint_names]),
                torch.tensor([m.jnt_range[m.joint(n).id, 1] for n in self.joint_names]))

    def heightmap(self, device, res=RES):
        return self._hm


def build_model(rects):
    spec = mujoco.MjSpec.from_file(str(root / "tgi/src/tgi/assets/bpx.xml"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 64, 64
    ground = spec.geom("ground")
    ground.pos = [0, 0, GAP_FLOOR]
    for x0, x1, y0, y1, top in rects:
        body = spec.worldbody.add_body()
        body.pos = [(x0 + x1) / 2, (y0 + y1) / 2, (top + GAP_FLOOR) / 2]
        geom = body.add_geom()
        geom.type, geom.size = mujoco.mjtGeom.mjGEOM_BOX, [(x1 - x0) / 2, (y1 - y0) / 2, (top - GAP_FLOOR) / 2]
        geom.friction = [1.0, 0.005, 0.0001]
    return spec.compile()


def run(rects, truth, origin, noise_cm, fov, bundle, offset_cm=0.0):
    h = truth.copy()
    known = np.ones_like(truth, dtype=bool)
    if fov:
        known[:] = False  # a forward camera has seen nothing at t=0
    model = build_model(rects)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0], data.qpos[1], data.qpos[2] = a.start_x, 0.0, NOMINAL_HEIGHT + 0.02 + height_at(rects, a.start_x, 0.0)
    mujoco.mj_forward(model, data)
    from tgi.mdp.interface import FootholdScan
    shim = Shim(torch.from_numpy(truth.copy()), origin, RES)
    use_mem = bool(bundle.args.get("mem", 0))
    mdp = build_mdp(shim, "cpu", reward_cfg=RewardCfg(**{k: v for k, v in bundle.args.items()
                                                          if k in RewardCfg.__dataclass_fields__}), use_mem=use_mem)
    percept = HeightMap(torch.from_numpy(h.copy()), origin, RES)   # degraded map: what the policy reads
    mdp.scan = FootholdScan(percept, "cpu")
    rng = np.random.default_rng(0)
    sigma = noise_cm / 100.0
    mem = init_memory(1, "cpu") if use_mem else None
    hist = None
    cmd = torch.tensor([[a.command, 0.0, 0.0]])
    act_prev = torch.zeros(1, 12)
    xi = torch.zeros(1, 21)
    xi[:, 0], xi[:, -3] = 0.8, 1.0
    bid = model.body("torso").id
    xs = np.arange(origin[0], origin[0] + (truth.shape[0] - 1) * RES + 1e-9, RES)
    ys = np.arange(origin[1], origin[1] + (truth.shape[1] - 1) * RES + 1e-9, RES)
    fell, steps = False, 0
    x0 = float(data.qpos[0])
    with torch.no_grad():
        for step in range(round(a.seconds / CONTROL_DT)):
            mujoco.mj_forward(model, data)
            if fov:  # a cell becomes known once the forward camera could have seen it
                px, py = float(data.qpos[0]), float(data.qpos[1])
                yaw = math.atan2(data.xmat[bid].reshape(3, 3)[1, 0], data.xmat[bid].reshape(3, 3)[0, 0])
                dx, dy = xs[:, None] - px, ys[None, :] - py
                d = np.hypot(dx, dy)
                ang = np.abs(np.arctan2(dy, dx) - yaw)
                ang = np.minimum(ang, 2 * np.pi - ang)
                known |= (d < a.fov_range) & (ang < math.radians(a.fov_half_angle))
            h = np.where(known, truth + offset_cm / 100.0 + rng.normal(0.0, sigma, truth.shape), 0.0)
            percept.h.copy_(torch.from_numpy(h.astype(np.float32)))
            R = data.xmat[bid].reshape(3, 3).copy()
            flat = np.concatenate((data.qpos[:7], data.qvel[:3], R @ data.qvel[3:6],
                                   data.qpos[7:], data.qvel[6:])).astype(np.float32)
            S = unflatten_state(torch.from_numpy(flat)[None])
            K = mdp.context(S)
            if mdp.termination(S, K).item():
                fell = True
                break
            if use_mem and step > 0:
                mem = update_memory(mem, torch.from_numpy(contact(model, data))[None], CONTROL_DT, mdp.cfg.gait_period)
            o = mdp.proprio(S, act_prev, cmd, xi, noise=False)
            hist = append_history(hist, o, bundle.pi.hist)
            fh = bundle.fpm(mdp.system_state(S, cmd)) if bundle.fpm else bootstrap_footholds(1, "cpu")
            extra = perceive(mdp, S, K, fh, noise=False, mem=mem, cmd=cmd)
            act_prev = bundle.pi(hist, extra).clamp(-10, 10)
            target = np.array(DEFAULT_Q) + ACTION_SCALE * act_prev[0].numpy()
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                data.ctrl[:] = np.clip(KP * (target - data.qpos[7:]) - KD * data.qvel[6:], -EFFORT_LIMIT, EFFORT_LIMIT)
                mujoco.mj_step(model, data)
            steps += 1
    dist = float(data.qpos[0]) - x0   # signed displacement along x
    need = 0.8 * abs(a.command) * a.seconds
    return dict(noise_cm=noise_cm, fov=int(fov), fell=fell, seconds=round(steps * CONTROL_DT, 2),
                displacement=round(dist, 2), needed=round(need, 2),
                success=bool((not fell) and abs(dist) >= need))


def contact(model, data):
    bids = [model.body(n).id for n in FOOT_BODIES]
    net, force = np.zeros((4, 3)), np.zeros(6)
    for ci, c in enumerate(data.contact):
        mujoco.mj_contactForce(model, data, ci, force)
        fw = c.frame.reshape(3, 3).T @ force[:3]
        for geom, sign in ((c.geom1, -1), (c.geom2, 1)):
            b = int(model.geom_bodyid[geom])
            if b in bids:
                net[bids.index(b)] += sign * fw
    return (np.linalg.norm(net, axis=1) > 1).astype(np.float32)


rects = terrain_rects(a.terrain, a.difficulty)
xs = np.arange(-2.0, 6.0 + 1e-9, RES)
ys = np.arange(-1.6, 1.6 + 1e-9, RES)
truth = np.array([[height_at(rects, x, y) for y in ys] for x in xs], dtype=np.float32)
origin = (float(xs[0]), float(ys[0]))
bundle = load_policy_bundle(str(Path(a.run) / f"ckpt_{a.iter:06d}.pt"), "cpu")

rows = []
for fov in [int(v) for v in a.fov.split(",")]:
    for noise in [float(v) for v in a.noise_cm.split(",")]:
        for off in [float(v) for v in a.offset_cm.split(",")]:
            r = run(rects, truth, origin, noise, fov, bundle, off)
            r["offset_cm"] = off
            print(json.dumps(r), flush=True)
            rows.append(r)
out = {"terrain": a.terrain, "difficulty": a.difficulty, "command": a.command, "seconds": a.seconds, "rows": rows}
Path(a.out or (root / "logs" / f"degrade_{a.terrain}.json")).write_text(json.dumps(out, indent=2))

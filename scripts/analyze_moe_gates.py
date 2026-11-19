"""Record MoE gate weights during a MuJoCo rollout and report expert specialisation.

Usage:
  python scripts/analyze_moe_gates.py --run logs/moe_scratch_... --iter 10000 \
      --cases forward,turn,reverse --seconds 20 --out logs/moe_gates.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="/home/odroid/bpx-tgi-fpm")
ap.add_argument("--run", required=True)
ap.add_argument("--iter", type=int, default=10000)
ap.add_argument("--cases", default="forward,turn,reverse")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

root = Path(a.project)
import sys
sys.path[:0] = [str(root / "tgi/src"), str(root / "fpm/src")]

from tgi.robot import *  # noqa: F401,F403
from tgi.policy_io import load_policy_bundle, bootstrap_footholds, perceive
from tgi.mdp.mdp import RewardCfg, unflatten_state
from tgi.mdp.history import append_history
from tgi.mdp.memory import init_memory, update_memory

torch.set_num_threads(2)
CMDS = {"forward": [0.3, 0.0, 0.0], "stand": [0.0, 0.0, 0.0], "reverse": [-0.2, 0.0, 0.0], "turn": [0.0, 0.0, 0.3]}


def rollout(bundle, case: str):
    ck = Path(a.run) / f"ckpt_{a.iter:06d}.pt"
    cmd = torch.tensor([CMDS[case]])
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mdp = bundle.make_runner.__self__.make_runner  # placeholder to keep import shape explicit
    runner_mdp = None
    del mdp, runner_mdp
    from tgi.policy_io import build_mdp
    rc = RewardCfg(**{k: v for k, v in bundle.args.items() if k in RewardCfg.__dataclass_fields__})
    mdp = build_mdp(_SimShim(model, data), "cpu", reward_cfg=rc, use_mem=bool(bundle.args.get("mem", 0)))
    use_mem = bool(bundle.args.get("mem", 0))
    mem = init_memory(1, "cpu") if use_mem else None
    hist = None
    act_prev = torch.zeros(1, 12)
    xi = torch.zeros(1, 21)
    xi[:, 0] = 0.8
    xi[:, -3] = 1.0
    bid = model.body("torso").id
    gates, phases, speeds, contacts = [], [], [], []
    fell = False
    with torch.no_grad():
        for step in range(round(a.seconds / CONTROL_DT)):
            mujoco.mj_forward(model, data)
            R = data.xmat[bid].reshape(3, 3).copy()
            flat = np.concatenate((data.qpos[:7], data.qvel[:3], R @ data.qvel[3:6], data.qpos[7:], data.qvel[6:]))
            flat = flat.astype(np.float32)
            S = unflatten_state(torch.from_numpy(flat)[None])
            K = mdp.context(S)
            if mdp.termination(S, K).item():
                fell = True
                break
            if use_mem and step > 0:
                mem = update_memory(mem, torch.from_numpy(_contacts(model, data)).unsqueeze(0), CONTROL_DT,
                                    mdp.cfg.gait_period)
            o = mdp.proprio(S, act_prev, cmd, xi, noise=False)
            hist = append_history(hist, o, bundle.pi.hist)
            fh = bundle.fpm(mdp.system_state(S, cmd)) if bundle.fpm else bootstrap_footholds(1, "cpu")
            extra = perceive(mdp, S, K, fh, noise=False, mem=mem, cmd=cmd)
            latent, w = bundle.pi.encode(hist)
            gates.append(w[0].numpy())
            phases.append(float(mem[0, 12])) if use_mem else phases.append(0.0)
            v_b = R.T @ data.qvel[:3]
            speeds.append(float(np.linalg.norm(v_b[:2])))
            contacts.append(_contacts(model, data))
            act_prev = bundle.pi(hist, extra).clamp(-10, 10)
            target = np.array(DEFAULT_Q) + ACTION_SCALE * act_prev[0].numpy()
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                data.ctrl[:] = np.clip(KP * (target - data.qpos[7:]) - KD * data.qvel[6:], -EFFORT_LIMIT, EFFORT_LIMIT)
                mujoco.mj_step(model, data)
    return dict(gates=np.array(gates), phases=np.array(phases), speeds=np.array(speeds),
                contacts=np.array(contacts), fell=fell)


def _contacts(model, data) -> np.ndarray:
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


class _SimShim:
    """Minimal object for build_mdp (it only needs the MJCF kinematics and joint metadata)."""

    def __init__(self, model, data):
        self.model = model
        self.joint_names = list(JOINT_NAMES)
        self.default_q = torch.tensor(DEFAULT_Q)
        self.action_scale = torch.full((12,), ACTION_SCALE)
        self.kin_model = model
        self.soft_limits = (torch.tensor(model.jnt_range[[model.joint(n).id for n in JOINT_NAMES], 0]),
                            torch.tensor(model.jnt_range[[model.joint(n).id for n in JOINT_NAMES], 1]))

    def heightmap(self, device, res=0.04):
        from tgi.mdp.heightmap import HeightMap
        return HeightMap(torch.zeros(101, 101), (-2.0, -2.0), res)


out = {"checkpoint": str(Path(a.run) / f"ckpt_{a.iter:06d}.pt"), "cases": {}}
bundle = load_policy_bundle(str(Path(a.run) / f"ckpt_{a.iter:06d}.pt"), "cpu")
for case in [c.strip() for c in a.cases.split(",") if c.strip()]:
    r = rollout(bundle, case)
    g = r["gates"]
    if len(g) == 0:
        continue
    ent = -(g * np.log(g + 1e-9)).sum(1)
    argmax = g.argmax(1)
    out["cases"][case] = {
        "steps": int(len(g)), "fell": r["fell"],
        "mean_weight": g.mean(0).round(4).tolist(),
        "std_weight": g.std(0).round(4).tolist(),
        "mean_entropy": float(ent.mean()), "std_entropy": float(ent.std()),
        "argmax_share": [float((argmax == i).mean()) for i in range(g.shape[1])],
        "gate_switches_per_s": float((np.diff(argmax) != 0).mean() / 0.02),
        "sample_weights": g[::max(1, len(g) // 6)][:6].round(3).tolist(),
    }
Path(a.out or (root / "logs" / "moe_gates.json")).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

"""Export a training checkpoint for inference.

Writes ``model_<it>.pt`` (MLP: rsl-rl ActorCritic; MoE: bpx-tgi-moe-export-v1),
``foot_model_<it>.pt`` (TorchScript FPM), and ``exported/`` (policy.ts, policy/FPM ONNX, deploy.json).

    python scripts/export.py --ckpt logs/<run>/ckpt_100000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
import torch.nn as nn

from tgi.robot import NUM_J, NUM_FEET, INTERFACE, KP, KD, EFFORT_LIMIT, SCAN_OFFSET, URDF, LEGS, CONTROL_DT
import xml.etree.ElementTree as ET
from tgi.mdp.interface import (ACTION_SCALE, DEFAULT_POSE, FOOTHOLD_SCAN_DIM, ISAAC_JOINTS, POLICY_OBS_DIM, PROPRIO_DIM,
                               SYSTEM_STATE_DIM)  # fmt: skip
from tgi.policy_io import load_policy_bundle, policy_spec
from tgi.world_model.moe import MoEPolicy

CRITIC_OBS_DIM = PROPRIO_DIM + 33 * 21 + FOOTHOLD_SCAN_DIM  # base env CriticCfg: policy terms + height_scan + foothold_scan
# --scan full: the critic has no foothold_scan
CRITIC_OBS_DIM_FULL = PROPRIO_DIM + 33 * 21


class FlatHistoryActor(nn.Module):
    """MoE导出保持归一化在图内，外部输入为旧到新历史展平后接当前extra。"""
    def __init__(self, pi):
        super().__init__()
        self.pi = pi
        self.hist, self.width = pi.hist, pi.obs_dim
        self.cut = self.hist * self.width

    def forward(self, obs):
        return self.pi(obs[:, :self.cut].reshape(-1, self.hist, self.width), obs[:, self.cut:])


def baked_actor(pi):
    """Raw-observation actor: bake MLP normalization, retain MoE normalization inside the graph."""
    if isinstance(pi, MoEPolicy):
        return FlatHistoryActor(pi).eval()
    mean = torch.cat((pi.obs_norm.mean, pi.extra_norm.mean))
    std = torch.cat((pi.obs_norm.std(), pi.extra_norm.std()))
    layers = [m for m in pi.net]
    first = layers[0]
    lin = nn.Linear(first.in_features, first.out_features)
    with torch.no_grad():
        lin.weight.copy_(first.weight / std)
        lin.bias.copy_(first.bias - (first.weight / std) @ mean)
    return nn.Sequential(lin, *layers[1:]).float().eval()


class _ActorCritic(nn.Module):
    """Parameter layout of rsl-rl-lib 3.3.0's ``ActorCritic`` (scalar std), without importing rsl-rl."""

    def __init__(self, actor: nn.Sequential, std: float, critic_dim: int = CRITIC_OBS_DIM):
        super().__init__()
        self.std = nn.Parameter(std * torch.ones(NUM_J))
        self.actor = actor
        dims, layers = [critic_dim, 512, 256, 128], []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d0, d1), nn.ELU()]
        self.critic = nn.Sequential(*layers, nn.Linear(128, 1))


def rsl_rl_checkpoint(actor: nn.Sequential, iteration: int, std: float = 0.3, critic_dim: int = CRITIC_OBS_DIM) -> dict:
    ac = _ActorCritic(actor, std, critic_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    return {"model_state_dict": ac.state_dict(), "optimizer_state_dict": opt.state_dict(), "iter": iteration,
            "infos": {"source": "Terrain Grounded Imagination (world-model training)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = load_policy_bundle(a.ckpt, "cpu")
    if b.args.get("scan", "foothold") != "foothold" or b.fpm is None or not b.ck.get("fpm_active", False):
        raise ValueError("Export requires an active four-foot FPM; bootstrap checkpoints are not deployable")
    it = int(b.ck["iteration"])
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), f"export_{it:06d}")
    os.makedirs(os.path.join(out, "exported"), exist_ok=True)

    actor = baked_actor(b.pi)
    # the baked actor must match the trained policy
    full = b.fpm is None and b.args.get("scan", "foothold") == "full"
    o, e = torch.randn(64, b.pi.hist, PROPRIO_DIM), torch.randn(64, b.pi.extra_dim) * 0.3
    actor_width = b.pi.hist * PROPRIO_DIM + b.pi.extra_dim
    with torch.no_grad():
        err = (actor(torch.cat((o.flatten(1), e), -1)) - b.pi(o, e)).abs().max().item()
    assert err < 1e-3, err
    if isinstance(b.pi, MoEPolicy):
        torch.save({"format": "bpx-tgi-moe-export-v1", "policy_spec": policy_spec(b.pi),
                    "state_dict": b.pi.state_dict(), "iteration": it}, os.path.join(out, f"model_{it}.pt"))
    else:
        torch.save(rsl_rl_checkpoint(actor, it, critic_dim=CRITIC_OBS_DIM_FULL if full else CRITIC_OBS_DIM),
                   os.path.join(out, f"model_{it}.pt"))
    traced = torch.jit.trace(actor, torch.zeros(1, actor_width))
    torch.jit.save(traced, os.path.join(out, "exported", "policy.ts"))
    if b.fpm is None:  # --scan full (evaluate with FPM_POLICY_SCAN=full) or --fpm 0 (fixed footholds): no foot model
        torch.onnx.export(actor, (torch.zeros(1, actor_width),), os.path.join(out, "exported", "policy.onnx"),
                          input_names=["obs"], output_names=["actions"], dynamo=False)
        kind = "full-height-scan" if full else "fixed-foothold"
        print(f"exported {kind} policy to {out} (baked-actor max err {err:.2e}); no foot model")
        return

    foot = torch.jit.script(b.fpm.exportable().eval())
    torch.jit.save(foot, os.path.join(out, f"foot_model_{it}.pt"))
    s = torch.randn(8, SYSTEM_STATE_DIM)
    assert (torch.jit.load(os.path.join(out, f"foot_model_{it}.pt"))(s) - b.fpm(s).flatten(-2)).abs().max() < 1e-4

    torch.onnx.export(actor, (torch.zeros(1, actor_width),), os.path.join(out, "exported", "policy.onnx"),
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    torch.onnx.export(b.fpm.exportable().eval(), (torch.zeros(1, 1, SYSTEM_STATE_DIM),),
                      os.path.join(out, "exported", "foot_model.onnx"), input_names=["obs"], output_names=["pred"],
                      dynamic_axes={"obs": {0: "batch_size"}, "pred": {0: "batch_size"}}, dynamo=False)
    cfg = {"kp": [KP]*NUM_J, "kd": [KD]*NUM_J, "effort_limit": [EFFORT_LIMIT]*NUM_J,
           "mem": bool(b.args.get("mem", 0)), "gait_period": float(b.args.get("gait_period", 0.6)),
           "stance_fraction": float(b.args.get("stance_fraction", 0.56)),
           "gait_cmd_threshold": float(b.args.get("gait_cmd_threshold", 0.1)),
           "actor_obs_dim": actor_width, "control_dt": float(CONTROL_DT),
           "policy": b.args.get("policy", "mlp"), "history_length": b.pi.hist,
           "proprio_dim": PROPRIO_DIM, "history_order": "oldest_first", "policy_spec": policy_spec(b.pi)}
    joints = {j.get("name"):j for j in ET.parse(URDF).getroot().findall("joint")}
    cfg["joint_lower"] = [float(joints[n].find("limit").get("lower")) for n in ISAAC_JOINTS]
    cfg["joint_upper"] = [float(joints[n].find("limit").get("upper")) for n in ISAAC_JOINTS]
    cfg["foot_order"] = list(LEGS)
    cfg["foot_reference"] = "toe cylinder centre; not the instantaneous contact manifold"
    cfg["scan_offset"] = SCAN_OFFSET
    cfg["fpm_active"] = bool(b.ck.get("fpm_active", True))
    q0 = [next((v for p, v in DEFAULT_POSE.items() if re.fullmatch(p, j)), 0.0) for j in ISAAC_JOINTS]
    cfg.update({
        "interface": INTERFACE,
        "joint_names": ISAAC_JOINTS, "default_q": q0, "action_scale": ACTION_SCALE, "control_dt": 0.02,
        "policy_obs": "base_lin_vel(3), 0.2*base_ang_vel(3), projected_gravity(3), cmd(3), q-q0(12), 0.05*qd(12), "
                      "last_action(12, clip 10), foothold_scan(4 x [foothold_xy(2), 7x7 heights])",
        "fpm_input": "system_state(36): base_lin_vel, base_ang_vel, projected_gravity, cmd, q-q0, qd",
        "foothold_scan": "torso-mounted yaw-aligned 1.6x1.0 m grid, 0.05 m; 7x7 nodes around the nearest node to each "
                         "foothold (x-major); value clip(torso_z - terrain_z - 0.43, -1, 1)",
        "source_ckpt": os.path.abspath(a.ckpt), "iteration": it, "train_hours": b.ck["wall_time"] / 3600,
    })  # fmt: skip
    cfg["policy_input_layout"] = "flattened_proprio_history, current_terrain_scan, gated_phase_if_mem"
    json.dump(cfg, open(os.path.join(out, "exported", "deploy.json"), "w"), indent=1)
    print(f"exported to {out} (baked-actor max err {err:.2e})")


if __name__ == "__main__":
    main()

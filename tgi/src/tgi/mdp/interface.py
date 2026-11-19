"""Canonical BPX observations: 48 proprio + 204 terrain; FPM 36 -> 4x2."""

from __future__ import annotations

import torch

from .rotations import yaw_mat

from ..robot import (NUM_J, NUM_FEET, JOINT_NAMES, DEFAULT_POSE, ACTION_SCALE, ACTION_CLIP,
                     BOOTSTRAP_FOOTHOLDS, SCAN_SIZE, SCAN_RES, PATCH_N, SCAN_OFFSET)
# Historical API name: this is the canonical BPX order, independent of backend ordering.
ISAAC_JOINTS = list(JOINT_NAMES)
PROPRIO_DIM = 12 + 3 * NUM_J
SYSTEM_STATE_DIM = 12 + 2 * NUM_J
FOOTHOLD_SCAN_DIM = NUM_FEET * (2 + PATCH_N**2)
# full height scan: the base env's height_scan grid (1.6 x 1.0 m at 0.05 m, y outer, x inner)
FULL_SCAN_NX, FULL_SCAN_NY = int(round(SCAN_SIZE[0] / SCAN_RES)) + 1, int(round(SCAN_SIZE[1] / SCAN_RES)) + 1
FULL_SCAN_DIM = FULL_SCAN_NX * FULL_SCAN_NY
POLICY_OBS_DIM = PROPRIO_DIM + FOOTHOLD_SCAN_DIM

# Isaac observation noise (uniform half-widths, applied before scaling) and scales
LIN_VEL_NOISE, ANG_VEL_NOISE, GRAV_NOISE, QPOS_NOISE, QVEL_NOISE, SCAN_NOISE = 0.15, 0.2, 0.05, 0.01, 1.5, 0.1
ANG_VEL_SCALE, QVEL_SCALE = 0.2, 0.05
XY_DRIFT, Z_DRIFT = 0.05, 0.05


class JointMap:
    """Permutations between the simulator's (MJCF) joint order and Isaac Lab's order."""

    def __init__(self, mj_joints: list[str], device: str):
        assert sorted(mj_joints) == sorted(ISAAC_JOINTS)
        self.m2i = torch.tensor([mj_joints.index(n) for n in ISAAC_JOINTS], device=device)  # x_isaac = x_mj[m2i]
        self.i2m = torch.tensor([ISAAC_JOINTS.index(n) for n in mj_joints], device=device)  # x_mj = x_isaac[i2m]

    def to_isaac(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.m2i]

    def to_mj(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.i2m]


def _u(x: torch.Tensor, h: float) -> torch.Tensor:
    return (2 * torch.rand_like(x) - 1) * h


def base_quantities(S):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
    w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
    g_b = -R[..., 2, :]
    return v_b, w_b, g_b


def policy_proprio(S, a_prev_isaac, cmd, q0_isaac, jmap: JointMap, bias_mj=None, noise: bool = True) -> torch.Tensor:
    """First 48 entries of the base env's policy observation (Isaac order). ``bias_mj``: encoder bias (sim DR)."""
    q, qd = S[4], S[5]
    v_b, w_b, g_b = base_quantities(S)
    q_rel = jmap.to_isaac(q if bias_mj is None else q + bias_mj) - q0_isaac
    qd_i = jmap.to_isaac(qd)
    if noise:
        spike = (torch.rand_like(v_b[..., :1]) < 1e-4).float()
        v_b = v_b + _u(v_b, LIN_VEL_NOISE) + spike * _u(v_b, 5.0)
        w_b = w_b + _u(w_b, ANG_VEL_NOISE) + (torch.rand_like(w_b[..., :1]) < 1e-4).float() * _u(w_b, 10.0)
        g_b = g_b + _u(g_b, GRAV_NOISE)
        q_rel = q_rel + _u(q_rel, QPOS_NOISE)
        qd_i = qd_i + _u(qd_i, QVEL_NOISE)
    return torch.cat((v_b, ANG_VEL_SCALE * w_b, g_b, cmd, q_rel, QVEL_SCALE * qd_i,
                      a_prev_isaac.clamp(-ACTION_CLIP, ACTION_CLIP)), -1)  # fmt: skip


def system_state(S, cmd, q0_isaac, jmap: JointMap) -> torch.Tensor:
    """FPM input (36): base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel."""
    v_b, w_b, g_b = base_quantities(S)
    return torch.cat((v_b, w_b, g_b, cmd, jmap.to_isaac(S[4]) - q0_isaac, jmap.to_isaac(S[5])), -1)


class FootholdScan:
    """The base env's ``foothold_scan`` term on top of a global heightmap (stand-in for the torso-mounted ray caster)."""

    def __init__(self, hm, device: str):
        self.hm = hm
        self.nx = int(SCAN_SIZE[0] / SCAN_RES) + 1
        self.ny = int(SCAN_SIZE[1] / SCAN_RES) + 1
        k = torch.arange(-(PATCH_N // 2), PATCH_N // 2 + 1, device=device)
        ox, oy = torch.meshgrid(k, k, indexing="ij")
        self.off_x, self.off_y = ox.flatten(), oy.flatten()

    def __call__(self, p, R, torso_p, torso_R, footholds_root, xy_drift=None, z_drift=None, noise: bool = True,
                 return_points: bool = False):
        """footholds_root: (..., 4, 2) in the root frame. Returns (..., 204)."""
        fb = torch.cat((footholds_root, torch.zeros_like(footholds_root[..., :1])), -1)
        feet_w = p.unsqueeze(-2) + (R.unsqueeze(-3) @ fb.unsqueeze(-1)).squeeze(-1)
        rel = feet_w - torso_p.unsqueeze(-2)
        feet_s = (torso_R.transpose(-1, -2).unsqueeze(-3) @ rel.unsqueeze(-1)).squeeze(-1)  # full sensor rotation
        if xy_drift is not None:
            feet_s = torch.cat((feet_s[..., :2] + xy_drift.unsqueeze(-2), feet_s[..., 2:]), -1)
        ax = torch.round((feet_s[..., 0] + SCAN_SIZE[0] / 2) / SCAN_RES).long()
        ay = torch.round((feet_s[..., 1] + SCAN_SIZE[1] / 2) / SCAN_RES).long()
        ix = (ax.unsqueeze(-1) + self.off_x).clamp(0, self.nx - 1)
        iy = (ay.unsqueeze(-1) + self.off_y).clamp(0, self.ny - 1)
        gx = ix.float() * SCAN_RES - SCAN_SIZE[0] / 2
        gy = iy.float() * SCAN_RES - SCAN_SIZE[1] / 2
        Y = yaw_mat(torso_R)[..., :2, :2].unsqueeze(-3).unsqueeze(-3)  # yaw-aligned grid
        pts = torso_p[..., None, None, :2] + (Y @ torch.stack((gx, gy), -1).unsqueeze(-1)).squeeze(-1)
        hz = self.hm.query_nearest(pts)
        h = torso_p[..., 2, None, None] - hz - SCAN_OFFSET
        if z_drift is not None:  # drift before noise and clipping, as in the base env
            h = h + z_drift[..., None, None]
        out = torch.cat((footholds_root, h), -1)  # (..., 2, 2 + 49)
        if noise:
            out = out + _u(out, SCAN_NOISE)
        out = out.clamp(-1.0, 1.0).flatten(-2)
        if return_points:
            return out, pts, hz
        return out


class FullHeightScan:
    """The whole torso-mounted, yaw-aligned height grid (the base env's ``height_scan`` term) instead of FPM-gated
    foothold patches: clip(torso_z - terrain_z - 0.5 + noise, -1, 1) on 33 x 21 nodes, y outer / x inner."""

    def __init__(self, hm, device: str):
        self.hm = hm
        x = torch.arange(FULL_SCAN_NX, device=device).float() * SCAN_RES - SCAN_SIZE[0] / 2
        y = torch.arange(FULL_SCAN_NY, device=device).float() * SCAN_RES - SCAN_SIZE[1] / 2
        gx, gy = torch.meshgrid(x, y, indexing="xy")  # (ny, nx): Isaac Lab's "xy" ordering
        self.grid = torch.stack((gx.flatten(), gy.flatten()), -1)

    def __call__(self, p, R, torso_p, torso_R, footholds_root=None, xy_drift=None, z_drift=None, noise: bool = True,
                 return_points: bool = False):
        g = self.grid
        if xy_drift is not None:  # sensor-frame shift, as the foothold scan's drift
            g = g + xy_drift.unsqueeze(-2)
        Y = yaw_mat(torso_R)[..., :2, :2].unsqueeze(-3)
        pts = torso_p[..., None, :2] + (Y @ g.unsqueeze(-1)).squeeze(-1)
        hz = self.hm.query_nearest(pts)
        h = torso_p[..., 2, None] - hz - SCAN_OFFSET
        if z_drift is not None:
            h = h + z_drift[..., None]
        if noise:
            h = h + _u(h, SCAN_NOISE)
        out = h.clamp(-1.0, 1.0)
        if return_points:
            return out, pts, hz
        return out


def mirror_full_scan(e: torch.Tensor) -> torch.Tensor:
    """Mirror the full height scan left/right: flip the grid along y."""
    return e.reshape(*e.shape[:-1], FULL_SCAN_NY, FULL_SCAN_NX).flip(-2).reshape(*e.shape[:-1], FULL_SCAN_DIM)


# Reflection y -> -y swaps FL/FR and HL/HR; roll changes sign, pitch/knee do not.
FOOT_PERM = [1, 0, 3, 2]
JOINT_PERM = [3 * leg + j for leg in FOOT_PERM for j in range(3)]
JOINT_SIGN = [-1.0, 1.0, 1.0] * NUM_FEET

def mirror_joints(x):
    return x[..., JOINT_PERM] * x.new_tensor(JOINT_SIGN)

def mirror_proprio(o):
    s3 = lambda x, s: x * x.new_tensor(s)
    return torch.cat((s3(o[..., :3], (1,-1,1)), s3(o[..., 3:6], (-1,1,-1)),
                      s3(o[..., 6:9], (1,-1,1)), s3(o[..., 9:12], (1,-1,-1)),
                      mirror_joints(o[..., 12:12+NUM_J]),
                      mirror_joints(o[..., 12+NUM_J:12+2*NUM_J]),
                      mirror_joints(o[..., 12+2*NUM_J:12+3*NUM_J])), -1)

def mirror_scan(e):
    stride = 2 + PATCH_N**2
    feet = e.reshape(*e.shape[:-1], NUM_FEET, stride)[..., FOOT_PERM, :]
    xy = feet[..., :2] * feet.new_tensor([1, -1])
    patch = feet[..., 2:].reshape(*feet.shape[:-1], PATCH_N, PATCH_N).flip(-1).flatten(-2)
    return torch.cat((xy, patch), -1).flatten(-2)

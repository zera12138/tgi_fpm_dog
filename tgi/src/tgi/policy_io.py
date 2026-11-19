"""Closed-loop policy execution in the simulator and checkpoint loading."""

from __future__ import annotations

import torch

from .robot import NUM_FEET, INTERFACE
from .foot_pred import FootholdPredictor
from .mdp import MDP, RobotKinematics, RewardCfg, TerrainFeatures
from .mdp.interface import (BOOTSTRAP_FOOTHOLDS, FOOTHOLD_SCAN_DIM, FULL_SCAN_DIM, PROPRIO_DIM, XY_DRIFT, Z_DRIFT,
                             FootholdScan, FullHeightScan)  # fmt: skip
from .mdp.memory import MEM_DIM, init_memory, update_memory
from .mdp.history import append_history
from .world_model.moe import MoEPolicy

# policy perception: "foothold" (FPM foothold patches) or "full" (whole height scan)
SCAN = {"mode": "foothold"}


def set_scan_mode(mode: str):
    assert mode in ("foothold", "full"), mode
    SCAN["mode"] = mode


def scan_dim() -> int:
    return FULL_SCAN_DIM if SCAN["mode"] == "full" else FOOTHOLD_SCAN_DIM
from .world_model import Policy


def build_mdp(sim, device: str, reward_cfg: RewardCfg | None = None, field_window: float = 0.5,
              field_hole: float = 0.25, hm_res: float = 0.04, hm_exact: bool = False,
              use_mem: bool = False) -> MDP:
    """Terrain mesh -> heightmap (+ support field), MJCF kinematics, MJCF-ordered limits."""
    hm = sim.heightmap(device, res=hm_res)
    hm.exact_support = hm_exact
    hm.build_support_field(window=field_window, hole_depth=field_hole)
    lo, hi = sim.soft_limits
    terrain = TerrainFeatures(hm, device)
    kin = RobotKinematics(sim.kin_model, sim.joint_names, device)
    mdp = MDP(kin, terrain, sim.default_q, sim.joint_names, lo, hi, sim.action_scale, reward_cfg or RewardCfg(),
              use_mem=use_mem)
    mdp.scan = FullHeightScan(hm, device) if SCAN["mode"] == "full" else FootholdScan(hm, device)
    return mdp


def bootstrap_footholds(n: int, device: str) -> torch.Tensor:
    return torch.tensor(BOOTSTRAP_FOOTHOLDS, device=device).expand(n, NUM_FEET, 2)


def perceive(mdp: MDP, S, K, fh, xy_drift=None, z_drift=None, noise: bool = True, return_points: bool = False,
             mem=None, cmd=None):
    """Foothold patches (+ gait phase when memory is on, gated exactly like the phase/air rewards)."""
    out = mdp.scan(S[0], S[1], K["torso_p"], K["torso_R"], fh, xy_drift, z_drift, noise, return_points)
    phase = None
    if mem is not None:
        phase = mem[..., 12:14]
        if cmd is not None:  # standing commands see no clock, matching the reward gate
            moving = (cmd[..., :2].norm(dim=-1) + cmd[..., 2].abs()) > mdp.cfg.gait_cmd_threshold
            phase = phase * moving.to(phase.dtype).unsqueeze(-1)
    if return_points:
        extra, pts, hz = out
        return (extra if phase is None else torch.cat((extra, phase), -1)), pts, hz
    return out if phase is None else torch.cat((out, phase), -1)


def extra_dim(mem: bool = False) -> int:
    return scan_dim() + (2 if mem else 0)


def make_policy(device: str, mem: bool = False, policy: str = "mlp", history_length: int = 5) -> Policy:
    if policy == "moe":
        return MoEPolicy(PROPRIO_DIM, hist=history_length, extra_dim=extra_dim(mem)).to(device)
    if policy != "mlp":
        raise ValueError(f"Unknown policy architecture: {policy}")
    return Policy(PROPRIO_DIM, hist=1, extra_dim=extra_dim(mem)).to(device)


def policy_spec(pi):
    return {"version": 1, "kind": "moe" if isinstance(pi, MoEPolicy) else "mlp",
            "history_length": pi.hist, "obs_dim": pi.obs_dim, "extra_dim": pi.extra_dim,
            "experts": getattr(pi, "expert_count", 0), "latent_dim": getattr(pi, "latent_dim", 0)}


class SimRunner:
    def __init__(self, pi: Policy, fpm: FootholdPredictor | None, mdp: MDP, sim, noise: bool = True, mem: bool = False):
        self.pi, self.fpm, self.mdp, self.sim, self.noise = pi, fpm, mdp, sim, noise
        self.use_mem = mem
        self.mem = init_memory(sim.N, mdp.default_q.device) if mem else None
        dev = sim.device
        self.use_fpm_model = fpm is not None
        self.xy_drift = torch.zeros(sim.N, 2, device=dev)
        self.z_drift = torch.zeros(sim.N, device=dev)
        self.footholds = bootstrap_footholds(sim.N, dev).clone()
        self.vis = None
        self.obs_history = None
        self._observation = None

    def _resample_drift(self, ids):
        if not self.noise:
            return
        self.xy_drift[ids] = (2 * torch.rand(len(ids), 2, device=self.sim.device) - 1) * XY_DRIFT
        self.z_drift[ids] = (2 * torch.rand(len(ids), device=self.sim.device) - 1) * Z_DRIFT

    @torch.no_grad()
    def observe(self):
        sim, mdp = self.sim, self.mdp
        if self._observation is not None:
            return self._observation
        S = sim.state()
        cmd = sim.command()
        if self.noise:  # new episodes redraw their perception drift (masked: no host sync)
            m = sim.ep_len == 0
            self.xy_drift = torch.where(m.unsqueeze(-1), (2 * torch.rand_like(self.xy_drift) - 1) * XY_DRIFT, self.xy_drift)
            self.z_drift = torch.where(m, (2 * torch.rand_like(self.z_drift) - 1) * Z_DRIFT, self.z_drift)
        if self.mem is not None:
            fresh = sim.ep_len == 0
            if bool(fresh.any()):
                self.mem = torch.where(fresh.unsqueeze(-1), init_memory(sim.N, self.mem.device), self.mem)
        o = mdp.proprio(S, sim.a_prev, cmd, sim.xi(), noise=self.noise)
        if self.fpm is not None and self.use_fpm_model:
            fh = self.fpm(mdp.system_state(S, cmd))
        else:
            fh = bootstrap_footholds(sim.N, sim.device)
        self.footholds = fh
        K = mdp.context(S)
        extra, pts, hz = perceive(mdp, S, K, fh, self.xy_drift, self.z_drift, self.noise, return_points=True,
                                  mem=self.mem, cmd=cmd)
        self.vis = {"pts": pts, "hz": hz}
        self.obs_history = append_history(self.obs_history, o, self.pi.hist, reset=sim.ep_len == 0)
        self._observation = (o, extra)
        return self._observation

    @torch.no_grad()
    def step(self, a: torch.Tensor) -> dict:
        """One simulator step keeping the task memory aligned with ``s_t`` (``mem``) and ``s_{t+1}``."""
        if self.pi.hist > 1 and self._observation is None:
            raise RuntimeError("Call observe/act before runner.step so the replay history exists")
        out = self.sim.step(a)
        self._observation = None
        if self.pi.hist > 1:
            out["obs_hist"] = self.obs_history
        if self.mem is not None:
            out["mem"] = self.mem
            self.mem = update_memory(self.mem, out["c1"], self.sim.dt, self.mdp.cfg.gait_period)
        return out

    def reset(self):
        """外部显式重置仿真后调用；正常done由下一次observe逐环境处理。"""
        self._observation = None
        self.obs_history = None
        if self.use_mem:
            self.mem = init_memory(self.sim.N, self.mdp.default_q.device)

    @torch.no_grad()
    def act(self):
        """Returns actions in the simulator's (MJCF) joint order."""
        o, e = self.observe()
        return self.mdp.jmap.to_mj(self.pi(self.obs_history, e))


class Bundle:
    """Policy and FPM of a training checkpoint (``ckpt_<it>.pt``)."""

    def __init__(self, ck: dict, device: str):
        if ck.get("interface") != INTERFACE:
            raise ValueError("Checkpoint is not a BPX TGI v1 checkpoint; G1 weights are incompatible")
        self.ck, self.args = ck, ck["args"]
        set_scan_mode(self.args.get("scan", "foothold"))
        self.use_mem = bool(self.args.get("mem", 0))
        self.pi = make_policy(device, mem=self.use_mem, policy=self.args.get("policy", "mlp"),
                              history_length=self.args.get("history_length", 5))
        if self.args.get("policy", "mlp") == "moe" and ck.get("policy_spec") != policy_spec(self.pi):
            raise ValueError("MoE checkpoint policy_spec does not match its configuration")
        self.pi.load_state_dict(ck["pi"])
        self.pi.eval()
        self.fpm = None
        if ck.get("fpm") is not None:
            self.fpm = FootholdPredictor().to(device)
            self.fpm.load_state_dict(ck["fpm"])
            self.fpm.eval()
        self.device = device

    def make_runner(self, sim, noise: bool = True):
        a = self.args
        reward_cfg = RewardCfg(**{k: v for k, v in a.items() if k in RewardCfg.__dataclass_fields__})
        mdp = build_mdp(sim, self.device, reward_cfg=reward_cfg, field_window=a.get("field_window", 0.5), field_hole=a.get("field_hole", 0.25),
                        hm_res=a.get("hm_res", 0.04), hm_exact=bool(a.get("hm_exact", 0)), use_mem=self.use_mem)
        r = SimRunner(self.pi, self.fpm, mdp, sim, noise, mem=self.use_mem)
        r.use_fpm_model = self.fpm is not None and self.ck.get("fpm_active", True)
        return r


def load_policy_bundle(path: str, device: str) -> Bundle:
    return Bundle(torch.load(path, map_location=device, weights_only=False), device)

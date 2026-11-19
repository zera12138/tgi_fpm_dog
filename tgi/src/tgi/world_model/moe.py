# Copyright (c) 2026-2027 zh
# Adapted from quadruped_lab d1a9911, agents/supermimic/policy.py.
# MIT license and original copyright: licenses/quadruped_lab_MIT.txt.
"""本体历史的8专家编码器；保留TGI当前帧、地形与相位，不增加速度估计头。"""
import torch
from torch import nn
from torch.nn import functional as F
from .networks import Policy, RunningNorm, mlp, run_mlp
from ..constants import NUM_J


class Experts(nn.Module):
    """共享骨干后使用分组1x1卷积，保留参考的专家参数划分。"""
    def __init__(self, count, width, latent):
        super().__init__()
        self.count, self.latent = count, latent
        self.backbone = nn.Sequential(*mlp(width, [512, 256], count * 128), nn.ELU())
        self.experts = nn.Conv1d(count * 128, count * latent, 1, groups=count)

    def forward(self, x):
        x = self.experts(self.backbone(x).unsqueeze(-1)).squeeze(-1)
        return x.reshape(-1, self.count, self.latent)


class MoEPolicy(Policy):
    """history按最旧到最新排列；潜特征放动作头末尾，便于旧MLP等价迁移。"""
    def __init__(self, obs_dim, hist=5, extra_dim=204, experts=8, latent=32):
        nn.Module.__init__(self)
        if hist < 2 or experts < 2 or latent < 1:
            raise ValueError('MoE needs history >= 2, experts >= 2 and latent >= 1')
        self.hist, self.obs_dim, self.extra_dim = hist, obs_dim, extra_dim
        self.expert_count, self.latent_dim = experts, latent
        self.obs_norm = RunningNorm(obs_dim)
        self.extra_norm = RunningNorm(max(extra_dim, 1))
        self.history_norm = RunningNorm(obs_dim * hist)
        self.experts = Experts(experts, obs_dim * hist, latent)
        self.gating_network = nn.Sequential(mlp(obs_dim * hist, [512, 256], experts), nn.Softmax(dim=-1))
        self.net = mlp(obs_dim + extra_dim + latent, [512, 256, 128], NUM_J)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def encode(self, obs_hist):
        x = self.history_norm(obs_hist.flatten(1))
        weights = self.gating_network(x)
        features = (weights.unsqueeze(-1) * self.experts(x)).sum(1)
        return F.normalize(features, dim=-1), weights

    def forward(self, obs_hist, extra):
        latent, _ = self.encode(obs_hist)
        x = torch.cat((self.obs_norm(obs_hist[:, -1]), self.extra_norm(extra), latent), -1)
        return run_mlp(self.net, x)

    @torch.no_grad()
    def update_statistics(self, obs_hist, extra):
        self.obs_norm.update(obs_hist[:, -1])
        self.extra_norm.update(extra)
        self.history_norm.update(obs_hist.flatten(1))


def migrate_mlp_to_moe(model, old):
    """旧MLP的动作函数严格保留；新历史支路初始对动作贡献为零。"""
    state = model.state_dict()
    old_extra = old['extra_norm.mean'].numel()
    if old_extra > model.extra_dim or old['net.0.weight'].shape[1] != model.obs_dim + old_extra:
        raise ValueError('MLP observation layout is incompatible with the requested MoE')
    for key, value in old.items():
        if key not in state:
            raise ValueError(f'Unexpected MLP checkpoint tensor: {key}')
        if key == 'net.0.weight':
            state[key] = torch.zeros_like(state[key])
            state[key][:, :value.shape[1]] = value
        elif key in ('extra_norm.mean', 'extra_norm.var'):
            state[key] = torch.zeros_like(state[key]) if key.endswith('mean') else torch.ones_like(state[key])
            state[key][:old_extra] = value
        elif state[key].shape == value.shape:
            state[key] = value
        else:
            raise ValueError(f'Cannot migrate MLP tensor {key}: {value.shape} -> {state[key].shape}')
    if old_extra != model.extra_dim:
        state['extra_norm.count'] = torch.full_like(state['extra_norm.count'], 1e4)
    state['history_norm.mean'] = old['obs_norm.mean'].repeat(model.hist)
    state['history_norm.var'] = old['obs_norm.var'].repeat(model.hist)
    state['history_norm.count'] = torch.full_like(state['history_norm.count'], 1e4)
    model.load_state_dict(state)

"""Task memory (per-foot contact state, contact/air timers, gait clock) for TGI.

Layout (``MEM_DIM = 14``)::

    [0:4]   contact probability (0/1 from the contact sensor, sigmoid(logit) in imagination)
    [4:8]   probability-weighted air time [s]
    [8:12]  probability-weighted stance time [s]
    [12]    sin(phase)
    [13]    cos(phase)

The timer recursion is the continuous one from the reference SuperMimic task
(``go2_supermimic/mdp/reference_task.py:advance``), not a thresholded state machine::

    contact_stay = min(prev, c) / prev
    air_stay     = min(1 - prev, 1 - c) / (1 - prev)
    air_time     = air_stay * air_time + (1 - c) * dt
    contact_time = contact_stay * contact_time + c * dt

so a change of 1e-6 in the predicted contact probability changes the timers by 1e-6.
Conditional durations are recovered at scoring time by dividing out the probability
(``air_time / (1 - c)``, ``contact_time / c``), which is what the reference rewards do.

The clock is stored as sin/cos and advanced by ``dt / period`` radians per step, so it
never saturates or freezes regardless of episode length.
"""

from __future__ import annotations

import math

import torch

NUM_FEET = 4
GAIT_OFFSETS = (0.0, 0.5, 0.5, 0.0)  # fl, fr, hl, hr -> diagonal trot
MEM_DIM = 3 * NUM_FEET + 2  # contact, air, stance, sin, cos

_CONTACT = slice(0, 4)
_AIR = slice(4, 8)
_STANCE = slice(8, 12)
_SIN, _COS = 12, 13


def init_memory(n: int, device) -> torch.Tensor:
    mem = torch.zeros(n, MEM_DIM, device=device)
    mem[:, _COS] = 1.0
    return mem


def update_memory(mem: torch.Tensor, contact: torch.Tensor, dt: float, period: float,
                  reset: torch.Tensor | None = None) -> torch.Tensor:
    """One control step; ``contact`` is (..., 4), probability or 0/1."""
    c = contact.clamp(0.0, 1.0)
    prev = mem[..., _CONTACT].clamp(0.0, 1.0)
    contact_stay = torch.minimum(prev, c) / prev.clamp_min(1e-6)
    air_stay = torch.minimum(1.0 - prev, 1.0 - c) / (1.0 - prev).clamp_min(1e-6)
    air = air_stay * mem[..., _AIR] + (1.0 - c) * dt
    stance = contact_stay * mem[..., _STANCE] + c * dt

    ang = torch.atan2(mem[..., _SIN], mem[..., _COS]) + 2 * math.pi * dt / period
    ang = torch.remainder(ang, 2 * math.pi)
    out = torch.cat((c, air, stance, torch.sin(ang).unsqueeze(-1), torch.cos(ang).unsqueeze(-1)), -1)
    if reset is not None and bool(reset.any()):
        out = torch.where(reset.unsqueeze(-1), init_memory(out.shape[0], out.device), out)
    return out


def gait_phase(mem: torch.Tensor) -> torch.Tensor:
    """Phase in [0, 1) from the stored sin/cos pair."""
    return torch.remainder(torch.atan2(mem[..., _SIN], mem[..., _COS]) / (2 * math.pi), 1.0)


def gait_targets(mem: torch.Tensor, stance_fraction: float, offsets=GAIT_OFFSETS) -> torch.Tensor:
    """Desired stance flags (..., 4) for the clock stored in ``mem``."""
    phase = gait_phase(mem).unsqueeze(-1)
    off = torch.tensor(offsets, device=mem.device, dtype=mem.dtype)
    return (torch.remainder(phase + off, 1.0) < stance_fraction).to(mem.dtype)


def durations(mem: torch.Tensor, contact: torch.Tensor):
    """Conditional air/stance durations [s] (reference: divide the probability-weighted timers)."""
    air = mem[..., _AIR] / (1.0 - contact).clamp_min(1e-6)
    stance = mem[..., _STANCE] / contact.clamp_min(1e-6)
    return air, stance


def duration_scores(mem: torch.Tensor, contact: torch.Tensor, air_target: float, stance_target: float):
    """(swing_score, stance_score) in [0, 1], 1 when the duration matches the target."""
    air, stance = durations(mem, contact)
    swing = (1.0 - (air / air_target - 1.0).abs()).clamp_min(0.0)
    stance_s = (1.0 - (stance / stance_target - 1.0).abs()).clamp_min(0.0)
    return swing, stance_s

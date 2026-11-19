"""策略本体历史的统一顺序及回合重置：最旧到最新，首帧重复填满。"""
import torch


def append_history(history, observation, length, reset=None):
    current = observation.unsqueeze(1)
    if history is None:
        return current.expand(-1, length, -1).clone()
    out = torch.cat((history[:, 1:], current), dim=1)
    if reset is not None:
        out = torch.where(reset[:, None, None], current.expand(-1, length, -1), out)
    return out

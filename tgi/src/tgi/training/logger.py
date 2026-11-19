"""Training statistics (``train_log.csv`` and console), accumulated on the GPU and read back only when logging."""

from __future__ import annotations

import os
import time
import json

import torch

COLUMNS = ("iteration,wall_h,sim_steps,wm_loss,wm_err8,contact_acc,fpm_loss,pi_loss,v_loss,imag_rew,pi_gradnorm,"
           "imag_survival,sim_rew,vel_err,ep_len,ep_rew,term_frac,terrain_level,it_per_s,lr")
N_TERRAIN_TYPES = 64


def _mean(xs) -> float:
    return float(sum(float(v) for v in xs) / max(len(xs), 1))


class TrainLogger:
    def __init__(self, log_dir: str, resume: bool, log_every: int, terrain_names: list[str], device: str,
                 total_it: int = 0, run_name: str = "", action_noise: float = 0.0):
        self.log_every = log_every
        self.total_it = total_it
        self.start_it = 0
        self.run_name = run_name or os.path.basename(os.path.normpath(log_dir))
        self.action_noise = action_noise
        self.t0 = time.time()
        self.terrain_names = terrain_names
        self.csv = open(os.path.join(log_dir, "train_log.csv"), "a")
        if self.csv.tell() == 0:
            self.csv.write(COLUMNS + "\n")
        self.extra_path = os.path.join(log_dir, "extra_metrics.jsonl")
        self.sim = {k: [] for k in ("r", "vel_err", "ep_len", "ep_ret", "term", "done", "level")}
        self.upd = {k: [] for k in ("wm", "wm8", "acc", "fpm", "pi", "v", "ir", "gn", "surv")}
        self.ep_terms: dict[str, list] = {}
        # per terrain type: episodes, terminations, summed level, summed length
        self.terrain = torch.zeros(N_TERRAIN_TYPES, 4, device=device)
        self.t_log = time.time()
        self._last_steps, self._last_steps_t = 0, time.time()

    def add_transition(self, out: dict, S1, dt: float):
        """One simulator step (``IsaacLocoSim.step`` output and the next state)."""
        self.sim["r"].append(out["r_sim"].mean() / dt)
        Rt = S1[1].transpose(-1, -2)
        v_b = (Rt @ S1[2].unsqueeze(-1)).squeeze(-1)
        self.sim["vel_err"].append((v_b[:, :2] - out["cmd"][:, :2]).norm(dim=-1).mean())
        if "ep_len" not in out:
            return
        self.sim["ep_len"].append(out["ep_len"].mean())
        self.sim["ep_ret"].append((out["ep_ret"] / out["ep_len"].clamp(min=1) / dt).mean())
        self.sim["term"].append(out["term"].float().sum())
        self.sim["done"].append(out["done"].float().sum())
        for k, v in out["episode_log"].items():
            self.ep_terms.setdefault(k, []).append(v)
        ty = out["ep_type"].long().clamp(0, N_TERRAIN_TYPES - 1)
        tm = out["term"][out["ep_ids"]].float()
        rows = torch.stack((torch.ones_like(tm), tm, out["ep_level"].float(), out["ep_len"].float()), -1)
        self.terrain.index_add_(0, ty, rows)

    def add_level(self, level: torch.Tensor):
        self.sim["level"].append(level)

    def add_wm(self, loss, err8, acc):
        self.upd["wm"].append(loss), self.upd["wm8"].append(err8), self.upd["acc"].append(acc)

    def add_fpm(self, loss):
        self.upd["fpm"].append(loss)

    def add_policy(self, pi_loss, v_loss, imag_rew, grad_norm, survival):
        for k, v in zip(("pi", "v", "ir", "gn", "surv"), (pi_loss, v_loss, imag_rew, grad_norm, survival)):
            self.upd[k].append(v)

    def write(self, it: int, wall_h: float, sim_steps: int, lr: float, extra: dict[str, float], sim=None):
        """Log line for iteration ``it`` (1-based); ``extra``: further values for the console; ``sim``: curriculum source."""
        ips = self.log_every / (time.time() - self.t_log)
        self.t_log = time.time()
        s, u = self.sim, self.upd
        term_frac = float(sum(float(v) for v in s["term"]) / max(sum(float(v) for v in s["done"]), 1))
        row = [it, wall_h, sim_steps, _mean(u["wm"]), _mean(u["wm8"]), _mean(u["acc"]), _mean(u["fpm"]), _mean(u["pi"]),
               _mean(u["v"]), _mean(u["ir"]), _mean(u["gn"]), _mean(u["surv"]), _mean(s["r"]), _mean(s["vel_err"]),
               _mean(s["ep_len"]), _mean(s["ep_ret"]), term_frac, _mean(s["level"]), ips, lr]  # fmt: skip
        self.csv.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        self.csv.flush()
        if extra:
            with open(self.extra_path, "a") as stream:
                stream.write(json.dumps({"iteration": it, **extra}) + "\n")
        self._print_block(it, row, ips, sim_steps, term_frac, extra)
        if it % (4 * self.log_every) == 0 and self.ep_terms:
            self._print_episodes()
            if sim is not None:
                self._print_curriculum(sim, profile=it % (20 * self.log_every) == 0)
        self.sim = {k: [] for k in self.sim}
        self.upd = {k: [] for k in self.upd}

    def _print_block(self, it: int, row: list, ips: float, sim_steps: int, term_frac: float, extra: dict) -> None:
        now = time.time()
        steps_per_s = (sim_steps - self._last_steps) / max(now - self._last_steps_t, 1e-6)
        self._last_steps, self._last_steps_t = sim_steps, now
        """Console block in the rsl_rl layout used by go2_rl_robotlab (banner, aligned fields, footer with ETA)."""
        width, pad = 80, 30
        elapsed = time.time() - self.t0
        remaining = max(self.total_it - it, 0) if self.total_it else 0
        done = max(it - self.start_it, 1)  # only this process's iterations, so a resume does not skew the ETA
        eta = elapsed / done * remaining
        lines = ["#" * width, f"\033[1m{f' Learning iteration {it}/{self.total_it} '.center(width)}\033[0m ", ""]
        lines.append(f"{'Run name:':>{pad}} {self.run_name}")
        lines.append(f"{'Total steps:':>{pad}} {sim_steps}")
        lines.append(f"{'Steps per second:':>{pad}} {steps_per_s:.0f}")
        for name, value in (("world model", row[3]), ("foothold predictor", row[6]), ("policy", row[7]),
                            ("value", row[8])):
            lines.append(f"{f'Mean {name} loss:':>{pad}} {value:.4f}")
        if "sym_loss" in extra:
            lines.append(f"{'Mean symmetry loss:':>{pad}} {extra['sym_loss']:.4f}")
        lines.append(f"{'Mean reward:':>{pad}} {row[12]:.2f}")
        lines.append(f"{'Mean episode length:':>{pad}} {row[14]:.2f}")
        lines.append(f"{'Mean action noise std:':>{pad}} {self.action_noise:.2f}")
        lines.append(f"{'Mean velocity error:':>{pad}} {row[13]:.3f}")
        lines.append(f"{'Mean survival:':>{pad}} {row[11]:.3f}")
        lines.append(f"{'Mean terrain level:':>{pad}} {row[17]:.2f}")
        lines.append(f"{'Mean termination fraction:':>{pad}} {term_frac:.2f}")
        short = {k: k.split("/")[-1] for k in self.ep_terms}
        dupes = {n for n in short.values() if list(short.values()).count(n) > 1}
        if self.ep_terms:  # one line instead of one line per reward term
            parts = [(k if short[k] in dupes else short[k], float(sum(v)) / len(v)) for k, v in self.ep_terms.items()]
            lines.append(f"{'Mean episode:':>{pad}} " + " ".join(f"{k}={v:.3f}" for k, v in parts))
        rest = {k: v for k, v in extra.items() if k != "sym_loss"}
        if rest:
            lines.append(f"{'Mean extra:':>{pad}} " + " ".join(f"{k}={v:.3f}" for k, v in rest.items()))
        lines.append("-" * width)
        lines.append(f"{'Iteration time:':>{pad}} {1.0 / ips:.2f}s")
        lines.append(f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        lines.append(f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta)) if self.total_it else '-'}")
        print("\n".join(lines), flush=True)

    def _name(self, ty: int) -> str:
        return self.terrain_names[ty] if ty < len(self.terrain_names) else str(ty)

    def _print_episodes(self):
        self.ep_terms.clear()  # already printed as "Mean episode ..." lines in the main block
        by_name: dict[str, list] = {}
        for ty, (n, tm, lv, ln) in enumerate(self.terrain.tolist()):
            if n > 0:
                b = by_name.setdefault(self._name(ty), [0, 0, 0.0, 0.0])
                b[0] += n
                b[1] += tm
                b[2] += lv
                b[3] += ln
        self.terrain.zero_()
        if by_name:
            print("   terrain: " + " ".join(f"{k}(n={n:.0f} fall={tm / max(n, 1):.2f} lvl={lv / max(n, 1):.1f} "
                                            f"len={ln / max(n, 1):.0f})" for k, (n, tm, lv, ln) in sorted(by_name.items())),
                  flush=True)

    def _print_curriculum(self, sim, profile: bool):
        print("   sgs: " + " ".join(f"{k}={v:.3f}" for k, v in sim.sgs_stats().items()), flush=True)
        if profile:
            for ty, (succ, share, cnt) in sim.sgs_profile().items():
                print(f"   sgs {self._name(ty):22s} success by level " + " ".join(f"{v:.2f}" for v in succ.tolist())
                      + " | share " + " ".join(f"{v:.2f}" for v in share.tolist())
                      + " | visits/config " + " ".join(f"{v:.0f}" for v in cnt.tolist()), flush=True)

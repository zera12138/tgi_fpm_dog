"""Same-trajectory perception check for the level-7 yard.

The policy reads the *truth* heightmap here, so the robot walks one fixed trajectory. Along that
trajectory two perception maps are rebuilt from rendered depth:

  single  -- only what the current frame sees (rebuilt from scratch every step)
  history -- fused from an empty map, never seeded with terrain truth

Group D (pose error, latency, dropouts) is not part of this script; it builds on the same maps.

Usage:
  python verify_perception.py --pitch 15,30 --spawn 1 --seconds 12 --command 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/odroid/bpx-tgi-fpm")
sys.path[:0] = [str(PROJECT / "tgi/src"), str(PROJECT / "fpm/src")]

from perception import CameraSpec, DepthCamera, HeightMap, build_model, support_plane  # noqa: E402
from tgi.deploy.controller import Controller, quat_to_mat  # noqa: E402

FEET = ["FL", "FR", "HL", "HR"]
PATCH = 7
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Truth:
    """Raytraced truth map: physics reference and error metric, never given to the policy as a map."""

    def __init__(self, path):
        arc = np.load(path)
        self.h = arc["heights"].astype(np.float32)
        self.origin = arc["origin"].astype(float)
        self.res = float(arc["res"])

    def __call__(self, x, y):
        ij = np.rint((np.array([x, y]) - self.origin) / self.res).astype(int)
        if (ij < 0).any() or ij[0] >= self.h.shape[0] or ij[1] >= self.h.shape[1]:
            return -0.8
        return float(self.h[ij[0], ij[1]])

    def at(self, xs, ys):
        return np.array([self(x, y) for x, y in zip(np.ravel(xs), np.ravel(ys))], np.float32)


class Runner:
    def __init__(self, cam: CameraSpec, truth: Truth, spawn: int, command):
        self.cam, self.truth = cam, truth
        # Physics never sees the mount pitch: both brackets must walk the *same* trajectory.
        self.model = build_model(HERE / "scene/level7.xml", cam, apply_pitch=False)
        self.data = mujoco.MjData(self.model)
        self.vmodel = build_model(HERE / "scene/level7.xml", cam, apply_pitch=True)
        self.vdata = mujoco.MjData(self.vmodel)
        assert (self.model.nq, self.model.nv) == (self.vmodel.nq, self.vmodel.nv)
        self.ctrl = Controller(str(HERE), height_fn=truth)
        ids = [self.model.joint(n).id for n in self.ctrl.names]
        self.qidx = [self.model.jnt_qposadr[j] for j in ids]
        self.vidx = [self.model.jnt_dofadr[j] for j in ids]
        self.actuators = [int(np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0]) for j in ids]
        self.substeps = round(self.ctrl.dt / self.model.opt.timestep)
        self.command = np.asarray(command, float)
        self.spawn = spawn
        self.cam_rig = DepthCamera(self.vmodel, cam)
        self.reset(spawn)
        self.plane = support_plane(self.cam_rig, self.vdata, cam)

    def reset(self, index):
        info = json.loads((HERE / "scene/scene.json").read_text())
        sp = info["spawns"][index]
        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, 0)
        d.qpos[:3] = [*sp["xy"], sp["z"] + 0.43]
        d.qpos[3:7] = [math.cos(sp["yaw"] / 2), 0, 0, math.sin(sp["yaw"] / 2)]
        d.qpos[self.qidx] = self.ctrl.q0
        d.qvel[:] = 0
        d.ctrl[:] = 0
        self.ctrl.reset()
        mujoco.mj_forward(self.model, d)
        self.sync_view()

    def sync_view(self):
        """Mirror the physics state into the render model so the camera sees this instant."""
        self.vdata.qpos[:] = self.data.qpos
        self.vdata.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.vmodel, self.vdata)

    def step(self):
        d = self.data
        R = quat_to_mat(d.qpos[3:7])
        vb = R.T @ d.qvel[:3]
        target = self.ctrl.step(d.qpos[:3].copy(), d.qpos[3:7].copy(), vb, d.qvel[3:6].copy(),
                                d.qpos[self.qidx].copy(), d.qvel[self.vidx].copy(), self.command)
        for _ in range(self.substeps):
            torque = np.clip(self.ctrl.kp * (target - d.qpos[self.qidx]) - self.ctrl.kd * d.qvel[self.vidx],
                             -self.ctrl.eff, self.ctrl.eff)
            d.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, d)
        self.sync_view()

    def contacts(self):
        bids = [self.model.body(n).id for n in ("fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link")]
        hits = np.zeros(4)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            for g, s in ((c.geom1, -1), (c.geom2, 1)):
                b = int(self.model.geom_bodyid[g])
                if b in bids:
                    hits[bids.index(b)] += abs(s)
        return hits > 0


def foot_metrics(maps, step, dt, fs, hist_map=None, fill_limit=0.6):
    """(coverage, median age s, mean |height error| m, fill fraction) per foot."""
    rows = []
    for f in range(4):
        pts = fs[f]
        out = {"foot": FEET[f], "points": len(pts)}
        for name, hm in maps.items():
            h, seen, last = hm.sample(pts[:, 0], pts[:, 1])
            cov = float(seen.mean())
            ages = (step - last[seen]) * dt if seen.any() else None
            err = None
            if seen.any():
                err = float(np.mean(np.abs(h[seen] - TRUTH.at(pts[seen, 0], pts[seen, 1]))))
            out[name] = {"coverage": round(cov, 4),
                         "age_median_s": None if ages is None else round(float(np.median(ages)), 3),
                         "age_max_s": None if ages is None else round(float(ages.max()), 3),
                         "height_err_m": err if err is None else round(err, 4),
                         "fill_plane": round(1.0 - cov, 4)}
        if hist_map is not None:
            out["fill_split"] = hist_map.fill_report(pts[:, 0], pts[:, 1], fill_limit)
        rows.append(out)
    return rows


def draw(canvas, step, dt, hist, single, fs, robot_xy, truth, title):
    """Four panels: truth, current-frame depth, fused history, age. One patch box per foot."""
    lo = robot_xy - np.array([2.0, 1.6])
    hi = robot_xy + np.array([3.4, 1.6])
    img_ext = [truth.origin[1], truth.origin[1] + truth.h.shape[1] * truth.res,
               truth.origin[0], truth.origin[0] + truth.h.shape[0] * truth.res]
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("0.86")  # unobserved ground is a flat grey, never a height
    vmin, vmax = -0.05, 1.10

    axes = canvas.axes
    for ax in axes:
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(lo[0], hi[0])
        ax.set_aspect("equal")
        ax.set_xlabel("y / lateral (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18, lw=0.4)

    axes[0].imshow(truth.h, origin="lower", extent=img_ext,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("A  truth heightmap + FPM patches", fontsize=9)
    axes[0].set_ylabel("x / forward (m)", fontsize=8)

    def layer(ax, hm):
        ax.imshow(np.where(hm.valid, hm.height, np.nan), origin="lower", extent=img_ext,
                  cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    layer(axes[1], single)
    axes[1].set_title("B  single frame (current depth only)", fontsize=9)
    layer(axes[2], hist)
    retained = hist.valid & ~hist.updated
    if retained.any():
        axes[2].imshow(np.where(retained, 1.0, np.nan), origin="lower", extent=img_ext,
                       cmap="autumn", vmin=0, vmax=1, alpha=0.40, aspect="auto")
    axes[2].set_title("C  fused history (orange = retained from earlier frames)", fontsize=9)
    age = np.where(hist.valid, (step - hist.last_step) * dt, np.nan)
    im = axes[3].imshow(age, origin="lower", extent=img_ext, cmap="magma_r", vmin=0,
                        vmax=min(12.0, max(1.0, step * dt)), aspect="auto")
    axes[3].set_title("D  age since last observation (s, grey = never seen)", fontsize=9)
    plt.colorbar(im, ax=axes[3], fraction=0.040, pad=0.02).ax.tick_params(labelsize=6)

    colors = ["#e02b20", "#1f6fe0", "#e0a800", "#12b34a"]
    lines = []
    for f in range(4):
        pts = fs[f]
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        for ax in axes:
            ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec=colors[f], lw=1.3))
        axes[0].text(y1 + 0.05, x0 - 0.06, FEET[f], color=colors[f], fontsize=8, fontweight="bold")
        h_obs, seen, last = hist.sample(pts[:, 0], pts[:, 1])
        if seen.any():
            ages = (step - last[seen]) * dt
            err = np.abs(h_obs[seen] - truth.at(pts[seen, 0], pts[seen, 1]))
            lines.append((f"{FEET[f]}  cov {seen.mean()*100:3.0f}%   age {np.median(ages):4.1f} s"
                          f"   err {np.mean(err)*1000:3.0f} mm", colors[f]))
        else:
            lines.append((f"{FEET[f]}  cov   0%   never observed", colors[f]))
    for i, (txt, col) in enumerate(lines):
        axes[2].text(0.015, 0.055 + i * 0.055, txt, transform=axes[2].transAxes, fontsize=7.5,
                     color=col, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, lw=0.7, alpha=0.9))
    for ax in axes:
        ax.plot(robot_xy[1], robot_xy[0], "k+", ms=11, mew=1.8)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    canvas.suptitle(title, fontsize=10)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitch", default="15,30")
    ap.add_argument("--spawn", type=int, default=1, choices=range(1, 6))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    ap.add_argument("--vfov", type=float, default=66.0)
    ap.add_argument("--panels", default="2.5,6.0,9.0", help="seconds at which to draw a four-panel figure")
    ap.add_argument("--nearest-limit", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=HERE / "perception_out")
    a = ap.parse_args()

    global TRUTH
    a.out.mkdir(parents=True, exist_ok=True)
    truth = Truth(HERE / "scene/heightmap.npz")
    TRUTH = truth
    panels_at = [float(v) for v in a.panels.split(",")]
    summary = {"vfov_assumed_deg": a.vfov, "spawn": a.spawn, "command": a.command,
               "seconds": a.seconds, "panels_s": panels_at, "fill_plane": "initial support plane (fixed)",
               "nearest_limit_m": a.nearest_limit,
               "note": "Assumed pinhole camera, not a ZED model. Policy drove on the truth map; "
                       "B/C maps are rebuilt along that single trajectory.", "runs": []}

    for pitch in [float(v) for v in a.pitch.split(",")]:
        cam = CameraSpec(label=f"assumed pitch{pitch:g}", pitch_deg=pitch, vfov_deg=a.vfov)
        run = Runner(cam, truth, a.spawn - 1, a.command)
        dt = run.ctrl.dt
        n = round(a.seconds / dt)
        shape = truth.h.shape
        hist = HeightMap(truth.origin, shape, truth.res, run.plane, stride=2)
        single = HeightMap(truth.origin, shape, truth.res, run.plane, stride=2)
        prev_contact = np.zeros(4, bool)
        blind = np.zeros(4, int)
        contacts = np.zeros(4, int)
        trace, drawn = [], set()
        print(f"[pitch {pitch:g}] hfov {cam.hfov_deg:.1f} deg, support plane {run.plane:+.3f} m, {n} steps", flush=True)

        for step in range(n):
            run.step()
            d = run.data
            terrain, with_robot = run.cam_rig.capture(run.vdata)
            pts, keep = run.cam_rig.points(run.vdata, terrain)
            occluded = with_robot < terrain - 0.02
            keep &= ~occluded
            hist.fuse(pts, keep, step)   # grows over time
            single.clear()               # current frame only
            single.fuse(pts, keep, step)

            fs = run.ctrl.vis["pts"]
            hit = run.contacts()
            for f in range(4):
                if hit[f] and not prev_contact[f]:
                    contacts[f] += 1
                    fp = d.xpos[run.model.body(f"{FEET[f].lower()}_toe_link").id][:2]
                    _, seen, _ = hist.sample(fp[None, 0], fp[None, 1])
                    if not seen[0]:
                        blind[f] += 1
            prev_contact = hit

            if (step + 1) % 25 == 0:
                trace.append({"t": round((step + 1) * dt, 3), "xy": d.qpos[:2].round(3).tolist(),
                          "feet": foot_metrics({"single": single, "history": hist}, step, dt, fs,
                                               hist, a.nearest_limit)})

            t = (step + 1) * dt
            for want in panels_at:
                if abs(t - want) < dt / 2 and want not in drawn:
                    drawn.add(want)
                    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
                    draw(fig, step, dt, hist, single, fs, d.qpos[:2], truth,
                         f"pitch {pitch:g}deg  t={t:.1f}s  x={d.qpos[0]:.2f}m")
                    fig.tight_layout()
                    path = a.out / f"panels_pitch{pitch:g}_t{want:g}.png"
                    fig.savefig(path, dpi=110)
                    plt.close(fig)

        fs = run.ctrl.vis["pts"]
        final = foot_metrics({"single": single, "history": hist}, n - 1, dt, fs, hist, a.nearest_limit)
        for row in final:
            f = FEET.index(row["foot"])
            row["contacts"] = int(contacts[f])
            row["blind_steps"] = int(blind[f])
        summary["runs"].append({"pitch_deg": pitch, "hfov_deg": round(cam.hfov_deg, 2),
                                "support_plane_m": round(run.plane, 4),
                                "final_x": round(float(run.data.qpos[0]), 3),
                                "feet": final, "trace": trace})
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        run.cam_rig.close()
        print(json.dumps({"pitch": pitch, "final_x": summary["runs"][-1]["final_x"],
                          "coverage": {r["foot"]: r["history"]["coverage"] for r in final}}), flush=True)


if __name__ == "__main__":
    main()

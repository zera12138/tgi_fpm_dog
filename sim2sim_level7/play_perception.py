"""Handheld test view for the perception pipeline in the level-7 yard.

Same controller and gamepad mapping as ``play.py``, plus a live view of *what the policy reads*:

  M / Tab   cycle   A truth map  ->  C fused history  ->  B single frame
  V         toggle the map carpet (height colour, grey = never observed)
  F         toggle FPM foothold markers (as in play.py)

The HUD reports, for the active map, the per-foot patch coverage, the patch age and the unknown
fraction -- the same numbers the offline experiments report.

  bash run_perception_view.sh
  bash run_perception_view.sh --headless --seconds 5      # smoke test, no window
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from play import CFG, HERE, Scene, load_reference
from inputs import Inputs
from async_perception import AsyncPerception
from perception import CameraSpec

MODES = [("A", "truth map"), ("C", "fused history"), ("B", "single frame")]
FEET = ["FL", "FR", "HL", "HR"]
CARPET_STEP = 0.10
CARPET_HALF = (1.6, 1.4)   # forward / lateral half-extent (m)


def height_rgba(z, valid, lo=-0.05, hi=1.10):
    if not valid:
        return np.array([0.70, 0.70, 0.72, 0.35], np.float32)
    t = float(np.clip((z - lo) / (hi - lo), 0.0, 1.0))
    # blue -> green -> yellow, readable against the grey yard
    c = np.array([0.15 + 0.75 * t, 0.25 + 0.65 * t, 0.85 - 0.70 * t, 0.85], np.float32)
    return c


class PerceptionScene(Scene):
    def __init__(self, cam: CameraSpec):
        super().__init__()
        self.cam = cam
        self.mode = 0
        self.carpet = True
        self.perc = None
        self.fill_now = None
        self.sim_t = 0.0
        self.stats = {}

    def ensure_perception(self):
        if self.perc is not None:
            return
        plane = float(self.height(*self.data.qpos[:2]))
        self.perc = AsyncPerception(HERE / "scene/level7.xml", self.cam, self.origin, self.h.shape,
                                    self.res, plane, stride=2, also_single=True,
                                    window=(200, 160),
                                    center=(float(self.data.qpos[0]), float(self.data.qpos[1]))
                                    ).start()
        deadline = time.monotonic() + 5.0
        while self.perc.latest() is None and time.monotonic() < deadline:
            self.perc.submit(self.data.qpos, self.data.qvel, self.sim_t)
            time.sleep(0.01)

    def active_map(self):
        if self.mode == 0:
            return None
        self.ensure_perception()
        return self.perc.latest() if self.mode == 1 else self.perc.latest_single()

    def torso_z(self):
        return float(self.data.xpos[self.model.body("torso").id][2])

    def refresh_height_fn(self):
        """True when a map is available. In B/C a missing map is never replaced by truth."""
        if self.mode == 0:
            self.ctrl.height_fn = self.height
            return True
        snap = self.active_map()
        if snap is None:
            return False
        # Unknown cells use the map's configured fill; observed heights stay absolute, so they
        # deepen correctly as the robot climbs.
        self.fill_now = snap.fill_value
        self.ctrl.height_fn = snap.height_at
        return True

    def step(self, command):
        """Returns (vb, peak), or None when B/C has no map yet -- the caller should hold still."""
        self.ensure_perception()
        if self.perc is not None:
            self.perc.submit(self.data.qpos, self.data.qvel, self.sim_t)
        if not self.refresh_height_fn():
            self.stats = {"mode": MODES[self.mode][1], "waiting_for_map": True}
            return None
        vb, peak = self.act(command)
        self.sim_t += self.ctrl.dt
        self.update_stats()
        return vb, peak

    def update_stats(self):
        snap = self.active_map()
        pts = self.ctrl.vis["pts"] if self.ctrl.vis else None
        if snap is None or pts is None:
            self.stats = {"mode": MODES[self.mode][1], "coverage": None}
            return
        cover, ages, unknown = [], [], []
        for f in range(4):
            p = pts[f]
            seen = snap.seen(p[:, 0], p[:, 1])
            cover.append(float(seen.mean()))
            ii, jj = snap.keys(p[:, 0], p[:, 1])
            inside = ((ii >= 0) & (ii < snap.height.shape[0])
                      & (jj >= 0) & (jj < snap.height.shape[1]))
            last = np.full(len(ii), -1, np.int32)
            last[inside] = snap.last_step[ii[inside], jj[inside]]   # patches can leave the grid
            ok = seen & inside & (last >= 0)
            if ok.any():
                ages.append(float(np.median(self.sim_t - last[ok] / 1000.0)))
            unknown.append(1.0 - cover[-1])
        self.stats = {"mode": MODES[self.mode][1], "coverage": [round(c, 3) for c in cover],
                      "age_s": round(float(np.median(ages)), 2) if ages else None,
                      "unknown": round(float(np.mean(unknown)), 3),
                      "version": snap.version, "age_of_frame_s": round(self.sim_t - snap.state_t, 3)}

    def cycle_mode(self):
        self.mode = (self.mode + 1) % len(MODES)
        self.refresh_height_fn()

    def decorate(self, scn):
        scn.ngeom = 0                      # user geometry is rebuilt from scratch every frame
        # No text labels: play.Scene.decorate only adds FL/FR/HL/HR and zone captions, and those
        # were covering the foothold markers. The four predicted points are real mocap bodies
        # (group 5, moved by Scene.prediction), so they stay visible without any label geometry.
        snap = self.active_map()
        if snap is None or not self.carpet:
            return
        cap = int(min(scn.maxgeom, len(scn.geoms)))
        budget = cap - scn.ngeom - 16       # keep room for labels/markers added above
        if budget <= 0:
            return
        step = CARPET_STEP
        while True:
            n = ((2 * CARPET_HALF[0]) / step + 1) * ((2 * CARPET_HALF[1]) / step + 1)
            if n <= budget or step > 0.4:
                break
            step *= 1.25
        px, py = float(self.data.qpos[0]), float(self.data.qpos[1])
        for dx in np.arange(-CARPET_HALF[0], CARPET_HALF[0] + 1e-9, step):
            for dy in np.arange(-CARPET_HALF[1], CARPET_HALF[1] + 1e-9, step):
                if scn.ngeom >= cap:
                    return
                i, j = snap.key(px + dx, py + dy)
                inside = 0 <= i < snap.height.shape[0] and 0 <= j < snap.height.shape[1]
                ok = bool(inside and snap.valid[i, j])
                if ok:
                    z = float(snap.height[i, j])
                elif inside and snap.fill_grid is not None:
                    z = float(snap.fill_grid[i, j])
                else:
                    z = float(self.fill_now if self.fill_now is not None else snap.fill_value)
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                    np.array([step * 0.46, step * 0.46, 0.002]),
                                    np.array([px + dx, py + dy, z + 0.004]),
                                    np.eye(3).flatten(), height_rgba(z, ok))
                scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--spawn", type=int, choices=range(1, 6), default=1)
    ap.add_argument("--command", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--mode", choices=["A", "C", "B"], default="A",
                    help="headless map source: A truth, C history, B single frame")
    ap.add_argument("--out", type=Path, default=HERE / "perception_view.json")
    a = ap.parse_args()

    cam = CameraSpec(label="assumed", pitch_deg=a.pitch, vfov_deg=66.0)
    ref = load_reference(CFG["reference_helpers"])
    inputs = Inputs(CFG, ref, enabled=not a.headless)
    scene = PerceptionScene(cam)
    scene.reset(a.spawn - 1)
    scene.mode = {"A": 0, "C": 1, "B": 2}[a.mode]
    scene.refresh_height_fn()
    scene.update_stats()

    if a.headless:  # smoke: drive the real perception path without a window
        command = np.clip(np.asarray(a.command, float), inputs.minimum, inputs.maximum)
        held = 0
        for _ in range(round((a.seconds or 3.0) / scene.ctrl.dt)):
            if scene.step(command) is None:
                held += 1
                time.sleep(0.01)
                continue
            if scene.fallen():
                break
        print(json.dumps({"mode": scene.stats, "held_for_map_steps": held,
                          "final_x": float(scene.data.qpos[0]),
                          "frames_published": scene.perc.frames if scene.perc else 0}, indent=2))
        if scene.perc:
            scene.perc.stop()
        a.out.write_text(json.dumps({"stats": scene.stats}, indent=2))
        inputs.close()
        return

    from mujoco import viewer

    from queue import SimpleQueue

    keys = SimpleQueue()
    handle = viewer.launch_passive(scene.model, scene.data, key_callback=keys.put)
    with handle.lock():
        ref.setup_tracking_camera(handle, trackbodyid=scene.model.body("torso").id,
                                  distance=3.1, elevation=-30, azimuth=135)
        handle.opt.geomgroup[3] = 0
        handle.opt.geomgroup[4] = 1        # same visibility set as play.py, so group-5 markers show
        handle.opt.geomgroup[5] = 1
    print("M cycle map (A truth / C history / B single) | V carpet | F markers | "
          "Back/R reset | 1-5 spawn | Start/P pause", flush=True)
    paused, waiting, overview, command = False, False, False, np.zeros(3)
    try:
        while handle.is_running():
            tick = time.monotonic()
            command, events = inputs.poll()
            while not keys.empty():
                code = keys.get()
                ch = chr(code).lower() if isinstance(code, int) and 32 <= code < 127 else ""
                if ch == "m":                      # not an Inputs key: handle before it eats it
                    scene.cycle_mode()
                elif ch == "v":
                    scene.carpet = not scene.carpet
                events.extend(inputs.key(code))     # WASD, space, R, F, C, P, 1-5
            for ev in events:
                if ev in ("reset", "next_spawn") or isinstance(ev, tuple):
                    index = ((scene.spawn + 1) % 5 if ev == "next_spawn"
                             else ev[1] if isinstance(ev, tuple) else scene.spawn)
                    if scene.perc:
                        scene.perc.stop()
                    scene.perc = None
                    scene.reset(index)
                    scene.sim_t = 0.0
                    scene.update_stats()
                    inputs.reset()          # clear keyboard velocity, require stick recentre
                    paused = False
                    waiting = False
                    command[:] = 0
                elif ev == "pause":
                    paused = not paused
                elif ev == "camera":
                    overview = not overview
                elif ev == "markers":
                    scene.marker_visible = not scene.marker_visible
            if not paused:
                if scene.step(command) is None:
                    waiting = True          # B/C: no map yet, hold position
                else:
                    waiting = False
                    if scene.fallen():
                        paused = True
            # set_texts waits for the render thread; never hold its viewer lock here.
            handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f"PERCEPTION: {scene.stats.get('mode')}\n"
                               f"coverage FL/FR/HL/HR\n"
                               f"{scene.stats.get('coverage')}\n"
                               f"patch age {scene.stats.get('age_s')} s\n"
                               f"unknown {scene.stats.get('unknown')}"
                               + ("\nWAITING FOR MAP" if waiting else ""), "")])
            with handle.lock():
                if overview:
                    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    handle.cam.lookat[:] = [4, 0, 0.3]
                    handle.cam.distance, handle.cam.elevation = 20.0, -55.0
                elif handle.cam.type != mujoco.mjtCamera.mjCAMERA_TRACKING:
                    ref.setup_tracking_camera(handle, trackbodyid=scene.model.body("torso").id,
                                              distance=3.1, elevation=-30, azimuth=135)
                scene.decorate(handle.user_scn)
            handle.sync()
            time.sleep(max(0.0, scene.ctrl.dt - (time.monotonic() - tick)))
    finally:
        if scene.perc:
            scene.perc.stop()
        handle.close()
        inputs.close()


if __name__ == "__main__":
    main()

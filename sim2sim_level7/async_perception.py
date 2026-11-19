"""Asynchronous depth perception: rendering and fusion run off the control thread.

The control loop never blocks on a render. It submits the current simulator state into a one-slot
mailbox (newer states replace older ones) and reads the most recent **published snapshot**, which
is an immutable copy: a snapshot object is never mutated after publication, so the policy can never
observe a half-updated map.

Recorded times per snapshot:
  state_t      simulated time of the state the depth frame was rendered from
  capture_wall wall clock when the render started
  publish_wall wall clock when the snapshot became readable
  read_wall    wall clock when the policy grabbed it (recorded by the caller)

The EGL context and the render model are created *inside* the worker thread on purpose.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from perception import CameraSpec, DepthCamera, HeightMap, build_model


@dataclass(frozen=True)
class MapSnapshot:
    """Immutable view of the map at one publish instant."""

    height: np.ndarray
    valid: np.ndarray
    last_step: np.ndarray
    origin: np.ndarray
    res: float
    fill_plane: float
    fill_mode: str
    fill_value: float
    fill_grid: np.ndarray | None
    stand_height: float
    fill_margin: float
    version: int
    state_t: float
    capture_wall: float
    publish_wall: float

    def key(self, x, y):
        i = int(round((x - self.origin[0]) / self.res))
        j = int(round((y - self.origin[1]) / self.res))
        return i, j

    def keys(self, xs, ys):
        ii = np.rint((np.ravel(xs) - self.origin[0]) / self.res).astype(int)
        jj = np.rint((np.ravel(ys) - self.origin[1]) / self.res).astype(int)
        return ii, jj

    def fill_for_torso(self, torso_z):
        """Experimental ``relative`` mode only; the shipped default is a fixed world height."""
        if self.fill_mode == "relative":
            return float(torso_z) - self.stand_height - self.fill_margin
        return float(self.fill_value)

    def height_at(self, x, y, fill=None):
        i, j = self.key(x, y)
        if 0 <= i < self.height.shape[0] and 0 <= j < self.height.shape[1]:
            if self.valid[i, j]:
                return float(self.height[i, j])
            if fill is not None:
                return float(fill)
            if self.fill_grid is not None:
                return float(self.fill_grid[i, j])
        return float(self.fill_value if fill is None else fill)

    def seen(self, xs, ys):
        out = np.zeros(len(np.ravel(xs)), bool)
        for k, (x, y) in enumerate(zip(np.ravel(xs), np.ravel(ys))):
            i, j = self.key(x, y)
            out[k] = (0 <= i < self.height.shape[0] and 0 <= j < self.height.shape[1]
                      and bool(self.valid[i, j]))
        return out


class AsyncPerception:
    """One worker thread: state -> render -> back-project -> fuse -> publish snapshot."""

    def __init__(self, scene_xml, cam: CameraSpec, origin, shape, res, plane, stride=2,
                 also_single=False, fill_mode="low", window=None, center=None,
                 stand_height=0.43):
        self.scene_xml, self.cam = scene_xml, cam
        self.origin, self.shape, self.res = np.asarray(origin, float), tuple(shape), float(res)
        self.plane, self.stride = float(plane), int(stride)
        self.fill_mode, self.window = fill_mode, window
        self.stand_height = float(stand_height)
        if window is not None:
            self.shape = (int(window[0]), int(window[1]))
            cx, cy = (center if center is not None else (0.0, 0.0))
            half = np.array([window[0] // 2, window[1] // 2], float)
            self.origin = np.array([cx, cy], float) - half * self.res  # window centred on the robot
        self._box: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._snap: MapSnapshot | None = None
        self._snap_single: MapSnapshot | None = None
        self.also_single = also_single
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="perception")
        self.frames = 0
        self.dropped_submissions = 0

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def submit(self, qpos, qvel, state_t):
        """Never blocks: an unconsumed older state is discarded."""
        item = (np.array(qpos, float), np.array(qvel, float), float(state_t))
        try:
            self._box.put_nowait(item)
        except queue.Full:
            try:
                self._box.get_nowait()
                self.dropped_submissions += 1
            except queue.Empty:
                pass
            try:
                self._box.put_nowait(item)
            except queue.Full:
                pass

    def latest(self) -> MapSnapshot | None:
        with self._lock:
            return self._snap

    def latest_single(self) -> MapSnapshot | None:
        with self._lock:
            return self._snap_single

    def _publish(self, work, state_t, capture_wall):
        """Copy the map *and* the rule the policy will read for unknown cells."""
        local = work.fill_grid() if work.fill_mode == "local" else None
        return MapSnapshot(work.height.copy(), work.valid.copy(), work.last_step.copy(),
                           work.origin.copy(), work.res, work.fill_plane,
                           work.fill_mode, float(work.low_plane if work.fill_mode not in ("plane", "relative")
                                                else work.fill_plane),
                           None if local is None else local.copy(),
                           work.stand_height, work.fill_margin,
                           self.frames, state_t, capture_wall, time.perf_counter())

    def _run(self):
        model = build_model(self.scene_xml, self.cam, apply_pitch=True)
        data = mujoco.MjData(model)
        rig = DepthCamera(model, self.cam)
        work = HeightMap(self.origin, self.shape, self.res, self.plane, stride=self.stride,
                         fill_mode=self.fill_mode, window=self.window,
                         stand_height=self.stand_height)
        work_single = HeightMap(self.origin, self.shape, self.res, self.plane, stride=self.stride,
                                fill_mode=self.fill_mode, window=self.window,
                                stand_height=self.stand_height)
        try:
            while not self._stop.is_set():
                try:
                    qpos, qvel, state_t = self._box.get(timeout=0.2)
                except queue.Empty:
                    continue
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(model, data)
                capture_wall = time.perf_counter()
                terrain, with_robot = rig.capture(data)
                pts, keep = rig.points(data, terrain)
                keep &= ~(with_robot < terrain - 0.02)
                self.frames += 1
                work.recenter(float(qpos[0]), float(qpos[1]))   # rolling window follows the robot
                work_single.recenter(float(qpos[0]), float(qpos[1]))
                work.fuse(pts, keep, int(round(state_t * 1000)))  # store ms, not frame count
                snap = self._publish(work, state_t, capture_wall)
                single = None
                if self.also_single:
                    work_single.clear()
                    work_single.fuse(pts, keep, int(round(state_t * 1000)))
                    single = self._publish(work_single, state_t, capture_wall)
                with self._lock:
                    self._snap = snap
                    if single is not None:
                        self._snap_single = single
        finally:
            rig.close()

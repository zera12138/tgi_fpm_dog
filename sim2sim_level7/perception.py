"""Assumed-camera depth perception for the level-7 yard.

This is an *assumed pinhole camera* at the BPX camera mount, not a ZED simulator. The intrinsics
below are one explicit, self-consistent choice; the published ZED 2i field-of-view numbers are
mutually inconsistent for a 16:9 rectilinear image, so no single product figure is used here.
Replace ``CameraSpec`` with a real calibration when one exists.

Responsibility split, matching ``play.py``:

* truth   -- ``mujoco.mj_ray`` / the raytraced ``scene/heightmap.npz`` (physics, termination, error)
* perceivable -- what this module renders, back-projects and fuses

A cell is only written by a depth measurement. Unobserved cells are never stored as terrain; the
fill rule exists only to keep the policy's ``height_fn`` finite and is always reported as
``valid=False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

# MuJoCo cameras look along their own -z. The ROS optical frame has +z forward, +y down, so a
# half turn about x puts the camera on the optical axis (view +z, up -y, right +x).
_OPTICAL_TO_MUJOCO_CAM = [0.0, 1.0, 0.0, 0.0]

DEPTH_CAM = "depth_cam"
_SCENE_GROUP = (0, 2, 3)  # terrain + robot collision/visual; 4 = lane markings, 5 = FPM markers


@dataclass(frozen=True)
class CameraSpec:
    """One explicit pinhole assumption. Square pixels, centred principal point."""

    label: str
    pitch_deg: float
    vfov_deg: float = 66.0
    width: int = 424
    height: int = 240
    near_m: float = 0.25
    far_m: float = 3.0

    @property
    def fy(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def fx(self) -> float:
        return self.fy  # square pixels

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


def build_model(scene_xml, cam: CameraSpec, apply_pitch: bool = True):
    """Compile the yard with a depth camera welded to the ZED optical frame.

    ``apply_pitch=False`` keeps the mount angle baked into the XML untouched. Rotating the camera
    link also rotates the ZED body's inertia, which perturbs the walker over a long rollout, so
    physics always uses the unmodified mount and only the render model carries the pitch.
    """
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if apply_pitch:
        half = math.radians(cam.pitch_deg) / 2.0
        spec.body("zed_front_camera_link").quat = [math.cos(half), 0.0, math.sin(half), 0.0]
    camera = spec.body("zed_front_left_camera_frame_optical").add_camera()
    camera.name = DEPTH_CAM
    camera.quat = _OPTICAL_TO_MUJOCO_CAM
    camera.fovy = cam.vfov_deg
    spec.visual.global_.offwidth, spec.visual.global_.offheight = cam.width, cam.height
    return spec.compile()


def robot_geom_ids(model) -> np.ndarray:
    """Geoms belonging to the robot subtree (everything hanging off ``torso``)."""
    root = model.body("torso").id
    subtree = {root}
    changed = True
    while changed:
        changed = False
        for i in range(model.nbody):
            if i not in subtree and model.body_parentid[i] in subtree:
                subtree.add(i)
                changed = True
    return np.flatnonzero(np.isin(model.geom_bodyid, sorted(subtree)))


class DepthCamera:
    """Two-pass depth: with the robot for occlusion, without it for terrain."""

    def __init__(self, model, cam: CameraSpec):
        self.model, self.cam = model, cam
        self.cid = model.camera(DEPTH_CAM).id
        self.robot_geoms = robot_geom_ids(model)
        self._saved_groups = model.geom_group[self.robot_geoms].copy()
        self.renderer = mujoco.Renderer(model, height=cam.height, width=cam.width)
        self.renderer.enable_depth_rendering()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        for g in _SCENE_GROUP:
            self.option.geomgroup[g] = 1
        u, v = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
        self.dirs = np.stack([(u - cam.cx) / cam.fx, -(v - cam.cy) / cam.fy, -np.ones_like(u)], -1)

    def close(self):
        self.renderer.close()

    def capture(self, data):
        """Returns (terrain_depth, occlusion_depth). Both are axial (-z) distances in metres."""
        self.model.geom_group[self.robot_geoms] = self._saved_groups
        self.renderer.update_scene(data, camera=DEPTH_CAM, scene_option=self.option)
        with_robot = self.renderer.render().copy()

        self.model.geom_group[self.robot_geoms] = 1  # group 1 is unused in this scene
        self.option.geomgroup[1] = 0
        self.renderer.update_scene(data, camera=DEPTH_CAM, scene_option=self.option)
        terrain = self.renderer.render().copy()
        self.option.geomgroup[1] = 1
        self.model.geom_group[self.robot_geoms] = self._saved_groups
        return terrain, with_robot

    def points(self, data, depth):
        """Back-project an axial depth image to world points, dropping self-occluded pixels."""
        rot = data.cam_xmat[self.cid].reshape(3, 3)
        pos = data.cam_xpos[self.cid]
        flat = (rot @ (self.dirs * depth[..., None]).reshape(-1, 3).T).T + pos
        pts = flat.reshape(depth.shape + (3,))
        rng = np.linalg.norm(pts - pos, axis=-1)
        keep = np.isfinite(depth) & (rng > self.cam.near_m) & (rng < self.cam.far_m)
        return pts, keep


class HeightMap:
    """Rolling 2-D elevation grid: observed height, validity and time of last observation.

    Unobserved cells stay ``valid=False`` and hold ``nan``. The fill rule is applied only when the
    policy queries a height, and is never fused, never counted as coverage and never given an age.
    """

    def __init__(self, origin, shape, res, fill_plane, nearest_limit=None, stride=1,
                 fill_mode="low", fill_margin=0.35, local_radius=0.30, window=None,
                 stand_height=0.43):
        # fill_mode: plane | low (default) | local | relative (experimental, see set_torso)
        self.origin = np.array(origin, dtype=float, copy=True)
        self.shape = tuple(int(s) for s in shape)
        self.res = float(res)
        self.height = np.full(self.shape, np.nan, np.float32)
        self.valid = np.zeros(self.shape, bool)
        self.last_step = np.full(self.shape, -1, np.int32)
        self.fill_plane = float(fill_plane)
        self.nearest_limit = nearest_limit
        self.stride = int(stride)  # depth pixels are ~4x denser than 4 cm cells; thin before fusing
        self.fill_mode = fill_mode          # plane | low | local  (low is the default rule)
        self.window = window                # (nx, ny) cells; None keeps the whole fixed grid
        self.fill_margin = float(fill_margin)
        self.local_radius = float(local_radius)
        self.stand_height = float(stand_height)   # nominal torso height above the surface
        self.fill_reference = float(fill_plane)   # plane the unknown fill hangs from
        self._local = None
        self._near_h = None
        self._near_d = None
        self.updated = np.zeros(self.shape, bool)  # cells written by the most recent fuse()

    @property
    def cell_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def recenter(self, x, y, margin_cells=20):
        """Roll a fixed-size window so ``(x, y)`` stays inside it; vacated cells become unknown."""
        if self.window is None:
            return 0
        moved = 0
        for axis, value in ((0, x), (1, y)):
            ci = int(round((value - self.origin[axis]) / self.res))
            keep = self.shape[axis] - margin_cells
            if margin_cells <= ci < keep:
                continue
            shift = ci - self.shape[axis] // 2
            if shift == 0:
                continue
            if abs(shift) >= self.shape[axis]:       # jumped past the whole window
                self.clear()
                self.origin[axis] += shift * self.res
                moved += abs(shift)
                continue
            for arr, fill in ((self.height, np.nan), (self.valid, False),
                              (self.last_step, -1), (self.updated, False)):
                rolled = np.roll(arr, -shift, axis=axis)
                if shift > 0:
                    sl = [slice(None)] * arr.ndim
                    sl[axis] = slice(arr.shape[axis] - shift, None)
                else:
                    sl = [slice(None)] * arr.ndim
                    sl[axis] = slice(0, -shift)
                rolled[tuple(sl)] = fill
                arr[...] = rolled
            self.origin[axis] += shift * self.res
            moved += abs(shift)
        if moved:
            self._near_h = self._near_d = self._local = None
        return moved

    def clear(self):
        """Forget everything: the map goes back to the state before the first observation."""
        self.height.fill(np.nan)
        self.valid.fill(False)
        self.last_step.fill(-1)
        self.updated.fill(False)
        self._near_h = self._near_d = self._local = None

    def cells(self, xs, ys):
        ij = np.rint((np.stack([np.ravel(xs), np.ravel(ys)], -1) - self.origin) / self.res).astype(int)
        inside = (ij >= 0).all(1) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        return ij, inside

    def fuse(self, points, in_range, step):
        """One frame: highest surface wins inside each cell, then overwrite that cell."""
        pts = points[in_range]
        if not len(pts):
            self.updated[:] = False
            return 0
        if self.stride > 1:
            pts = pts[:: self.stride]
        ij, inside = self.cells(pts[:, 0], pts[:, 1])
        ij, pts = ij[inside], pts[inside]
        if not len(pts):
            self.updated[:] = False
            return 0
        flat = ij[:, 0] * self.shape[1] + ij[:, 1]
        order = np.lexsort((pts[:, 2], flat))  # group by cell, ascending in height
        flat, high = flat[order], pts[order, 2]
        last = np.r_[flat[1:] != flat[:-1], True]  # highest point of each cell
        flat, high = flat[last], high[last]
        self.height.reshape(-1)[flat] = high
        self.valid.reshape(-1)[flat] = True
        self.last_step.reshape(-1)[flat] = step
        self.updated[:] = False
        self.updated.reshape(-1)[flat] = True
        self._near_h = self._near_d = self._local = None
        return len(flat)

    def _nearest(self):
        if self._near_h is not None:
            return self._near_h, self._near_d
        if not self.valid.any():
            nan = np.full(self.shape, np.nan, np.float32)
            return nan, np.full(self.shape, np.inf, np.float32)
        idx = ndimage.distance_transform_edt(~self.valid, return_distances=True, return_indices=True)
        self._near_h = self.height[tuple(idx[1])]
        self._near_d = idx[0].astype(np.float32) * self.res
        return self._near_h, self._near_d

    def sample(self, xs, ys):
        """Observed height / validity / step of last observation, without any fill applied."""
        ij, inside = self.cells(xs, ys)
        seen = np.zeros(len(ij), bool)
        h = np.full(len(ij), np.nan, np.float32)
        last = np.full(len(ij), -1, np.int32)
        if inside.any():
            ii, jj = ij[inside, 0], ij[inside, 1]
            seen[inside] = self.valid[ii, jj]
            h[inside] = self.height[ii, jj]
            last[inside] = self.last_step[ii, jj]
        return h, seen, last

    def fill_report(self, xs, ys, limit):
        """How the unknown part of a patch would be filled, without changing anything."""
        _, seen, _ = self.sample(xs, ys)
        unknown = np.flatnonzero(~seen)
        out = {"observed": round(float(seen.mean()), 4), "nearest": 0.0, "plane": 0.0}
        if not unknown.size:
            return out
        nearest = 0
        if self.valid.any():
            near_h, near_d = self._nearest()
            ij, inside = self.cells(xs[unknown], ys[unknown])
            sub = np.flatnonzero(inside)
            ii, jj = ij[sub, 0], ij[sub, 1]
            ok = (near_d[ii, jj] <= limit) & np.isfinite(near_h[ii, jj])
            nearest = int(ok.sum())
        out["nearest"] = round(nearest / len(xs), 4)
        out["plane"] = round((len(unknown) - nearest) / len(xs), 4)
        return out

    @property
    def low_plane(self):
        return self.fill_plane - self.fill_margin

    def set_torso(self, torso_z):
        """Experimental only -- NOT a fix for unknown regions.

        Re-anchoring the fill to the torso swaps one assumption ("unknown ground is level with
        the spawn") for another ("unknown ground is level with wherever the robot stands now").
        Neither is knowledge. The default stays a fixed world height; observed cells are always
        absolute, so they deepen correctly as the robot climbs.
        """
        ref = float(torso_z) - self.stand_height
        if self._local is not None and ref != self.fill_reference:
            self._local = None
        self.fill_reference = ref

    def fill_grid(self):
        """The value ``height_at`` returns for unobserved cells, as a grid (never fused)."""
        if self.fill_mode == "plane":
            return np.full(self.shape, self.fill_plane, np.float32)
        if self.fill_mode == "low":
            return np.full(self.shape, self.low_plane, np.float32)
        if self.fill_mode == "relative":
            return np.full(self.shape, self.fill_reference - self.fill_margin, np.float32)
        if self._local is None:
            k = 2 * int(round(self.local_radius / self.res)) + 1
            big = np.where(self.valid, self.height, np.inf).astype(np.float32)
            mn = ndimage.minimum_filter(big, size=k, mode="constant", cval=np.inf)
            local = np.where(np.isfinite(mn), mn - self.fill_margin, self.low_plane)
            self._local = local.astype(np.float32)
        return self._local

    def height_at(self, x, y):
        """Scalar query used by ``Controller.height_fn``: ~40x cheaper than the vectorised path."""
        i = int(round((x - self.origin[0]) / self.res))
        j = int(round((y - self.origin[1]) / self.res))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            if self.valid[i, j]:
                return float(self.height[i, j])
            if self.fill_mode != "plane":
                return float(self.fill_grid()[i, j])
            return float(self.fill_plane)
        if self.nearest_limit is not None and self.valid.any():
            near_h, near_d = self._nearest()
            if 0 <= i < self.shape[0] and 0 <= j < self.shape[1] and near_d[i, j] <= self.nearest_limit:
                return float(near_h[i, j])
        if self.fill_mode == "low":
            return float(self.low_plane)
        if self.fill_mode == "relative":
            return float(self.fill_reference - self.fill_margin)
        return float(self.fill_plane)

    def query(self, xs, ys):
        """Batch heights: observed where known, otherwise exactly what ``height_at`` returns."""
        xs, ys = np.atleast_1d(np.asarray(xs, float)), np.atleast_1d(np.asarray(ys, float))
        h, seen, _ = self.sample(xs, ys)
        out = h.copy()
        unknown = np.flatnonzero(~seen)
        if not unknown.size:
            return out
        fgrid = self.fill_grid()                       # same rule as the policy path
        filled = np.full(unknown.size, self.height_at(float(xs[unknown[0]]),
                                                      float(ys[unknown[0]])), np.float32)
        ij, inside = self.cells(xs[unknown], ys[unknown])
        sub = np.flatnonzero(inside)
        filled[sub] = fgrid[ij[sub, 0], ij[sub, 1]]
        out[unknown] = filled
        return out


def support_plane(cam: DepthCamera, data, cam_spec: CameraSpec) -> float:
    """Initial support plane from the first depth frame only -- no truth is consulted."""
    terrain, _ = cam.capture(data)
    pts, keep = cam.points(data, terrain)
    good = pts[keep]
    if not len(good):
        return 0.0
    near = np.linalg.norm(good - data.cam_xpos[cam.cid], axis=-1) < 1.0
    return float(np.median(good[near, 2])) if near.any() else float(np.median(good[:, 2]))

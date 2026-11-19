"""Exercise the interactive viewer loop without a window (stub launch_passive).

Covers the code that only runs on the desktop: handle.lock, tracking camera,
set_texts, user_scn overlays, arrow helpers and the per-frame pacing.
"""
import os
import sys
from contextlib import contextmanager

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import mujoco
import mujoco.viewer as mjv
import play


class Handle:
    def __init__(self, model):
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.user_scn = mujoco.MjvScene(model, maxgeom=1000)
        self.texts = None
        self.frames = 0
        self.closed = False

    @contextmanager
    def lock(self):
        yield

    def set_texts(self, texts):
        self.texts = texts

    def is_running(self):
        return not self.closed

    def sync(self):
        self.frames += 1

    def close(self):
        self.closed = True


created = []


def launch(model, data, **kwargs):
    handle = Handle(model)
    created.append(handle)
    return handle


def main():
    mjv.launch_passive = launch
    sys.argv = ['play.py', '--seconds', '2', '--out', sys.argv[1]]
    play.main()
    handle = created[0]
    assert handle.texts is not None and len(handle.texts) == 2, handle.texts
    assert handle.frames > 50, handle.frames
    assert handle.user_scn.ngeom >= 8, handle.user_scn.ngeom
    assert handle.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    print(f'viewer loop ok: frames={handle.frames} texts={len(handle.texts)} overlay_geoms={handle.user_scn.ngeom}')


if __name__ == '__main__':
    main()

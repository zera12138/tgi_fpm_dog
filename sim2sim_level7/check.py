"""Check input mapping/latches, real scene-map agreement, FPM coordinate projection and reset."""
import json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from inputs import Inputs, load_reference
from play import Scene, CFG, quat_to_mat

ref=load_reference(CFG['reference_helpers'])
class Joy:
    def __init__(self):self.axes=[0.,0.,0.,0.];self.buttons=set();self.attached=True
    def get_axis(self,i):return self.axes[i]
    def get_numaxes(self):return 4
    def get_numbuttons(self):return 8
    def get_button(self,i):return i in self.buttons
    def get_attached(self):return self.attached
joy=Joy()
# Feed synthetic events through the actual reference mapping and adapter, without a physical device.
original_pump=ref.pygame.event.pump
ref.pygame.event.pump=lambda:None
try:
    inp=Inputs(CFG,ref,False);inp.enabled=True;inp.joystick=joy
    inp.pg=SimpleNamespace(event=SimpleNamespace(pump=lambda:None,get=lambda:[]),error=RuntimeError,
                           joystick=SimpleNamespace(get_count=lambda:0))
    joy.axes=[.8,-1.,0.,-.5]
    np.testing.assert_allclose(inp.poll()[0],[.6,-.2,.35],atol=1e-7)
    joy.axes=[.05,.05,0.,.05];np.testing.assert_array_equal(inp.poll()[0],[0,0,0])
    joy.axes=[0.,1.,0.,0.];assert inp.poll()[0][0]==-.3
    joy.buttons={1};assert not inp.poll()[0].any()
    joy.buttons=set();assert not inp.poll()[0].any() # STOP remains latched.
    joy.buttons={0};assert not inp.poll()[0].any() # Held stick cannot immediately resume.
    joy.axes=[0.,0.,0.,0.];joy.buttons=set();inp.poll()
    joy.axes=[0.,-1.,0.,0.];assert inp.poll()[0][0]>.5
    joy.buttons={6};assert inp.poll()[1]==['reset'];assert inp.poll()[1]==[]
    joy.attached=False;assert not inp.poll()[0].any();assert inp.stop_latched
    inp.enabled=False;inp.key(257);inp.key(265);assert inp.poll()[0][0]>.09
    inp.key(32);assert not inp.poll()[0].any()
finally:ref.pygame.event.pump=original_pump

s=Scene();rng=np.random.default_rng(19)
# Use map-node positions, where sampled terrain heights must equal collision ray heights.
error=0.
for _ in range(150):
    ix=int(rng.integers(1,s.h.shape[0]-1));iy=int(rng.integers(1,s.h.shape[1]-1))
    p=s.origin+np.array([ix,iy])*s.res
    error=max(error,abs(s.height(*p)-s.surface(*p)))
assert error<2e-6,error
assert np.isclose(s.surface(2.1,0),.1275)
assert np.isclose(s.surface(5.,0),1.02)
assert np.isclose(s.surface(6.5,0),1.02-(6.5-5.7)*.4125)
# FPM xy must use full root rotation, not yaw-only; rendering z is scene-surface height.
s.data.qpos[3:7]=[math.cos(.2),0,math.sin(.2),0]
mujoco.mj_forward(s.model,s.data)
s.ctrl.last_fh=np.array([[.4,.2],[.4,-.2],[-.3,.2],[-.3,-.2]])
s.prediction()
R=quat_to_mat(s.data.qpos[3:7]);want=s.data.qpos[:3]+(R@np.c_[s.ctrl.last_fh,np.zeros(4)].T).T
np.testing.assert_allclose(s.predicted[:,:2],want[:,:2],atol=1e-9)
for p in s.predicted:assert abs(p[2]-s.surface(*p[:2])-.035)<1e-9
# Some pygame builds (seen on the host) lack Joystick.get_attached: unplug must still latch STOP.
class Bare:
    def __init__(self):self.axes=[0.]*4;self.buttons=set()
    def get_axis(self,i):return self.axes[i]
    def get_numaxes(self):return 4
    def get_numbuttons(self):return 8
    def get_button(self,i):return i in self.buttons
    def get_init(self):return True
bare=Bare();inp2=Inputs(CFG,ref,False);inp2.enabled=True;inp2.joystick=bare
inp2.pg=SimpleNamespace(event=SimpleNamespace(pump=lambda:None,get=lambda:[]),error=RuntimeError,
                        joystick=SimpleNamespace(get_count=lambda:1))
pump=ref.pygame.event.pump;ref.pygame.event.pump=lambda:None
try:
    assert not inp2.poll()[0].any() and not inp2.stop_latched
    inp2.pg.joystick.get_count=lambda:0
    assert not inp2.poll()[0].any() and inp2.stop_latched and inp2.joystick is None
finally:ref.pygame.event.pump=pump
reports=[]
for spawn in range(5):
    s.reset(spawn)
    assert s.ctrl.history is None and s.ctrl.phase==0 and not s.ctrl.a_prev.any()
    for step in range(100):s.act(np.zeros(3))
    assert not s.fallen(),spawn
    assert np.isfinite(s.predicted).all()
    reports.append(dict(spawn=spawn+1,seconds=2,standing=True,xyz=s.data.qpos[:3].tolist()))
summary=dict(input_mapping='reference axes/signs/deadzone passed',stop_resume_disconnect='passed (synthetic joystick)',
             map_collision_max_error=error,marker_full_rotation_and_surface='passed',resets_and_standing=reports,
             hardware_gamepad='not available in sandbox')
Path(__file__).with_name('checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

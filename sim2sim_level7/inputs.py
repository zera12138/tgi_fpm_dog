"""Reuse moe_lant joystick mapping; add hotplug, stop latch and reset buttons."""
import importlib.util
import time
import numpy as np


def load_reference(path):
    spec=importlib.util.spec_from_file_location('moe_lant_sim2sim_helpers', path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class MappedJoystick:
    def __init__(self, joystick, axes):
        self.joystick=joystick;self.indices={0:axes['vy'],1:axes['vx'],3:axes['yaw']}
    def get_axis(self, index):return self.joystick.get_axis(self.indices[index])


class Inputs:
    def __init__(self, config, reference, enabled=True):
        self.cfg=config;self.ref=reference;self.pg=reference.pygame;self.enabled=enabled
        self.maximum=np.array(config['max_command']);self.minimum=np.array(config['min_command'])
        self.keyboard=np.zeros(3);self.joystick=None;self.name='keyboard';self.previous=set()
        self.stop_latched=False;self.need_center=False;self.last_probe=-1.;self.last_axes=[]
        if enabled:
            self.pg.init();self.pg.joystick.init();self._connect()
    def _connect(self):
        if self.pg.joystick.get_count():
            device=self.pg.joystick.Joystick(0);device.init()
            if max(self.cfg['axes'].values())>=device.get_numaxes():
                self.name='axis config invalid';return
            self.joystick=device;self.name=device.get_name();self.previous=set();self.need_center=True
            print(f'Gamepad connected: {self.name}; axes={device.get_numaxes()}, buttons={device.get_numbuttons()}',flush=True)
    def reset(self):
        self.keyboard[:]=0;self.need_center=self.joystick is not None;self.stop_latched=False
    def key(self, key):
        events=[]
        if key in (265,87):self.keyboard[0]+=.1
        elif key in (264,83):self.keyboard[0]-=.1
        elif key==65:self.keyboard[1]+=.05
        elif key==68:self.keyboard[1]-=.05
        elif key in (263,81):self.keyboard[2]+=.1
        elif key in (262,69):self.keyboard[2]-=.1
        elif key==32:self.stop_latched=True;self.keyboard[:]=0
        elif key==257:self.stop_latched=False;self.need_center=self.joystick is not None
        elif key==82:events.append('reset')
        elif key==80:events.append('pause')
        elif key==67:events.append('camera')
        elif key==70:events.append('markers')
        elif 49<=key<=53:events.append(('spawn',key-49))
        self.keyboard=np.clip(self.keyboard,self.minimum,self.maximum)
        return events
    def _attached(self):
        """pygame builds differ: some expose Joystick.get_attached, some do not."""
        joy=self.joystick
        if joy is None:return False
        probe=getattr(joy,'get_attached',None)
        if callable(probe):
            try:return bool(probe())
            except Exception:pass
        try:
            if not joy.get_init():return False
            return self.pg.joystick.get_count()>0
        except Exception:return True
    def poll(self):
        events=[];command=self.keyboard.copy()
        if self.enabled:
            try:
                self.pg.event.pump()
                self.pg.event.get()  # Drain SDL events during long interactive sessions.
                if self.joystick is not None and not self._attached():
                    self.joystick=None;self.name='disconnected / keyboard';self.stop_latched=True;self.keyboard[:]=0;command[:]=0
                    print('Gamepad disconnected: command stopped. Enter resumes keyboard control.',flush=True)
                if self.joystick is None and time.monotonic()-self.last_probe>1.:
                    self.last_probe=time.monotonic();self._connect()
                if self.joystick is not None:
                    joy=self.joystick
                    self.last_axes=[round(joy.get_axis(i),3) for i in range(joy.get_numaxes())]
                    command=self.ref.read_joystick_command(MappedJoystick(joy,self.cfg['axes']),self.maximum)
                    now={name for name,index in self.cfg['buttons'].items() if index<joy.get_numbuttons() and joy.get_button(index)}
                    for name in now-self.previous:
                        if name=='stop':self.stop_latched=True
                        elif name=='resume':self.stop_latched=False;self.need_center=True
                        else:events.append(name)
                    self.previous=now
                    if self.need_center:
                        if np.max(np.abs(command))==0:self.need_center=False
                        command[:]=0
            except self.pg.error as exc:
                self.joystick=None;self.name='input error / keyboard';self.stop_latched=True;command[:]=0
                print(f'Gamepad error; command stopped: {exc}',flush=True)
        if self.stop_latched:command[:]=0
        return np.clip(command,self.minimum,self.maximum),events
    def close(self):
        if self.enabled:self.pg.quit()

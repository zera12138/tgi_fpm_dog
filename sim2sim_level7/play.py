"""Interactive BPX 100k ONNX sim2sim: level-7 yard, gamepad and four FPM markers."""
from pathlib import Path
from queue import SimpleQueue
import argparse, json, math, sys, time
import numpy as np
import mujoco
from inputs import Inputs, load_reference

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'controls.json').read_text())
sys.path[:0]=[str(Path(CFG['project'])/'tgi/src'),str(Path(CFG['project'])/'fpm/src')]
from tgi.deploy.controller import Controller, quat_to_mat


class Scene:
    def __init__(self):
        self.info=json.loads((HERE/'scene/scene.json').read_text())
        self.model=mujoco.MjModel.from_xml_path(str(HERE/'scene/level7.xml'))
        self.data=mujoco.MjData(self.model)
        archive=np.load(HERE/'scene/heightmap.npz')
        self.h=archive['heights'];self.origin=archive['origin'];self.res=float(archive['res'])
        self.mask=np.array([1,0,0,0,0,0],np.uint8);self.gid=np.zeros(1,np.int32)
        self.marker_ids=[int(self.model.body_mocapid[self.model.body('fpm_'+n).id]) for n in ['FL','FR','HL','HR']]
        self.predicted=np.zeros((4,3));self.marker_visible=True
        self.ctrl=Controller(str(HERE),height_fn=self.height)
        ids=[self.model.joint(n).id for n in self.ctrl.names]
        self.qidx=[self.model.jnt_qposadr[j] for j in ids];self.vidx=[self.model.jnt_dofadr[j] for j in ids]
        self.actuators=[int(np.flatnonzero(self.model.actuator_trnid[:,0]==j)[0]) for j in ids]
        self.substeps=round(self.ctrl.dt/self.model.opt.timestep)
        assert np.isclose(self.substeps*self.model.opt.timestep,self.ctrl.dt)
        self.spawn=0;self.reset(0)
    def height(self,x,y):
        ij=np.rint((np.array([x,y])-self.origin)/self.res).astype(int)
        if (ij<0).any() or ij[0]>=self.h.shape[0] or ij[1]>=self.h.shape[1]:return -.8
        return float(self.h[ij[0],ij[1]])
    def surface(self,x,y):
        distance=mujoco.mj_ray(self.model,self.data,np.array([x,y,5.]),np.array([0.,0.,-1.]),self.mask,True,-1,self.gid)
        return 5.-distance if distance>=0 else -.8
    def reset(self,index=None):
        if index is not None:self.spawn=index
        spawn=self.info['spawns'][self.spawn];d=self.data
        mujoco.mj_resetDataKeyframe(self.model,d,0)
        d.qpos[:3]=[*spawn['xy'],spawn['z']+.43]
        d.qpos[3:7]=[math.cos(spawn['yaw']/2),0,0,math.sin(spawn['yaw']/2)]
        d.qpos[self.qidx]=self.ctrl.q0;d.qvel[:]=0;d.ctrl[:]=0
        self.ctrl.reset();d.mocap_pos[:]=[0,0,-4];mujoco.mj_forward(self.model,d)
    def prediction(self):
        R=quat_to_mat(self.data.qpos[3:7]);p=self.data.qpos[:3]
        for f,xy in enumerate(self.ctrl.last_fh):
            pt=p+R@np.array([xy[0],xy[1],0.])
            pt[2]=self.surface(pt[0],pt[1])+.035
            self.predicted[f]=pt
        self.data.mocap_pos[self.marker_ids]=self.predicted if self.marker_visible else [0,0,-4]
    def act(self,command):
        d=self.data;R=quat_to_mat(d.qpos[3:7]);vb=R.T@d.qvel[:3]
        target=self.ctrl.step(d.qpos[:3].copy(),d.qpos[3:7].copy(),vb,d.qvel[3:6].copy(),d.qpos[self.qidx].copy(),d.qvel[self.vidx].copy(),command)
        if not np.isfinite(target).all():raise FloatingPointError('Nonfinite policy target')
        self.prediction()
        peak=0.
        for _ in range(self.substeps):
            torque=np.clip(self.ctrl.kp*(target-d.qpos[self.qidx])-self.ctrl.kd*d.qvel[self.vidx],-self.ctrl.eff,self.ctrl.eff)
            d.ctrl[self.actuators]=torque;peak=max(peak,float(np.abs(torque).max()));mujoco.mj_step(self.model,d)
        return vb,peak
    def fallen(self):
        d=self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():return True
        # Use terrain-relative clearance; elevated platforms are not world-zero ground.
        return bool(quat_to_mat(d.qpos[3:7])[2,2]<math.cos(.7) or d.qpos[2]-self.surface(*d.qpos[:2])<.16)
    def decorate(self,scn):
        labels=[('FL', [1,.16,.12,1]),('FR',[.15,.5,1,1]),('HL',[1,.75,.1,1]),('HR',[.12,.95,.4,1])]
        if self.marker_visible:
            for i,(name,color) in enumerate(labels):
                g=scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),self.predicted[i]+[0,0,.2],np.eye(3).flatten(),np.array(color,np.float32))
                g.label=name;scn.ngeom+=1
        for label,pos in [('STAIRS L7 / 12.75 cm',[2,-2.25,.08]),('SLOPE L7 / 22.42 deg',[8.2,-2.25,.08]),('STONES L7',[2,6.7,.08]),('ROUGH L7',[2,-6.8,.08])]:
            g=scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),np.array(pos),np.eye(3).flatten(),np.array([.95,.95,.95,1],np.float32))
            g.label=label;scn.ngeom+=1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--headless',action='store_true');ap.add_argument('--seconds',type=float,default=None)
    ap.add_argument('--command',type=float,nargs=3,default=[0,0,0],help='fixed command for headless checks only')
    ap.add_argument('--spawn',type=int,choices=range(1,6),default=1)
    ap.add_argument('--preview',action='store_true',help='render overview and robot close-up PNGs, then exit')
    ap.add_argument('--out',type=Path,default=None);ap.add_argument('--pad-test',action='store_true')
    a=ap.parse_args()
    ref=load_reference(CFG['reference_helpers'])
    inputs=Inputs(CFG,ref,enabled=not(a.headless or a.preview))
    if a.pad_test:
        try:
            until=time.monotonic()+(a.seconds or 30)
            while time.monotonic()<until:
                cmd,events=inputs.poll();print(f'{inputs.name}: axes={inputs.last_axes} command={cmd.round(3)} events={events}',flush=True);time.sleep(.2)
        finally:inputs.close()
        return
    scene=Scene();scene.reset(a.spawn-1);d=scene.data;m=scene.model
    if a.preview:
        scene.act(np.zeros(3));renderer=mujoco.Renderer(m,height=900,width=1440)
        opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1;opt.geomgroup[5]=1
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        dest=a.out or HERE/'preview';dest.mkdir(parents=True,exist_ok=True)
        from PIL import Image
        try:
            for name,look,distance,elevation,azimuth in [('overview',[4,0,.3],20,-55,130),('robot',[.2,0,.25],2.5,-32,145)]:
                camera.lookat[:]=look;camera.distance=distance;camera.elevation=elevation;camera.azimuth=azimuth
                renderer.update_scene(d,camera=camera,scene_option=opt);scene.decorate(renderer.scene)
                Image.fromarray(renderer.render()).save(dest/f'{name}.png')
        finally:renderer.close();inputs.close()
        return
    handle=None;keys=SimpleQueue();overview=False;paused=False;fallen=False
    arrows=ref.MujocoRenderUtils()
    if not a.headless:
        from mujoco import viewer
        handle=viewer.launch_passive(m,d,key_callback=keys.put)
        with handle.lock():
            ref.setup_tracking_camera(handle,trackbodyid=m.body('torso').id,distance=3.1,elevation=-30,azimuth=135)
            handle.opt.geomgroup[3]=0;handle.opt.geomgroup[4]=1;handle.opt.geomgroup[5]=1
        print('Left Y=vx, Left X=vy, Right X=yaw. B/Space STOP; A/Enter resume; Back/R reset; RB next spawn; Start/P pause; 1-5 spawn; C camera; F markers.',flush=True)
    duration=a.seconds if a.seconds is not None else (10. if a.headless else float('inf'))
    started=time.monotonic();steps=0;max_torque=0.;resets=0;min_clearance=1e9;vel_errors=[]
    command=np.zeros(3);vb=np.zeros(3)
    try:
        while (steps*scene.ctrl.dt<duration if a.headless else time.monotonic()-started<duration):
            tick=time.monotonic()
            if handle is not None and not handle.is_running():break
            command,events=inputs.poll()
            while not keys.empty():events.extend(inputs.key(keys.get()))
            for event in events:
                if event in ('reset','next_spawn') or isinstance(event,tuple):
                    index=(scene.spawn+1)%5 if event=='next_spawn' else event[1] if isinstance(event,tuple) else scene.spawn
                    scene.reset(index);inputs.reset();paused=False;fallen=False;command[:]=0;resets+=1
                elif event=='pause':paused=not paused
                elif event=='camera':overview=not overview
                elif event=='markers':scene.marker_visible=not scene.marker_visible;d.mocap_pos[scene.marker_ids]=scene.predicted if scene.marker_visible else [0,0,-4]
            if a.headless:command=np.clip(a.command,inputs.minimum,inputs.maximum)
            elif inputs.stop_latched:command[:]=0
            if not(paused or fallen):
                vb,peak=scene.act(command);max_torque=max(max_torque,peak);steps+=1
                vel_errors.append(float(np.linalg.norm(vb[:2]-command[:2])))
                min_clearance=min(min_clearance,float(d.qpos[2]-scene.surface(*d.qpos[:2])))
                fallen=scene.fallen()
                if fallen:
                    inputs.stop_latched=True;d.ctrl[:]=0
                    print('MDP fall threshold reached: physics paused. Back/R or a spawn key resets.',flush=True)
                    if a.headless:break
            if handle is not None:
                status='FALL - RESET' if fallen else 'PAUSED' if paused else 'STOPPED' if inputs.stop_latched else 'CENTER STICKS' if inputs.need_center else 'RUNNING'
                handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                   f'LEVEL 7\n{scene.info["spawns"][scene.spawn]["name"]}\n{inputs.name}\n{status}\nCMD vx / vy / yaw\nACTUAL vx / vy',
                   f'100k | 50 Hz\ndifficulty 0.75\n\n\n{command[0]:+.2f}  {command[1]:+.2f}  {command[2]:+.2f}\n{vb[0]:+.2f}  {vb[1]:+.2f}'),
                   (mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    'Left stick: move | Right stick: turn\nB/Space stop | A/Enter resume | Back/R reset\nRB / 1-5: spawn | Start/P pause | C camera | F predictions\nFPM XY markers: FL red / FR blue / HL yellow / HR green', '')])
                with handle.lock():
                    if overview:
                        handle.cam.type=mujoco.mjtCamera.mjCAMERA_FREE;handle.cam.lookat[:]=[4,0,.3];handle.cam.distance=20;handle.cam.elevation=-55
                    elif handle.cam.type!=mujoco.mjtCamera.mjCAMERA_TRACKING:
                        ref.setup_tracking_camera(handle,trackbodyid=m.body('torso').id,distance=3.1,elevation=-30,azimuth=135)
                    arrows.update(command,d);arrows.update_external_rendering(handle,'viewer');scene.decorate(handle.user_scn)
                handle.sync();time.sleep(max(0.,scene.ctrl.dt-(time.monotonic()-tick)))
    finally:
        if handle is not None:handle.close()
        inputs.close()
    report=dict(seconds=steps*scene.ctrl.dt,spawn=scene.spawn+1,fallen=fallen,final_xyz=d.qpos[:3].tolist(),
                command=command.tolist(),max_torque=max_torque,min_base_clearance=min_clearance,
                mean_xy_error=float(np.mean(vel_errors)) if vel_errors else None,resets=resets,
                marker_positions=scene.predicted.tolist(),input_device=inputs.name,
                note='Nominal MuJoCo + ONNX controller, ideal scene heightmap; gamepad hardware only tested if a device was available.')
    out=a.out or HERE/'last_run.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

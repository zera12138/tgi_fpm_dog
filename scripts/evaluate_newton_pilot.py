"""Nominal-physics Newton evaluation of a trained BPX policy, CPU or CUDA.
Uses the same IsaacLocoSim and SimRunner as training. No training/hardware IO.
"""
from pathlib import Path
import sys,argparse,json
p=argparse.ArgumentParser();p.add_argument('--project',default='/home/odroid/bpx-tgi-fpm');p.add_argument('--ckpt',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seconds',type=float,default=10);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.project);sys.path[:0]=[str(root/'tgi/src'),str(root/'fpm/src')]
import torch
from tgi.env.env_cfg import make_isaac_env_cfg
from tgi.env.sim import IsaacLocoSim
from tgi.policy_io import load_policy_bundle
from tgi.mdp.mdp import unflatten_state
from tgi.robot import NUM_J,FOOT_RADIUS,FOOT_HALF_LENGTH

class NominalSim(IsaacLocoSim):
    def _randomize(self):
        import warp as wp
        from newton import ModelFlags
        N,dev=self.N,self.device
        model=self.env.sim.backend_manager.get_model()
        mu=wp.to_torch(self.robot._root_view.get_attribute('shape_material_mu',model)[:,0]);mu[:]=.8
        self.env.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        self.fr=torch.full((N,),.8,device=dev);self.com=torch.zeros(N,3,device=dev);self.dmass=torch.zeros(N,device=dev)
        self.jf=torch.zeros(N,device=dev);self.arm=torch.ones(N,device=dev);self.bias=torch.zeros(N,NUM_J,device=dev);self.lag=torch.zeros(N,dtype=torch.long,device=dev)
        self.act_term.lag[:]=0

torch.set_num_threads(2)
b=load_policy_bundle(a.ckpt,a.device)
c=make_isaac_env_cfg(2);c.sim.device=a.device
c.scene.terrain.terrain_generator.border_width=0.;c.scene.terrain.terrain_generator.num_rows=1;c.scene.terrain.terrain_generator.num_cols=1
rough=c.scene.terrain.terrain_generator.sub_terrains['random_rough'];rough.noise_range=(0.,0.)
c.scene.terrain.terrain_generator.sub_terrains={'random_rough':rough}
c.events.reset_base.params['pose_range']={k:(0.,0.) for k in c.events.reset_base.params['pose_range']}
c.events.reset_base.params['velocity_range']={k:(0.,0.) for k in c.events.reset_base.params['velocity_range']}
c.events.reset_robot_joints.params['velocity_range']=(0.,0.)
sim=NominalSim(2,a.device,cfg=c,sgs_spawns=1,sgs_dirs=1,play=True,term_angle=b.args.get('term_angle',.7))
runner=b.make_runner(sim,noise=False)
commands=torch.tensor([[.3,0.,0.],[0.,0.,.3]],device=a.device)
term=sim.env.command_manager.get_term('base_velocity')
logs=[[],[]];alive=torch.ones(2,dtype=torch.bool,device=a.device);fell=[False,False]
start=sim.flat_state()[:,:3].clone();last=start.clone()
with torch.no_grad():
    for step in range(round(a.seconds/sim.dt)):
        term.is_heading_env[:]=False;term.is_standing_env[:]=False;term.vel_command_b[:]=commands
        S=sim.state();v=(S[1].transpose(-1,-2)@S[2][...,None]).squeeze(-1);w=(S[1].transpose(-1,-2)@S[3][...,None]).squeeze(-1)
        con=sim.foot_contact();fp,fv,rot,_=runner.mdp.kin.forward(*S)
        az=rot[:,1:5,2,1].clamp(-1,1);extent=FOOT_RADIUS*(1-az.square()).clamp_min(1e-8).sqrt()+FOOT_HALF_LENGTH*az.abs();clear=fp[:,:,2]-extent-sim.env.scene.env_origins[:,None,2]
        for i in range(2):
            if alive[i]:logs[i].append({'v':v[i].cpu().tolist(),'wz':float(w[i,2]),'contact':con[i].cpu().tolist(),'clearance':clear[i].cpu().tolist(),'foot_speed':fv[i,:,:2].norm(dim=-1).cpu().tolist()})
        tr=runner.step(runner.act().clamp(-10,10));last=torch.where(alive[:,None],tr['s1'][:,:3],last)
        for i in range(2):
            if alive[i] and tr['done'][i]:fell[i]=bool(tr['term'][i]);alive[i]=False
        if not alive.any():break
results=[]
for i,name in enumerate(['forward','turn']):
    rows=logs[i];data=torch.tensor([r['v'] for r in rows]);w=torch.tensor([r['wz'] for r in rows]);contact=torch.tensor([r['contact'] for r in rows]);clear=torch.tensor([r['clearance'] for r in rows]);speed=torch.tensor([r['foot_speed'] for r in rows]);warm=min(100,len(rows)-1)
    results.append({'case':name,'command':commands[i].cpu().tolist(),'seconds':len(rows)*sim.dt,'fell':fell[i],'mean_xy_error_mps':float((data[:,:2]-commands[i,:2].cpu()).norm(dim=-1).mean()),'mean_yaw_rate_error_radps':float((w-commands[i,2].cpu()).abs().mean()),'mean_body_velocity_after_2s':data[warm:].mean(0).tolist(),'contact_share_after_2s':contact[warm:].mean(0).tolist(),'clearance_p95_cm':(clear[warm:].quantile(.95,dim=0)*100).tolist(),'stance_xy_speed_mean_mps':[(float(speed[warm:,f][contact[warm:,f]>0].mean()) if (contact[warm:,f]>0).any() else None) for f in range(4)],'world_displacement':(last[i]-start[i]).cpu().tolist()})
report={'backend':'Isaac Lab Newton, same collector/runner as training','device':a.device,'nominal_physics':True,'terrain':'flat','observation_noise':False,'checkpoint':a.ckpt,'results':results}
Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
sim.env.close()

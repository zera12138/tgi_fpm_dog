"""CPU MuJoCo flat-ground check of a BPX checkpoint; not a terrain benchmark."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
import torch,numpy as np,mujoco
from tgi.robot import *
from tgi.policy_io import load_policy_bundle,bootstrap_footholds,perceive
from tgi.mdp.mdp import unflatten_state
from tgi.mdp.kinematics import RobotKinematics
from tgi.mdp.heightmap import HeightMap
from tgi.mdp.terrain import TerrainFeatures
from tgi.mdp.mdp import MDP,RewardCfg
from tgi.mdp.interface import FootholdScan,FullHeightScan
from tgi.mdp.history import append_history
from tgi.mdp.memory import init_memory,update_memory

p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--seconds',type=float,default=10);p.add_argument('--cmd',nargs=3,type=float,default=[.3,0,0]);p.add_argument('--out');a=p.parse_args()
torch.set_num_threads(2);b=load_policy_bundle(a.ckpt,'cpu');m=mujoco.MjModel.from_xml_path(MJCF)
kin=RobotKinematics(m,list(JOINT_NAMES),'cpu')
hm=HeightMap(torch.zeros(101,101),(-2.,-2.),.04);hm.build_support_field()
reward_cfg=RewardCfg(**{k:v for k,v in b.args.items() if k in RewardCfg.__dataclass_fields__})
use_mem=bool(b.args.get('mem',0))
mdp=MDP(kin,TerrainFeatures(hm,'cpu'),torch.tensor(DEFAULT_Q),list(JOINT_NAMES),kin.q_lo,kin.q_hi,torch.full((12,),ACTION_SCALE),reward_cfg,use_mem=use_mem)
mdp.scan=(FullHeightScan if b.args.get('scan')=='full' else FootholdScan)(hm,'cpu')
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
cmd=torch.tensor([a.cmd]);ap=torch.zeros(1,12);xi=torch.zeros(1,21);xi[:,0]=.8;xi[:,-3]=1
err=[];contacts=[];fell=False
mem=init_memory(1,'cpu') if use_mem else None
history=None
foot_bids=[m.body(n).id for n in FOOT_BODIES]
with torch.no_grad():
    for step in range(int(a.seconds/CONTROL_DT)):
        mujoco.mj_forward(m,d)
        R=d.xmat[m.body('torso').id].reshape(3,3)
        flat=np.concatenate((d.qpos[:7],d.qvel[:3],R@d.qvel[3:6],d.qpos[7:],d.qvel[6:]))
        S=unflatten_state(torch.tensor(flat,dtype=torch.float32)[None]);K=mdp.context(S)
        if mdp.termination(S,K).item():fell=True;break
        fh=b.fpm(mdp.system_state(S,cmd)) if b.fpm and b.ck.get('fpm_active',False) else bootstrap_footholds(1,'cpu')
        if use_mem and step>0:
            net=np.zeros((1,4,3));force=np.zeros(6)
            for ci,contact in enumerate(d.contact):
                mujoco.mj_contactForce(m,d,ci,force)
                fw=contact.frame.reshape(3,3).T@force[:3]
                for geom,sign in ((contact.geom1,-1),(contact.geom2,1)):
                    body_id=int(m.geom_bodyid[geom])
                    if body_id in foot_bids: net[0,foot_bids.index(body_id)]+=sign*fw
            con=(np.linalg.norm(net,axis=2)>1).astype('float32')
            mem=update_memory(mem,torch.from_numpy(con),CONTROL_DT,mdp.cfg.gait_period)
        o=mdp.proprio(S,ap,cmd,xi,noise=False);e=perceive(mdp,S,K,fh,noise=False,mem=mem,cmd=cmd)
        history=append_history(history,o,b.pi.hist)
        ap=mdp.jmap.to_mj(b.pi(history,e)).clamp(-ACTION_CLIP,ACTION_CLIP)
        target=np.array(DEFAULT_Q)+ACTION_SCALE*ap[0].numpy()
        for _ in range(round(CONTROL_DT/PHYSICS_DT)):
            d.ctrl[:]=np.clip(KP*(target-d.qpos[7:])-KD*d.qvel[6:],-EFFORT_LIMIT,EFFORT_LIMIT)
            mujoco.mj_step(m,d)
        err.append(float(np.linalg.norm((R.T@d.qvel[:3])[:2]-np.array(a.cmd[:2]))))
report={'checkpoint':str(Path(a.ckpt).resolve()),'test':'MuJoCo flat ground, fixed physics, no sensor noise','tilt_limit_rad':mdp.cfg.term_angle,'fpm_active':bool(b.ck.get('fpm_active',False)),'command':a.cmd,'simulated_seconds':len(err)*CONTROL_DT,'fell':fell,'mean_xy_velocity_error':float(np.mean(err)) if err else None,'final_root_xyz':d.qpos[:3].tolist(),'not_evaluated':['rough-terrain success','FPM held-out touchdown accuracy','real camera map','hardware']}
print(json.dumps(report,indent=2))
if a.out:Path(a.out).write_text(json.dumps(report,indent=2))

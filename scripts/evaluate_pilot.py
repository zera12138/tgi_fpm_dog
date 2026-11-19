"""Frozen-checkpoint MuJoCo evaluation using BPX TGI's perception, FK and label code.
No training or hardware IO. The XML/PD backend is a transfer check, not Newton parity.
"""
from pathlib import Path
import argparse,sys,json,math
import numpy as np
import torch,mujoco
p=argparse.ArgumentParser();p.add_argument('--project',default='/home/odroid/bpx-tgi-fpm');p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=20);p.add_argument('--iters',default='10000');p.add_argument('--cases',default='forward,stand,reverse,turn')
a=p.parse_args();root=Path(a.project);sys.path[:0]=[str(root/'tgi/src'),str(root/'fpm/src')]
from tgi.robot import *
from tgi.policy_io import load_policy_bundle,bootstrap_footholds,perceive
from tgi.mdp.mdp import MDP,RewardCfg,unflatten_state
from tgi.mdp.kinematics import RobotKinematics
from tgi.mdp.heightmap import HeightMap
from tgi.mdp.terrain import TerrainFeatures
from tgi.mdp.interface import FootholdScan
from tgi.foot_pred.labels import foothold_labels
from tgi.mdp.memory import init_memory, update_memory
from tgi.mdp.history import append_history
from tgi.world_model.buffer import ReplayBuffer

torch.set_num_threads(2)

def evaluate(iteration,case,command):
    ck=Path(a.run)/f'ckpt_{iteration:06d}.pt';b=load_policy_bundle(str(ck),'cpu')
    model=mujoco.MjModel.from_xml_path(MJCF);data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
    use_mem=bool(b.args.get('mem',0))
    kin=RobotKinematics(model,list(JOINT_NAMES),'cpu')
    hm=HeightMap(torch.zeros(101,101),(-2.,-2.),.04);hm.build_support_field()
    rc=RewardCfg(**{k:v for k,v in b.args.items() if k in RewardCfg.__dataclass_fields__})
    mdp=MDP(kin,TerrainFeatures(hm,'cpu'),torch.tensor(DEFAULT_Q),list(JOINT_NAMES),kin.q_lo,kin.q_hi,torch.full((12,),ACTION_SCALE),rc,use_mem=use_mem)
    mdp.scan=FootholdScan(hm,'cpu')
    cmd=torch.tensor([command],dtype=torch.float32);ap=torch.zeros(1,12);xi=torch.zeros(1,21);xi[:,0]=.8;xi[:,-3]=1.
    bid=model.body('torso').id;foot_bids=[model.body(n).id for n in FOOT_BODIES]
    active=b.fpm is not None and b.ck.get('fpm_active',False)
    records=[];fell=False
    mem=init_memory(1,'cpu') if use_mem else None
    history=None
    with torch.no_grad():
        for step in range(round(a.seconds/CONTROL_DT)):
            mujoco.mj_forward(model,data)
            R=data.xmat[bid].reshape(3,3).copy()
            flat=np.concatenate((data.qpos[:7],data.qvel[:3],R@data.qvel[3:6],data.qpos[7:],data.qvel[6:])).astype(np.float32)
            S=unflatten_state(torch.from_numpy(flat)[None]);K=mdp.context(S)
            if mdp.termination(S,K).item():fell=True;break
            # Same 1 N net-force threshold used by the Newton collector.
            net=np.zeros((4,3));force=np.zeros(6)
            for ci,contact in enumerate(data.contact):
                mujoco.mj_contactForce(model,data,ci,force)
                fw=contact.frame.reshape(3,3).T@force[:3]
                for geom,sign in ((contact.geom1,-1),(contact.geom2,1)):
                    body=int(model.geom_bodyid[geom])
                    if body in foot_bids:net[foot_bids.index(body)]+=sign*fw
            contact=(np.linalg.norm(net,axis=1)>1).astype(np.float32)
            if use_mem and step>0:   # 与训练一致: 首个观测用 mem_0, 之后才是 update(mem, c_t)
                mem=update_memory(mem,torch.from_numpy(contact)[None],CONTROL_DT,mdp.cfg.gait_period)
            prediction=b.fpm(mdp.system_state(S,cmd)) if b.fpm else bootstrap_footholds(1,'cpu')
            used=prediction if active else bootstrap_footholds(1,'cpu')
            vb=R.T@data.qvel[:3];wb=data.qvel[3:6].copy()
            records.append((flat,contact,prediction[0].numpy().copy(),vb.copy(),float(wb[2]),float(math.atan2(R[1,0],R[0,0]))))
            o=mdp.proprio(S,ap,cmd,xi,noise=False);extra=perceive(mdp,S,K,used,noise=False,mem=mem,cmd=cmd)
            history=append_history(history,o,b.pi.hist)
            ap=mdp.jmap.to_mj(b.pi(history,extra)).clamp(-ACTION_CLIP,ACTION_CLIP)
            target=np.array(DEFAULT_Q)+ACTION_SCALE*ap[0].numpy()
            for _ in range(round(CONTROL_DT/PHYSICS_DT)):
                data.ctrl[:]=np.clip(KP*(target-data.qpos[7:])-KD*data.qvel[6:],-EFFORT_LIMIT,EFFORT_LIMIT)
                mujoco.mj_step(model,data)
    T=len(records)
    if T<2:return dict(iteration=iteration,case=case,fell=fell,seconds=T*CONTROL_DT)
    states=np.stack([r[0] for r in records]);con=np.stack([r[1] for r in records]);pred=np.stack([r[2] for r in records]);vel=np.stack([r[3] for r in records]);wz=np.array([r[4] for r in records]);yaw=np.unwrap([r[5] for r in records])
    buf=ReplayBuffer(T,1,'cpu');buf.size=T;buf.ptr=0
    buf.data['s1'][:,0]=torch.from_numpy(states);buf.data['c1'][:,0]=torch.from_numpy(con);buf.data['done'][-1]=True
    labels,valid,_=foothold_labels(buf,kin);labels=labels[:,0].numpy();valid=valid[:,0].numpy()
    warm=min(round(2/CONTROL_DT),T);valid[:warm]=False
    errors=np.linalg.norm(pred-labels,axis=-1)
    seq=unflatten_state(torch.from_numpy(states));fp,fv,rot,_=kin.forward(*seq)
    current_xy=(seq[1].transpose(-1,-2)[:,None]@(fp-seq[0][:,None])[...,None]).squeeze(-1)[...,:2].numpy()
    current_errors=np.linalg.norm(current_xy-labels,axis=-1)
    az=rot[:,1:5,2,1].clamp(-1,1)
    extent=FOOT_RADIUS*(1-az.square()).clamp_min(1e-8).sqrt()+FOOT_HALF_LENGTH*az.abs()
    clearance=(fp[:,:,2]-extent).numpy();foot_speed=fv[:,:,:2].norm(dim=-1).numpy()
    perfoot={}
    for f,name in enumerate(LEGS):
        e=errors[:,f][valid[:,f]]
        touchdown=np.flatnonzero((con[1:,f]>0)&(con[:-1,f]==0))+1
        event_errors=[];swing_seconds=[];baseline_errors=[]
        # One prediction at the start of each observed >=60 ms swing, excluding spawn settling.
        for end in touchdown:
            start=end-1
            while start>0 and con[start-1,f]==0:start-=1
            if start>=warm and end-start>=3 and valid[start,f]:
                event_errors.append(float(errors[start,f]));swing_seconds.append((end-start)*CONTROL_DT);baseline_errors.append(float(current_errors[start,f]))
        perfoot[name]={'valid_timestamp_labels':int(len(e)),'eligible_timestamps':max(T-warm,0),'contact_share_after_warmup':float(con[warm:,f].mean()) if T>warm else None,'timestamp_xy_mean_cm':float(e.mean()*100) if len(e) else None,'timestamp_xy_p95_cm':float(np.percentile(e,95)*100) if len(e) else None,'touchdowns_after_warmup':int((touchdown>=warm).sum()),'swings_at_least_60ms':len(event_errors),'liftoff_xy_mean_cm':float(np.mean(event_errors)*100) if event_errors else None,'liftoff_xy_p95_cm':float(np.percentile(event_errors,95)*100) if event_errors else None,'swing_seconds_mean':float(np.mean(swing_seconds)) if swing_seconds else None}
        stance=(np.arange(T)>=warm)&(con[:,f]>0)
        perfoot[name].update(current_foot_baseline_liftoff_mean_cm=float(np.mean(baseline_errors)*100) if baseline_errors else None,
                             stance_xy_speed_mean_mps=float(foot_speed[:,f][stance].mean()) if stance.any() else None,
                             clearance_p95_cm=float(np.percentile(clearance[warm:,f],95)*100) if T>warm else None,
                             clearance_max_cm=float(clearance[warm:,f].max()*100) if T>warm else None)
    err=np.linalg.norm(vel[:,:2]-np.array(command[:2]),axis=-1)
    result={'iteration':iteration,'case':case,'command':command,'fpm_used_by_policy':active,'seconds':T*CONTROL_DT,'fell':fell,'mean_xy_velocity_error_mps':float(err.mean()),'mean_xy_velocity_error_after_2s_mps':float(err[warm:].mean()) if T>warm else None,'mean_yaw_rate_error_radps':float(np.abs(wz-command[2]).mean()),'mean_body_velocity_after_2s':vel[warm:].mean(0).tolist() if T>warm else None,'final_root_xyz':data.qpos[:3].tolist(),'yaw_change_deg':float((yaw[-1]-yaw[0])*180/math.pi),'foot_metrics':perfoot}
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{iteration}_{case}.npz',states=states,contacts=con,predictions=pred,labels=labels,valid=valid,velocities=vel,yaw=yaw,yaw_rate=wz)
    print(json.dumps(result),flush=True)
    return result

CMDS={'forward':[.3,0.,0.],'stand':[0.,0.,0.],'reverse':[-.2,0.,0.],'turn':[0.,0.,.3]}
results=[]
for it in [int(x) for x in a.iters.split(',') if x.strip()]:
    for case in [c.strip() for c in a.cases.split(',') if c.strip()]:
        results.append(evaluate(it,case,CMDS[case]))
report={'run':a.run,'backend':'MuJoCo CPU, flat ground, fixed physics, no observation noise','contact_definition':'toe net contact force >1N at 50Hz','label_definition':'upcoming touchdown for swing; after next liftoff for stance; original TGI label function','error_reference':'toe cylinder centre xy in prediction-time root frame','liftoff_filter':'ignore first 2s; one prediction at start of each observed swing lasting >=60ms','limitations':['deterministic single rollout per condition','simulator transfer, not hardware','flat ground only','not an FPM causal ablation','FPM errors are out-of-buffer rollout errors, not measured robot accuracy'],'results':results}
Path(a.out,'evaluation.json').write_text(json.dumps(report,indent=2))

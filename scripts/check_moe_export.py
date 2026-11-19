"""验证MoE的TorchScript/ONNX及带历史的控制器；需要含onnxruntime的环境。
用法：python scripts/check_moe_export.py --out /tmp/bpx-moe-export-check [--baseline CHECKPOINT]
产物为test_only检查点，不可用于训练恢复。测试非零历史支路，不证明训练效果。
"""
from pathlib import Path
import argparse,sys,os,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
import numpy as np
import torch
from tgi.policy_io import make_policy,policy_spec
from tgi.world_model.moe import migrate_mlp_to_moe
from tgi.foot_pred import FootholdPredictor
from tgi.deploy.controller import Controller
from tgi.mdp.interface import JointMap,policy_proprio,system_state,FootholdScan
from tgi.mdp.history import append_history
from tgi.mdp.memory import init_memory,update_memory
from tgi.mdp.rotations import so3_exp,mat_to_quat
from tgi.robot import DEFAULT_Q,ACTION_SCALE,JOINT_NAMES,INTERFACE

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--baseline');a=p.parse_args()
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(41)
pi=make_policy('cpu',True,'moe',5);fp=FootholdPredictor()
args=dict(scan='foothold',mem=1,policy='moe',history_length=5,gait_period=.6,gait_cmd_threshold=.1)
if a.baseline:
 ck0=torch.load(a.baseline,map_location='cpu',weights_only=False)
 if ck0['args'].get('policy','mlp')!='mlp':raise ValueError('Baseline must be an MLP')
 migrate_mlp_to_moe(pi,ck0['pi']);fp.load_state_dict(ck0['fpm'])
 args={**ck0['args'],**args}
else:
 pi.update_statistics(torch.randn(64,5,48),torch.randn(64,206))
 fp.norm.update(torch.randn(64,36))
# A nonzero latent branch is required to expose history ordering/reset bugs in parity checks.
with torch.no_grad():pi.net[0].weight[:,-32:].normal_(std=.003)
ck=dict(interface=INTERFACE,args=args,policy_spec=policy_spec(pi),pi=pi.state_dict(),fpm=fp.state_dict(),
        fpm_active=True,iteration=0,wall_time=0,test_only=True)
path=out/'test_moe.pt';torch.save(ck,path)
env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path[:2])}
subprocess.run([sys.executable,str(ROOT/'tgi/scripts/export.py'),'--ckpt',str(path),'--out',str(out)],env=env,check=True)
ctrl=Controller(str(out),height_fn=lambda x,y:.08*x-.04*y)
ts=torch.jit.load(str(out/'exported/policy.ts'))
h=torch.randn(32,5,48);e=torch.randn(32,206);flat=torch.cat((h.flatten(1),e),-1)
with torch.no_grad():expected=pi(h,e).numpy();scripted=ts(flat).numpy()
actual=np.concatenate([ctrl.pi.run(None,{'obs':row[None]})[0] for row in flat.numpy()])
err_onnx=float(np.abs(actual-expected).max());err_ts=float(np.abs(scripted-expected).max())
class Plane:
 def query_nearest(self,pts):return .08*pts[...,0]-.04*pts[...,1]
scan=FootholdScan(Plane(),'cpu');jm=JointMap(list(JOINT_NAMES),'cpu');q0=torch.tensor(DEFAULT_Q)
p0=torch.tensor([[.1,-.15,.43]]);R=so3_exp(torch.tensor([[.13,-.1,.7]]));quat=mat_to_quat(R)[0].numpy()
history=None;mem=init_memory(1,'cpu');ap=torch.zeros(1,12);maxa=maxfh=maxt=0.;resets=0
with torch.no_grad():
 for i in range(300):
  if i in (71,203):ctrl.reset();history=None;mem=init_memory(1,'cpu');ap=torch.zeros(1,12);resets+=1
  cmd=torch.tensor([[.3,0.,0.]]) if i<100 else torch.zeros(1,3) if i<200 else torch.tensor([[0.,0.,.3]])
  q=q0[None]+.03*np.sin(i*.07);qd=torch.ones(1,12)*.01*np.cos(i*.07)
  v=torch.tensor([[.2+.03*np.sin(i*.11),.1,.02]],dtype=torch.float32);w=torch.tensor([[.1,.03,.2]])
  S=(p0,R,v,w,q,qd);vb=(R.transpose(-1,-2)@v[...,None])[0,:,0].numpy();wb=(R.transpose(-1,-2)@w[...,None])[0,:,0].numpy()
  tp,tr=ctrl.torso_pose(p0[0].numpy(),quat,q[0].numpy());tp=torch.tensor(tp,dtype=torch.float32)[None];tr=torch.tensor(tr,dtype=torch.float32)[None]
  fh=fp(system_state(S,cmd,q0,jm));extra=scan(p0,R,tp,tr,fh,noise=False)
  active=(cmd[:,:2].norm(dim=-1)+cmd[:,2].abs()>.1)[:,None]
  obs=policy_proprio(S,ap,cmd,q0,jm,noise=False);history=append_history(history,obs,pi.hist)
  act=pi(history,torch.cat((extra,mem[:,12:14]*active),-1))
  target=ctrl.step(p0[0].numpy(),quat,vb,wb,q[0].numpy(),qd[0].numpy(),cmd[0].numpy())
  maxa=max(maxa,float(np.max(np.abs(ctrl.a_prev-act[0].numpy()))));maxfh=max(maxfh,float(np.max(np.abs(ctrl.last_fh-fh[0].numpy()))))
  maxt=max(maxt,float(np.max(np.abs(target-(q0+ACTION_SCALE*act[0].clamp(-10,10)).numpy()))))
  ap=act;mem=update_memory(mem,torch.ones(1,4),.02,.6)
res=dict(actor_input=ctrl.pi.get_inputs()[0].shape,history_frames=pi.hist,onnx_max_error=err_onnx,torchscript_max_error=err_ts,
         controller_steps=300,reset_count=resets,controller_action_max_error=maxa,controller_target_max_error=maxt,fpm_max_error=maxfh,
         test='nonzero history branch, variable proprioception, sloped map, forward/stand/yaw; not a dynamics rollout')
(out/'parity.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
assert err_onnx<1e-3 and err_ts<1e-5 and maxa<1e-3 and maxt<1e-3 and maxfh<1e-4,res

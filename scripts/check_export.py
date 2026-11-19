"""Export/ONNX/controller parity on random test weights (never a trained policy)."""
from pathlib import Path
import sys,tempfile,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
import torch,numpy as np,onnxruntime as ort
from tgi.robot import *
from tgi.policy_io import make_policy
from tgi.foot_pred.model import FootholdPredictor
from tgi.deploy.controller import Controller
from tgi.mdp.interface import JointMap,policy_proprio,system_state,FootholdScan
from tgi.mdp.rotations import so3_exp,mat_to_quat

torch.set_num_threads(2);torch.manual_seed(5)
pi=make_policy('cpu');fp=FootholdPredictor()
pi.obs_norm.update(torch.randn(64,48));pi.extra_norm.update(torch.randn(64,204));fp.norm.update(torch.randn(64,36))
ck={'interface':INTERFACE,'args':{'scan':'foothold'},'pi':pi.state_dict(),'fpm':fp.state_dict(),'fpm_active':True,'iteration':0,'wall_time':0}
class Plane:
    def query_nearest(self,pts):return .08*pts[...,0]-.04*pts[...,1]
with tempfile.TemporaryDirectory(prefix='bpx-export-check-') as tmp:
    path=Path(tmp)/'test_only.pt';torch.save(ck,path)
    import os
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path[:2]))
    subprocess.run([sys.executable,str(ROOT/'tgi/scripts/export.py'),'--ckpt',str(path),'--out',tmp],check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_bpx.py'),'--ckpt',str(path),'--seconds','1'],check=True,env=env)
    ctrl=Controller(tmp,height_fn=lambda x,y:.08*x-.04*y)
    p=torch.tensor([[.1,-.15,.43]]);R=so3_exp(torch.tensor([[.13,-.1,.7]]))
    v=torch.tensor([[.2,.1,.02]]);w=torch.tensor([[.1,.03,.2]]);q=torch.tensor(DEFAULT_Q)[None]+.03;qd=torch.randn(1,12)*.2;cmd=torch.tensor([[.3,0,.1]])
    S=(p,R,v,w,q,qd);jm=JointMap(list(JOINT_NAMES),'cpu');q0=torch.tensor(DEFAULT_Q)
    x=system_state(S,cmd,q0,jm);fh=fp(x).detach()
    scan=FootholdScan(Plane(),'cpu')(p,R,p,R,fh,noise=False)
    scan_np=ctrl.foothold_scan(p[0].numpy(),R[0].numpy(),p[0].numpy(),R[0].numpy(),fh[0].numpy())
    np.testing.assert_allclose(scan[0].numpy(),scan_np,atol=2e-6)
    o=policy_proprio(S,torch.zeros(1,12),cmd,q0,jm,noise=False)
    a=pi(o[:,None],scan).detach()[0].numpy()
    target=ctrl.step(p[0].numpy(),mat_to_quat(R)[0].numpy(),(R.transpose(-1,-2)@v[...,None])[0,:,0].numpy(),(R.transpose(-1,-2)@w[...,None])[0,:,0].numpy(),q[0].numpy(),qd[0].numpy(),cmd[0].numpy())
    np.testing.assert_allclose(ctrl.last_fh,fh[0].numpy(),atol=2e-6)
    np.testing.assert_allclose(ctrl.a_prev,a,atol=2e-6)
    np.testing.assert_allclose(target,np.array(DEFAULT_Q)+ACTION_SCALE*np.clip(a,-10,10),atol=2e-6)
    print('EXPORT CHECK PASSED: normalized PyTorch / TorchScript / ONNX and 4-patch controller agree')

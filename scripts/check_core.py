"""CPU regression checks: BPX FK, contact labels, differentiable models and standing."""
from pathlib import Path
import sys, unittest, json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
import numpy as np
import torch
import mujoco
from tgi.robot import *
from tgi.mdp.kinematics import RobotKinematics
from tgi.mdp.rotations import quat_to_mat, so3_exp
from tgi.mdp.mdp import MDP, flatten_state, unflatten_state, normalize_xi
from tgi.mdp.interface import *
from tgi.mdp.heightmap import HeightMap
from tgi.mdp.terrain import TerrainFeatures
from tgi.world_model.networks import WorldModel
from tgi.world_model.buffer import ReplayBuffer
from tgi.foot_pred.labels import foothold_labels
from tgi.foot_pred.model import FootholdPredictor
from tgi.foot_pred.trainer import FPMTrainer
from tgi.policy_io import make_policy, Bundle

def fixture(n=3):
    m=mujoco.MjModel.from_xml_path(MJCF)
    kin=RobotKinematics(m,list(JOINT_NAMES),'cpu')
    hm=HeightMap(torch.zeros(101,101),(-2.,-2.),.04);hm.build_support_field()
    terrain=TerrainFeatures(hm,'cpu');q=torch.tensor(DEFAULT_Q)
    mdp=MDP(kin,terrain,q,list(JOINT_NAMES),kin.q_lo,kin.q_hi,torch.full((12,),ACTION_SCALE))
    mdp.scan=FootholdScan(hm,'cpu')
    S=(torch.tensor([0.,0.,NOMINAL_HEIGHT]).repeat(n,1),torch.eye(3).repeat(n,1,1),torch.zeros(n,3),torch.zeros(n,3),q.repeat(n,1),torch.zeros(n,12))
    return m,kin,mdp,S

class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);np.random.seed(7)
    def test_fk_position_velocity_and_gradient(self):
        m,kin,_,S=fixture(8)
        p,R,v,w,q,qd=S
        p+=torch.randn_like(p)*.2;R=so3_exp(torch.randn(8,3)*.4)
        v=torch.randn_like(v);w=torch.randn_like(w);q=q+torch.randn_like(q)*.1;qd=torch.randn_like(qd)
        S=(p,R,v,w,q.requires_grad_(),qd)
        fp,fv,_,_=kin.forward(*S)
        d=mujoco.MjData(m)
        from tgi.mdp.rotations import mat_to_quat
        for i in range(8):
            d.qpos[:3]=p[i];d.qpos[3:7]=mat_to_quat(R[i]);d.qpos[7:]=q[i].detach()
            d.qvel[:3]=v[i];d.qvel[3:6]=R[i].T@w[i];d.qvel[6:]=qd[i]
            mujoco.mj_forward(m,d)
            for f,name in enumerate(FOOT_SITES):
                sid=m.site(name).id
                np.testing.assert_allclose(fp[i,f].detach(),d.site_xpos[sid],atol=2e-6)
                jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,sid)
                np.testing.assert_allclose(fv[i,f].detach(),jp@d.qvel,atol=3e-6)
        fp.sum().backward();self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(q.grad.abs().sum().item(),1.)
    def test_reflection_geometry(self):
        _,kin,_,S=fixture();p,R,v,w,q,qd=S
        q=q+torch.randn_like(q)*.15
        f=kin.forward(p,R,v,w,q,qd)[0]
        fm=kin.forward(p,R,v,w,mirror_joints(q),qd)[0]
        torch.testing.assert_close(fm,f[:,FOOT_PERM]*torch.tensor([1.,-1.,1.]),atol=2e-5,rtol=1e-4)
        for fun,n in ((mirror_proprio,48),(mirror_scan,204)):
            x=torch.randn(5,n);torch.testing.assert_close(fun(fun(x)),x)
    def test_shapes_and_imagination_gradients(self):
        _,_,mdp,S=fixture();n=3;xi=torch.zeros(n,21);xi[:,0]=.8;xi[:,-3]=1.
        cmd=torch.ones(n,3)*.1;a=torch.zeros(n,12,requires_grad=True)
        self.assertEqual(flatten_state(S).shape,(n,37))
        o=mdp.proprio(S,a,cmd,xi,noise=False);x=mdp.system_state(S,cmd)
        self.assertEqual(o.shape,(n,48));self.assertEqual(x.shape,(n,36))
        fpm=FootholdPredictor();fh=fpm(x);K=mdp.context(S)
        extra=mdp.scan(S[0],S[1],K['torso_p'],K['torso_R'],fh,noise=False)
        self.assertEqual(extra.shape,(n,204));pi=make_policy('cpu')
        self.assertEqual(pi(o[:,None],extra).shape,(n,12))
        wm=WorldModel(mdp.terrain.wm_dim,hidden=(32,32))
        torch.nn.init.normal_(wm.net[-1].weight,std=.001)
        for _ in range(3):S,c=wm.step(S,mdp.wm_terrain(S),a,a,normalize_xi(xi))
        self.assertEqual(c.shape,(n,4));K=mdp.context(S)
        feat=mdp.critic_features(S,K,c.sigmoid(),cmd,normalize_xi(xi))
        self.assertEqual(feat.shape,(n,mdp.critic_dim))
        loss=mdp.reward(S,K,c.sigmoid(),a,torch.zeros_like(a),cmd,.02).sum()
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(a.grad.abs().sum().item(),0.)
    def test_labels_independent_feet_terminal_and_wrap(self):
        # Independent brute-force oracle: stance must first lift off; terminal forbids crossing resets.
        _,kin,_,S=fixture(1);b=ReplayBuffer(13,1,'cpu')
        cs=torch.randint(0,2,(19,4)).float();cs[:,3]=0 # no HR touchdown: other feet remain valid
        for t in range(19):
            s=flatten_state(S).clone();s[:,0]=t*.03
            tr={k:torch.zeros_like(v[0]) for k,v in b.data.items()}
            tr.update(s=s,s1=s,c1=cs[t:t+1],done=torch.tensor([t in (9,17)]),ep_step=torch.tensor([t]))
            b.add(tr)
        labels,valid,_=foothold_labels(b,kin);rows=b._row(torch.arange(b.size))
        ss=b.data['s1'][rows];feet=kin.forward(*unflatten_state(ss))[0]
        for i,row in enumerate(rows):
            for f in range(4):
                expected=None;lifted=not bool(b.data['c1'][row,0,f])
                for j in range(i+1,len(rows)):
                    if b.data['done'][rows[j-1],0]:break
                    con=bool(b.data['c1'][rows[j],0,f])
                    if not con:lifted=True
                    if lifted and con:expected=j;break
                self.assertEqual(bool(valid[row,0,f]),expected is not None)
                if expected is not None:
                    torch.testing.assert_close(labels[row,0,f],(feet[expected,0,f]-ss[i,0,:3])[:2])
        self.assertTrue(valid.any());self.assertFalse(valid[:,:,3].any())
    def test_fpm_loss_decreases_and_export(self):
        f=FootholdPredictor();x=torch.randn(64,36);y=x[:,:8].reshape(64,4,2)*.1
        f.norm.update(x);opt=torch.optim.Adam(f.parameters(),lr=.003)
        start=(f(x)-y).square().mean().item()
        for _ in range(80):
            loss=(f(x)-y).square().mean();opt.zero_grad();loss.backward();opt.step()
        self.assertLess(loss.item(),start*.1)
        scripted=torch.jit.script(f.exportable())
        torch.testing.assert_close(scripted(x),f(x).flatten(-2))
        torch.testing.assert_close(scripted(x[:,None]),f(x).flatten(-2))
    def test_fpm_replay_training_and_stale_rows(self):
        _,kin,mdp,S=fixture(1);b=ReplayBuffer(16,1,'cpu')
        for t in range(16):
            tr={k:torch.zeros_like(v[0]) for k,v in b.data.items()}
            s=flatten_state(S).clone();s[:,0]=t*.01
            s1=s.clone();s1[:,0]+=.01
            tr.update(s=s,s1=s1,c1=torch.tensor([[(t+f)%4<2 for f in range(4)]]).float(),ep_step=torch.tensor([t]),cmd=torch.tensor([[.2,0.,0.]]))
            b.add(tr)
        model=FootholdPredictor();trainer=FPMTrainer(model);trainer.refresh_labels(b,kin)
        self.assertGreater(trainer.n_cand,0)
        before=[p.detach().clone() for p in model.parameters()]
        loss=trainer.update(b,mdp,32,steps=3)
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(loss),0.)
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(before,model.parameters())))
        # Refresh candidates become entirely stale after a full overwrite; zero weights prevent training leakage.
        for _ in range(16):b.add(tr)
        _,_,weights=trainer.sample(b,mdp,32)
        self.assertEqual(weights.sum().item(),0.)

    def test_g1_checkpoint_rejected(self):
        with self.assertRaises(ValueError):Bundle({'args':{}},'cpu')
    def test_pd_standing(self):
        m=mujoco.MjModel.from_xml_path(MJCF);d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
        peak=0.;heights=[]
        for k in range(2000):
            tau=KP*(np.array(DEFAULT_Q)-d.qpos[7:])-KD*d.qvel[6:]
            d.ctrl[:]=np.clip(tau,-EFFORT_LIMIT,EFFORT_LIMIT);peak=max(peak,float(np.abs(d.ctrl).max()))
            mujoco.mj_step(m,d)
            if k>1000:heights.append(float(d.qpos[2]))
        self.assertTrue(np.isfinite(d.qpos).all());self.assertGreater(min(heights),.30)
        self.assertGreater(d.xmat[m.body('torso').id].reshape(3,3)[2,2],.95)
        feet=set()
        for c in d.contact:
            for g in (c.geom1,c.geom2):
                name=m.body(m.geom_bodyid[g]).name
                if name in FOOT_BODIES:feet.add(name)
        self.assertEqual(len(feet),4)
        report={'duration_s':10,'settled_height_min':min(heights),'settled_height_max':max(heights),'max_torque_nm':peak,'contact_feet':sorted(feet),'note':'MuJoCo PD check only; not a trained walking policy or Newton parity result'}
        (ROOT/'docs/standing_check.json').write_text(json.dumps(report,indent=2))
        print('Standing:',report)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

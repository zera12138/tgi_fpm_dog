"""MoE集成CPU检查：迁移、历史/重置/回放、想象梯度及实际Trainer更新。

用法：./run.sh check-moe
不启动长训练，不改写已有检查点。
"""
from pathlib import Path
from types import SimpleNamespace as NS
import sys, unittest, copy, tempfile, time, json
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tgi/src'), str(ROOT/'fpm/src'), str(ROOT/'scripts')]
import torch
from check_core import fixture
from tgi.policy_io import make_policy, policy_spec, Bundle, SimRunner
from tgi.world_model.moe import migrate_mlp_to_moe
from tgi.world_model.networks import WorldModel, Critic
from tgi.world_model.buffer import ReplayBuffer
from tgi.mdp.history import append_history
from tgi.mdp.memory import MEM_DIM
from tgi.mdp.mdp import flatten_state
from tgi.mdp.interface import mirror_proprio, mirror_scan
from tgi.foot_pred import FootholdPredictor
from tgi.training.config import parse_args
from tgi.training.trainer import Trainer, ERR_FLOOR
from tgi.training.collector import Collector
from tgi.training.logger import TrainLogger, COLUMNS
from tgi.robot import INTERFACE


class FakeSim:
    """只测试采集/回放边界的确定性转移；真实动力学另用Newton验证。"""
    def __init__(self, mdp, state):
        self.S = state
        self.N, self.device, self.dt = len(state[0]), 'cpu', .02
        self.t = torch.zeros(self.N)
        self.step_index = 0
        self.a_prev = torch.zeros(self.N, 12)
        self.ep_len = torch.zeros(self.N, dtype=torch.long)
        self.env = NS(scene=NS(terrain=NS(terrain_levels=torch.zeros(self.N))))
    def state(self): return self.S
    def command(self): return torch.tensor([[.3,0.,0.]]).repeat(self.N,1)
    def xi(self):
        x = torch.zeros(self.N,21);x[:,0]=.8;x[:,-3]=1.
        return x
    def step(self, a):
        old = flatten_state(self.S).clone();ep = self.ep_len.clone();prev=self.a_prev.clone()
        self.S = (*self.S[:4], self.S[4]+.001, self.S[5])
        end = flatten_state(self.S).clone();c=torch.ones(self.N,4)
        self.t += self.dt;self.ep_len += 1;self.step_index += 1
        done=torch.zeros(self.N,dtype=torch.bool)
        if self.step_index % 3 == 0:
            done[0]=True;self.ep_len[0]=0
        self.a_prev=a.clone();self.a_prev[done]=0
        return dict(s=old,s1=end,a=a.clone(),a_prev=prev,xi=self.xi(),cmd=self.command(),c1=c,
                    done=done,ep_step=ep,r_sim=torch.zeros(self.N),term=done)


class Checks(unittest.TestCase):
    def setUp(self): torch.manual_seed(31)

    def test_resume_logs_have_header_and_persistent_gate_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            logger=TrainLogger(directory,True,25,[],'cpu')
            with patch('torch.cuda.max_memory_reserved',return_value=0):
                logger.write(10025,0.,100,2e-4,{'moe_use_0':.125,'moe_entropy':2.079})
            logger.csv.close()
            self.assertEqual((Path(directory)/'train_log.csv').read_text().splitlines()[0],COLUMNS)
            row=json.loads((Path(directory)/'extra_metrics.jsonl').read_text())
            self.assertEqual(row['iteration'],10025);self.assertEqual(row['moe_use_0'],.125)

    def test_history_reset_and_old_frames(self):
        h=None
        for i in range(4):
            o=torch.full((2,48),float(i));prior=h
            h=append_history(h,o,5,reset=torch.tensor([i==3,False]))
            if prior is not None:self.assertEqual(prior[1,-1,0],i-1)
        self.assertEqual(h[0,:,0].tolist(),[3]*5)
        self.assertEqual(h[1,:,0].tolist(),[0,0,1,2,3])
        torch.testing.assert_close(mirror_proprio(mirror_proprio(h)),h)
        one=append_history(None,o,1)
        torch.testing.assert_close(one[:,0],o)

    def test_migration_and_checkpoint_architecture(self):
        for old_mem in (False,True):
            old=make_policy('cpu',old_mem)
            old.obs_norm.update(torch.randn(64,48)+.4);old.extra_norm.update(torch.randn(64,old.extra_dim)*.3)
            new=make_policy('cpu',True,'moe',5);migrate_mlp_to_moe(new,old.state_dict())
            h=torch.randn(8,5,48);extra=torch.randn(8,206)
            torch.testing.assert_close(new(h,extra),old(h[:,-1:],extra[:,:old.extra_dim]),atol=1e-7,rtol=1e-6)
            self.assertEqual(int(torch.count_nonzero(new.net[0].weight[:,48+old.extra_dim:])),0)
            ck=dict(interface=INTERFACE,args=dict(mem=1,policy='moe',history_length=5,scan='foothold'),
                    policy_spec=policy_spec(new),pi=new.state_dict(),fpm=None)
            restored=Bundle(ck,'cpu');torch.testing.assert_close(restored.pi(h,extra),new(h,extra))
            bad=copy.deepcopy(ck);bad['args']['history_length']=4
            with self.assertRaises(ValueError):Bundle(bad,'cpu')
            with self.assertRaises(ValueError):migrate_mlp_to_moe(make_policy('cpu',False,'moe'),make_policy('cpu',True).state_dict())

    def test_experts_gates_and_history_gradients(self):
        pi=make_policy('cpu',True,'moe',5);h=torch.randn(7,5,48,requires_grad=True);extra=torch.randn(7,206)
        latent,gate=pi.encode(h)
        torch.testing.assert_close(gate.sum(1),torch.ones(7));self.assertEqual(latent.shape,(7,32))
        pi(h,extra).square().sum().backward()
        for group in (pi.experts,pi.gating_network):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()),0.)
        self.assertGreater(float(h.grad[:,:-1].abs().sum()),0.)
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_trainer_save_migrate_and_resume(self):
        def trainer(kind, mem, directory):
            t=Trainer.__new__(Trainer)
            t.cfg=parse_args(['--policy',kind,'--mem',str(int(mem)),'--log_dir',directory])
            t.dev='cpu';t.pi=make_policy('cpu',mem,kind)
            t.critic=Critic(214 if mem else 204,hidden=(32,32))
            t.wm=WorldModel(16,hidden=(32,32));t.fpm=None
            t.opt_pi=torch.optim.Adam(t.pi.parameters(),lr=2e-4,fused=True)
            t.opt_v=torch.optim.Adam(t.critic.parameters(),lr=2e-4,fused=True)
            t.opt_wm=torch.optim.Adam(t.wm.parameters(),lr=2e-4,fused=True)
            t.t_start=time.time();t.collector=NS(steps=123)
            return t
        def populate(module,opt):
            opt.zero_grad();sum(p.square().sum() for p in module.parameters()).backward();opt.step()
        with tempfile.TemporaryDirectory() as directory:
            old=trainer('mlp',False,directory)
            for m,opt in [(old.pi,old.opt_pi),(old.critic,old.opt_v),(old.wm,old.opt_wm)]:populate(m,opt)
            old.save(10000)
            new=trainer('moe',True,directory)
            iteration,_,steps=new._load(str(Path(directory)/'last.pt'))
            self.assertEqual((iteration,steps),(10000,123))
            self.assertFalse(new.opt_pi.state);self.assertFalse(new.opt_v.state)
            self.assertTrue(new.opt_wm.state)
            h=torch.randn(8,5,48);e=torch.randn(8,206);x=torch.randn(8,214)
            torch.testing.assert_close(new.pi(h,e),old.pi(h[:,-1:],e[:,:204]),atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(new.critic(x),old.critic(x[:,:204]),atol=1e-7,rtol=1e-6)
            populate(new.pi,new.opt_pi);populate(new.critic,new.opt_v)
            new.save(10001)
            resumed=trainer('moe',True,directory);resumed._load(str(Path(directory)/'last.pt'))
            torch.testing.assert_close(resumed.pi(h,e),new.pi(h,e))
            for a,b in [(new.opt_pi,resumed.opt_pi),(new.opt_v,resumed.opt_v),(new.opt_wm,resumed.opt_wm)]:
                self.assertEqual(len(a.state),len(b.state))
                for sa,sb in zip(a.state.values(),b.state.values()):
                    for k in sa:torch.testing.assert_close(sa[k],sb[k])
            thin=torch.load(Path(directory)/'ckpt_010001.pt',weights_only=False)
            self.assertEqual(thin['policy_spec'],policy_spec(new.pi))
            with self.assertRaises(ValueError):trainer('mlp',True,directory)._load(str(Path(directory)/'last.pt'))

    def make_collector(self, kind='moe', mem=True):
        _,_,mdp,S=fixture(2);mdp.use_mem=mem
        if mem:mdp.critic_dim+=10
        pi=make_policy('cpu',mem,kind,5);sim=FakeSim(mdp,S);wm=WorldModel(mdp.terrain.wm_dim,hidden=(32,32))
        torch.nn.init.normal_(wm.net[-1].weight,std=.0001)
        fields={'mem':(14,torch.float32)} if mem else {}
        if kind=='moe':fields['obs_hist']=((5,48),torch.float32)
        buf=ReplayBuffer(8,2,'cpu',fields)
        logger=NS(add_transition=lambda *a:None,add_level=lambda *a:None)
        collector=Collector(sim,mdp,pi,None,wm,buf,logger,action_noise=0.,threaded=False,mem=mem)
        return mdp,pi,sim,wm,buf,collector

    def test_collector_history_idempotence_wrap_and_reset(self):
        mdp,pi,sim,wm,buf,col=self.make_collector()
        col.runner.observe();h0=col.runner.obs_history.clone();col.runner.act()
        torch.testing.assert_close(col.runner.obs_history,h0)
        captured=[]
        original=buf.add
        def remember(tr):captured.append({k:v.clone() for k,v in tr.items()});original(tr)
        buf.add=remember
        col.collect(11,False)
        # The fourth frame starts a new episode for env 0, but env 1 keeps its history.
        torch.testing.assert_close(captured[3]['obs_hist'][0],captured[3]['obs_hist'][0,-1:].expand(5,-1))
        self.assertFalse(torch.equal(captured[3]['obs_hist'][1,0],captured[3]['obs_hist'][1,-1]))
        for age,tr in enumerate(captured[-8:]):
            row=buf._row(torch.tensor(age))
            torch.testing.assert_close(buf.data['obs_hist'][row],tr['obs_hist'])
        s,h=buf.sample_starts(40,2)
        for i in range(len(s['s'])):
            self.assertTrue(any(torch.equal(s['s'][i],tr['s'][j]) and torch.equal(h['obs_hist'][i,-1],tr['obs_hist'][j])
                                for tr in captured[-8:] for j in range(2)))
        self.assertTrue(torch.isfinite(pi(col.runner.obs_history,torch.zeros(2,206))).all())

    def test_actual_trainer_update_with_history_and_legacy(self):
        for kind,mem in [('moe',True),('mlp',False)]:
            mdp,pi,sim,wm,buf,col=self.make_collector(kind,mem)
            col.collect(12,False)
            cfg=parse_args(['--policy',kind,'--mem',str(int(mem)),'--pi_batch','4','--horizon','3',
                            '--critic_epochs','1','--critic_minibatches','2','--wm_batch','2','--wm_window','3',
                            '--compile','0','--cudagraphs','0'])
            t=Trainer.__new__(Trainer);t.cfg=cfg;t.dev='cpu';t.mdp=mdp;t.pi=pi;t.wm=wm;t.fpm=None
            t.sim=sim;t.buffer=buf;t.critic=Critic(mdp.critic_dim,hidden=(32,32))
            t.opt_pi=torch.optim.Adam(pi.parameters(),lr=2e-4,fused=True);t.opt_v=torch.optim.Adam(t.critic.parameters(),lr=2e-4,fused=True)
            t.opt_wm=torch.optim.Adam(wm.parameters(),lr=2e-4,fused=True);t.prof=NS(tick=lambda *a:None)
            t.mirror_extra=(lambda e:torch.cat((mirror_scan(e[:,:204]),-e[:,204:]),-1)) if mem else mirror_scan
            t.err_floor=torch.tensor(ERR_FLOOR);t._compile()
            if kind=='moe':
                old=make_policy('cpu',True);migrate_mlp_to_moe(pi,old.state_dict())
            before=pi.net[0].weight.detach().clone()
            for _ in range(3):
                stats=t.policy_update(False)
                self.assertTrue(all(torch.isfinite(x).all() for x in stats))
            self.assertFalse(torch.equal(before,pi.net[0].weight))
            self.assertTrue(all(torch.isfinite(x).all() for x in t.wm_update()))
            if kind=='moe':
                for group in (pi.experts,pi.gating_network):
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()),0.)
                self.assertEqual(len([k for k in t.moe_metrics if k.startswith('moe_use_')]),8)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

"""Small Newton collector smoke test, using the same task and transitions as training."""
from pathlib import Path
import argparse,sys,torch,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
a=argparse.ArgumentParser();a.add_argument('--device',default='cuda:0');a.add_argument('--steps',type=int,default=50);args=a.parse_args()
from tgi.env.env_cfg import make_isaac_env_cfg
from tgi.env.sim import IsaacLocoSim
c=make_isaac_env_cfg(2);c.sim.device=args.device
c.scene.terrain.terrain_generator.border_width=0.
c.scene.terrain.terrain_generator.num_rows=1
c.scene.terrain.terrain_generator.num_cols=1
subs=c.scene.terrain.terrain_generator.sub_terrains
c.scene.terrain.terrain_generator.sub_terrains={'random_rough':subs['random_rough']}
c.scene.terrain.terrain_generator.sub_terrains['random_rough'].noise_range=(0.,0.)
c.events.reset_base.params['velocity_range']={k:(0.,0.) for k in c.events.reset_base.params['velocity_range']}
c.events.reset_robot_joints.params['velocity_range']=(0.,0.)
sim=IsaacLocoSim(2,args.device,cfg=c,sgs_spawns=1,sgs_dirs=1,play=True)
from tgi.policy_io import build_mdp,SimRunner,make_policy
from tgi.foot_pred import FootholdPredictor
mdp=build_mdp(sim,args.device,hm_res=.05)
runner=SimRunner(make_policy(args.device),FootholdPredictor().to(args.device),mdp,sim,noise=False)
o,e=runner.observe()
assert o.shape==(2,48) and e.shape==(2,204)
assert torch.isfinite(runner.act()).all()
# Compare the differentiable FK to Newton's measured toe frames at the reset state.
from tgi.robot import FOOT_BODIES,FOOT_OFFSET
from tgi.mdp.rotations import quat_to_mat
ids=[sim.robot.find_bodies(n)[0][0] for n in FOOT_BODIES]
pose=sim.robot.data.body_link_pose_w.torch[:,ids]
rot=quat_to_mat(torch.cat((pose[...,6:7],pose[...,3:6]),-1))
centres=pose[...,:3]+(rot@torch.tensor(FOOT_OFFSET,device=args.device)[...,None]).squeeze(-1)
torch.testing.assert_close(mdp.kin.forward(*sim.state())[0],centres,atol=3e-5,rtol=1e-4)
reset_seen=False
for step in range(args.steps):
    if step==args.steps-1:sim.env.episode_length_buf[:]=sim.env.max_episode_length-1
    tr=sim.step(torch.zeros(2,12,device=args.device))
    assert tr['s1'].shape==(2,37) and tr['c1'].shape==(2,4)
    assert torch.isfinite(tr['s1']).all()
    if tr['done'].any():
        reset_seen=True
        assert sim.ep_len[tr['done']].eq(0).all()
        assert sim._snap is not None
assert reset_seen
print('NEWTON CHECK PASSED: 2 envs, 50 Hz, four-foot FK, 48+204 observations, terminal snapshots; final pre-reset contacts:',tr['c1'].tolist())
report={'device':args.device,'num_envs':2,'steps':args.steps,'state_dim':37,'action_dim':12,'fpm_input_dim':36,'policy_obs_dim':252,'feet':4,'reset_seen':reset_seen,'pre_reset_contacts':tr['c1'].tolist()}
(ROOT/'docs/newton_check.json').write_text(json.dumps(report,indent=2))
sim.env.close()

"""CPU Newton regression: simulator and imagination use the same tilt limit."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tgi/src'),str(ROOT/'fpm/src')]
os.environ.setdefault('MPLCONFIGDIR','/tmp/bpx-tgi-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/bpx-tgi-cache')
import torch,math
from tgi.env.env_cfg import make_isaac_env_cfg
from tgi.env.sim import IsaacLocoSim
from tgi.policy_io import build_mdp,Bundle
from tgi.mdp.mdp import RewardCfg,unflatten_state
c=make_isaac_env_cfg(2);c.sim.device='cpu';g=c.scene.terrain.terrain_generator;g.num_rows=g.num_cols=1;g.border_width=0.
rough=g.sub_terrains['random_rough'];rough.noise_range=(0.,0.);g.sub_terrains={'random_rough':rough}
assert c.terminations.bad_orientation.params['limit_angle']==RewardCfg().term_angle==.7
sim=IsaacLocoSim(2,'cpu',cfg=c,sgs_spawns=1,sgs_dirs=1,play=True,term_angle=.9)
assert sim.cfg.terminations.bad_orientation.params['limit_angle']==.9
mdp=build_mdp(sim,'cpu',reward_cfg=RewardCfg(term_angle=.9),hm_res=.05)
pose=sim.robot.data.root_link_pose_w.torch.clone();pose[:,2]+=2.
angles=torch.tensor([1.,.8]);pose[:,3]=torch.sin(angles/2);pose[:,4:6]=0;pose[:,6]=torch.cos(angles/2)
sim.robot.write_root_link_pose_to_sim_index(root_pose=pose)
sim.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(2,6))
sim.env.scene.write_data_to_sim();sim.env.sim.forward();sim.env.scene.update(0.)
tr=sim.step(torch.zeros(2,12));S=unflatten_state(tr['s1']);imag=mdp.termination(S,mdp.context(S))
print('Simulator termination:',tr['term'].tolist(),'Imagined termination:',imag.tolist())
assert tr['term'].tolist()==[True,False]
assert imag.tolist()==[True,False]
sim.env.close()
print('TERMINATION PARITY PASSED: custom 0.9 rad; 1.0 rad terminates, 0.8 rad remains alive')

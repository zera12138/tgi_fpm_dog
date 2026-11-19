"""State, rewards, terrain features and the base G1 env's observation interface, all differentiable in torch."""

from .heightmap import HeightMap
from .interface import JointMap, FootholdScan
from .kinematics import RobotKinematics
from .mdp import MDP, RewardCfg, normalize_xi, unflatten_state
from .memory import MEM_DIM, duration_scores, gait_phase, gait_targets, init_memory, update_memory
from .terrain import TerrainFeatures

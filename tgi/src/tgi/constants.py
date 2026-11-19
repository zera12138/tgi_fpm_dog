"""BPX dimensions shared by simulation, imagination and validation."""
import os
from .robot import NUM_J, NUM_FEET, MJCF, USD
STATE_DIM = 13 + 2 * NUM_J
XI_DIM = 1 + 3 + NUM_J + 5
MAX_LAG = 4
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ROBOT_MJCF, ROBOT_USD = MJCF, USD
POSE_STD = {
 "std_standing": {".*": 0.10},
 "std_walking": {".*hip_roll.*": 0.20, ".*hip_pitch.*": 0.50, ".*knee.*": 0.60},
 "std_running": {".*hip_roll.*": 0.30, ".*hip_pitch.*": 0.70, ".*knee.*": 0.80},
}
def match_values(patterns, names):
 import re
 out = []
 for n in names:
  matches = [v for pat,v in patterns.items() if re.fullmatch(pat,n)]
  if not matches: raise ValueError(f"No parameter for joint {n}")
  out.append(matches[0])
 return out

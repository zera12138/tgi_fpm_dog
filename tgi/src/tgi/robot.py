"""BPX conventions shared by assets, learning and export. No hardware calibration is performed.

PD/pose follow the existing local BPX simulation configuration; limits follow the supplied URDF.
Feet are always FL, FR, HL, HR; joints are leg-major roll, pitch, knee.
"""
from pathlib import Path
import math
ROOT = Path(__file__).resolve().parents[3]
LEGS = ("fl", "fr", "hl", "hr")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip_roll", "hip_pitch", "knee"))
FOOT_BODIES = tuple(f"{leg}_toe_link" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
NUM_J = len(JOINT_NAMES)
NUM_FEET = len(LEGS)
DEFAULT_Q = (0.0, math.radians(40), math.radians(-65)) * NUM_FEET
DEFAULT_POSE = dict(zip(JOINT_NAMES, DEFAULT_Q))
KP, KD, ACTION_SCALE, ACTION_CLIP = 50.0, 1.0, 0.25, 10.0
EFFORT_LIMIT, VELOCITY_LIMIT = 30.0, 20.0
# Explicit simulation assumptions, not identified motor parameters.
ARMATURE, JOINT_DAMPING, JOINT_FRICTION = 0.01, 0.0, 0.0
NOMINAL_HEIGHT = 0.43
CONTROL_DT, PHYSICS_DT = 0.02, 0.005
FOOT_RADIUS, FOOT_HALF_LENGTH = 0.027, 0.0179
FOOT_OFFSET = (0.0, 0.0, -0.012)  # cylinder centre in each toe frame
HIP_XY = ((0.2386, 0.17122), (0.2386, -0.17122), (-0.2386, 0.17122), (-0.2386, -0.17122))
# Reference-point FK at DEFAULT_Q, including toe-centre offset.
_dx = -0.23 * math.sin(DEFAULT_Q[1]) - (0.2315648 + 0.012) * math.sin(DEFAULT_Q[1] + DEFAULT_Q[2])
BOOTSTRAP_FOOTHOLDS = tuple((x + _dx, y) for x, y in HIP_XY)
URDF = str(ROOT / "urdf/bpx.urdf")
MJCF = str(Path(__file__).parent / "assets/bpx.xml")
USD = str(Path(__file__).parent / "assets/usd/bpx_robot/bpx_robot.usda")
INTERFACE = "bpx-tgi-fpm-v1"
SCAN_SIZE, SCAN_RES, PATCH_N, SCAN_OFFSET = (1.6, 1.0), 0.05, 7, NOMINAL_HEIGHT

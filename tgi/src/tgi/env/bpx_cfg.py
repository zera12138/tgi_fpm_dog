"""BPX asset and manager terms for the upstream Isaac Lab 3/Newton collector.

Reuses the upstream task shell, command sampler, reset terms and terrain generator.
TGI constructs policy observations and reward gradients itself.
"""
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from fpm.tasks.locomotion.config.g1.env_cfg import FpmEnvCfg
from ..robot import (USD, DEFAULT_POSE, NOMINAL_HEIGHT, KP, KD, EFFORT_LIMIT,
                     VELOCITY_LIMIT, ARMATURE, PHYSICS_DT, CONTROL_DT)

def bpx_env_cfg():
    cfg = FpmEnvCfg()
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD, variants={"Physics": "mujoco"}, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,NOMINAL_HEIGHT), joint_pos=DEFAULT_POSE, joint_vel={".*":0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*_(hip_roll|hip_pitch|knee)_joint"],
            stiffness=KP, damping=KD, armature=ARMATURE,
            effort_limit_sim=EFFORT_LIMIT, velocity_limit_sim=VELOCITY_LIMIT)})
    cfg.scene.terrain.visual_material = None
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/torso(/.*)?"
    cfg.sim.dt = PHYSICS_DT
    cfg.decimation = round(CONTROL_DT / PHYSICS_DT)
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/torso"
    cfg.rewards.joint_deviation_waist = None
    cfg.rewards.joint_deviation_arms = None
    cfg.rewards.upright_posture = None  # G1 foot-separation/height reward is inapplicable
    cfg.rewards.joint_deviation_legs.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="(?!.*_toe_link$).*")
    cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso")
    cfg.events.gravity_assist.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso")
    cfg.events.constant_disturbance.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso")
    for term in (cfg.events.actuator_pd_gains, cfg.events.joint_friction):
        term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_(hip_roll|hip_pitch|knee)_joint"])
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.height_scanner.debug_vis = False
    cfg.terminations.bad_orientation.params["limit_angle"] = 0.7
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.8)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)
    return cfg

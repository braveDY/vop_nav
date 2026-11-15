from isaaclab.actuators import DCMotorCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as ISAACLAB_GO2_CFG

UNITREE_GO2_CFG = ISAACLAB_GO2_CFG.replace(soft_joint_pos_limit_factor=0.9)
UNITREE_GO2_CFG.spawn.articulation_props.enabled_self_collisions = True

UNITREE_GO2_CFG.actuators = {
    "hip": DCMotorCfg(
        joint_names_expr=[".*_hip_joint"],
        effort_limit=35.0,
        saturation_effort=35.0,
        velocity_limit=52.4,
        stiffness=40.0,
        damping=1.0,
        friction=0.0,
    ),
    "thigh": DCMotorCfg(
        joint_names_expr=[".*_thigh_joint"],
        effort_limit=40.0,
        saturation_effort=45.0,
        velocity_limit=30.1,
        stiffness=40.0,
        damping=1.0,
        friction=0.0,
    ),
    "calf": DCMotorCfg(
        joint_names_expr=[".*_calf_joint"],
        effort_limit=40.0,
        saturation_effort=45.0,
        velocity_limit=30.1,
        stiffness=40.0,
        damping=1.0,
        friction=0.0,
    ),
}

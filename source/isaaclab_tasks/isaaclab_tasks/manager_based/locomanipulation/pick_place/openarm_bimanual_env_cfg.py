# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base bimanual OpenArm pick-place environment for Quest 3 teleop recording.

Direct port of ``fixed_base_upper_body_ik_g1_env_cfg.py``: PinkIK with one
``LocalFrameTask`` per wrist for absolute-pose control, ``Se3AbsRetargeter``
driving each wrist target from a Quest controller, and a static XR anchor.
The G1's 14-joint ``TriHandMotionControllerRetargeter`` is replaced by two
``GripperRetargeter``s (1-DOF binary pincher per side), wired to separate
``BinaryJointPositionAction`` terms so the Quest trigger logic stays clean.
"""

from isaaclab_teleop import IsaacTeleopCfg, XrCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

import os as _os

# In-repo OpenArm bimanual assets (USDs + URDFs + meshes). All paths resolve
# relative to this file so the repo can be cloned anywhere.
# The ``_pink_ik`` USD has the wheel joints and the 4 unactuated hand/TCP
# revolute joints converted to fixed, so Pink IK's strict USD<->URDF
# joint-list correspondence holds.
_OPENARM_ASSETS_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "assets", "openarm_bimanual"
)
OPENARM_BIMANUAL_LOCAL_USD = _os.path.join(
    _OPENARM_ASSETS_DIR, "usds", "openarm_bimanual_pink_ik.usd"
)
OPENARM_BIMANUAL_LOCAL_URDF = _os.path.join(
    _OPENARM_ASSETS_DIR, "urdfs", "openarm_bimanual.urdf"
)
# Parent of ``openarm_description/`` so pinocchio resolves
# ``package://openarm_description/...`` mesh references in the URDF.
OPENARM_BIMANUAL_MESH_ROOT = _OPENARM_ASSETS_DIR


# Pink IK controller: one LocalFrameTask per wrist plus a null-space posture
# regulariser. Cost/gain/lm_damping mirror G1's fixed-base upper-body cfg.
OPENARM_BIMANUAL_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="openarm_body_link",  # USD body name
    num_hand_joints=0,  # grippers handled by separate BinaryJointPositionAction
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTaskCfg(
            frame="openarm_left_hand",  # URDF link name
            base_link_frame_name="openarm_body_link0",  # URDF root of the bimanual chain
            position_cost=8.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad] — start low while tuning RPY offsets
            lm_damping=1.0,
            gain=0.5,
        ),
        LocalFrameTaskCfg(
            frame="openarm_right_hand",
            base_link_frame_name="openarm_body_link0",
            position_cost=8.0,
            orientation_cost=1.0,
            lm_damping=1.0,
            gain=0.5,
        ),
        NullSpacePostureTaskCfg(
            cost=0.05,
            lm_damping=75,
            controlled_frames=["openarm_left_hand", "openarm_right_hand"],
            controlled_joints=[
                "openarm_left_joint1",
                # "openarm_left_joint2",
                # "openarm_left_joint3",
                "openarm_right_joint1",
                # "openarm_right_joint2",
                # "openarm_right_joint3",
            ],
            gain=0.075,
        ),
    ],
    fixed_input_tasks=[],
)


# Action cfg for the upper body IK. Action vector is
# ``pose_dim(7) * 2 = 14`` (left wrist + right wrist), xyzw quat convention.
# Grippers are wired as separate action terms below.
OPENARM_BIMANUAL_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        "openarm_left_joint[1-7]",
        "openarm_right_joint[1-7]",
    ],
    hand_joint_names=[],
    target_eef_link_names={
        "left_wrist": "openarm_left_hand",  # USD body name
        "right_wrist": "openarm_right_hand",
    },
    asset_name="robot",
    controller=OPENARM_BIMANUAL_IK_CONTROLLER_CFG,
)


def _build_openarm_bimanual_pipeline():
    """Build an IsaacTeleop retargeting pipeline for bimanual OpenArm teleop.

    Two ``Se3AbsRetargeter``s emit absolute wrist pose targets (xyzw quat)
    consumed by Pink IK, and two ``GripperRetargeter``s convert each Quest
    trigger into a ``-1/+1`` binary gripper command. ``TensorReorderer``
    flattens everything into the 16-D action tensor matching ``ActionsCfg``.

    Returns:
        Tuple of (OutputCombiner, list[BaseRetargeter]): the pipeline plus
        the Se3 retargeters exposed for a future tuning UI.
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # Absolute wrist pose targets. Defaults applied (target_offset_roll=90)
    # so the Quest "forward" axis lines up with a reasonable EE forward;
    # tune target_offset_{roll,pitch,yaw} on the configs below if grasps
    # land rotated for your OpenArm USD orientation.
    # OpenArm wrist axes differ per side (left link0 mounted at rpy=(-90°,0,0),
    # right at rpy=(+90°,0,0) in the URDF), so each side needs its own
    # Quest-grip-to-EE offset. Starting from G1's values as a known-good seed —
    # expect to tune from here in the tuning UI.
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=180.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=180.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # Quest trigger -> binary gripper command. +1.0 open / -1.0 closed,
    # matching BinaryJointPositionAction's sign convention.
    left_grip = GripperRetargeter(
        GripperRetargeterConfig(hand_side="left", controller_threshold=0.5),
        name="left_gripper",
    )
    connected_left_grip = left_grip.connect({"controller_left": controllers.output(ControllersSource.LEFT)})

    right_grip = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right", controller_threshold=0.5),
        name="right_gripper",
    )
    connected_right_grip = right_grip.connect({"controller_right": controllers.output(ControllersSource.RIGHT)})

    # Se3AbsRetargeter emits ``[px, py, pz, qx, qy, qz, qw]`` (xyzw).
    # Pink IK action consumes pose_dim(7) per frame in the same xyzw layout
    # (matches Isaac Lab 3.0's xyzw convention).
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_grip_elements = ["l_grip"]
    right_grip_elements = ["r_grip"]

    output_order = left_ee_elements + right_ee_elements + left_grip_elements + right_grip_elements

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_gripper": left_grip_elements,
            "right_gripper": right_grip_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_gripper": "scalar",
            "right_gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_gripper": connected_left_grip.output("gripper_command"),
            "right_gripper": connected_right_grip.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_se3, right_se3]


##
# Scene definition
##
@configclass
class OpenArmBimanualSceneCfg(InteractiveSceneCfg):
    """Scene for the bimanual OpenArm pick-place task.

    Mirrors the G1 fixed-base scene (packing table + steering wheel) with a
    stationary bimanual OpenArm rig in place of the humanoid.
    """

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.35, 0.6996], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    robot: ArticulationCfg = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        # Use the local sanitized OpenArm bimanual USD.
        self.robot.spawn.usd_path = OPENARM_BIMANUAL_LOCAL_USD
        # The bimanual asset sits on a wheeled base; pin it down for
        # stationary teleop.
        self.robot.spawn.articulation_props.fix_root_link = True
        self.robot.init_state.pos = (0.0, 0.08, 0.63)
        # Rotate the robot 90° about world +Z. xyzw quat: (0, 0, sin(45°), cos(45°)).
        self.robot.init_state.rot = (0.0, 0.0, 0.70710678, 0.70710678)

        # Teleop-only override: G1-class actuator headroom so the arm tracks
        # Quest motion as snappily as the G1 pick-place env. These limits are
        # far above the real Damiao motor spec sheets — do NOT use them when
        # collecting training data targeting real hardware. Revert to
        # OPENARM_BI_CFG's spec-sheet values for sim-to-real.
        self.robot.actuators["openarm_arm"].velocity_limit_sim = {
            "openarm_left_joint[1-2]": 10.0,
            "openarm_right_joint[1-2]": 10.0,
            "openarm_left_joint[3-4]": 10.0,
            "openarm_right_joint[3-4]": 10.0,
            "openarm_left_joint[5-7]": 12.0,
            "openarm_right_joint[5-7]": 12.0,
        }
        self.robot.actuators["openarm_arm"].effort_limit_sim = {
            "openarm_left_joint[1-2]": 250.0,
            "openarm_right_joint[1-2]": 250.0,
            "openarm_left_joint[3-4]": 200.0,
            "openarm_right_joint[3-4]": 200.0,
            "openarm_left_joint[5-7]": 50.0,
            "openarm_right_joint[5-7]": 50.0,
        }
        # Stiffer PD so the joints actually chase the IK targets at these speeds.
        self.robot.actuators["openarm_arm"].stiffness = 1000.0
        self.robot.actuators["openarm_arm"].damping = 100.0


@configclass
class ActionsCfg:
    """Action specifications.

    Declaration order = concat order in the action tensor:
    ``[upper_body_ik(14), left_gripper(1), right_gripper(1)]`` = 16-D total.
    """

    upper_body_ik = OPENARM_BIMANUAL_IK_ACTION_CFG

    left_gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_finger_joint.*"],
        open_command_expr={"openarm_left_finger_joint.*": 0.044},
        close_command_expr={"openarm_left_finger_joint.*": 0.0},
    )

    right_gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_right_finger_joint.*"],
        open_command_expr={"openarm_right_finger_joint.*": 0.044},
        close_command_expr={"openarm_right_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specs. Required by ManagerBasedRLEnvCfg but unused for teleop."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "openarm_left_hand"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "openarm_left_hand"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "openarm_right_hand"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "openarm_right_hand"})

        gripper_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={"joint_names": ["openarm_left_finger_joint.*", "openarm_right_finger_joint.*"]},
        )

        object = ObsTerm(
            func=manip_mdp.object_obs,
            params={"left_eef_link_name": "openarm_left_hand", "right_eef_link_name": "openarm_right_hand"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=manip_mdp.task_done_pick_place,
        params={"task_link_name": "openarm_right_hand"},
    )


@configclass
class OpenArmBimanualEnvCfg(ManagerBasedRLEnvCfg):
    """Bimanual OpenArm pick-place env for Quest 3 teleop recording (G1-style)."""

    scene: OpenArmBimanualSceneCfg = OpenArmBimanualSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    curriculum = None

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2

        # Local URDF + mesh root for the Pink IK kinematics.
        self.actions.upper_body_ik.controller.urdf_path = OPENARM_BIMANUAL_LOCAL_URDF
        self.actions.upper_body_ik.controller.mesh_path = OPENARM_BIMANUAL_MESH_ROOT

        # Static XR anchor — matches G1 fixed-base.
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, -0.30),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_openarm_bimanual_pipeline()[0],
            # retargeters_to_tune=lambda: _build_openarm_bimanual_pipeline()[1],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm sandbox env for Quest 3 teleop.

Strips the pick-place scene down to a ground plane. The bimanual robot has
active wheel joints so the Quest joysticks can drive the base around while
the controllers still teleop the arms.

USD and URDF in this env include 4 wheel joints (BLW/BRW/FRW/FLW). The
sibling pick-place env continues to use the wheel-fixed variants.
"""

import numpy as np
from dataclasses import dataclass

from isaaclab_teleop import IsaacTeleopCfg, XrAnchorRotationMode, XrCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
)
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG


import os as _os

# In-repo wheels-active assets (only TCP joints converted to fixed) and
# matching URDF (with wheel + caster_stem + lift joints/links appended).
# Paths resolve relative to this file so the repo is portable.
_OPENARM_ASSETS_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "assets", "openarm_bimanual"
)
OPENARM_BIMANUAL_WHEELS_USD = _os.path.join(
    _OPENARM_ASSETS_DIR, "usds", "openarm_bimanual_pink_ik_wheels.usd"
)
OPENARM_BIMANUAL_WHEELS_URDF = _os.path.join(
    _OPENARM_ASSETS_DIR, "urdfs", "openarm_bimanual_with_wheels.urdf"
)
OPENARM_BIMANUAL_MESH_ROOT = _OPENARM_ASSETS_DIR


# Wheel geometry (must match the URDF / USD layout).
WHEEL_RADIUS_M = 0.05
WHEEL_TRACK_HALF_M = 0.29  # half the lateral distance between left and right wheel rows


# Pink IK controller — same task setup as the pick-place env. The wheel joints
# exist in the articulation but are not in ``pink_controlled_joint_names``, so
# Pink IK leaves them alone.
OPENARM_BIMANUAL_SANDBOX_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="openarm_body_link",
    num_hand_joints=0,
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTaskCfg(
            frame="openarm_left_hand",
            base_link_frame_name="openarm_body_link0",
            position_cost=8.0,
            orientation_cost=1.0,
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
            controlled_joints=["openarm_left_joint1", "openarm_right_joint1"],
            gain=0.075,
        ),
    ],
    fixed_input_tasks=[],
)


OPENARM_BIMANUAL_SANDBOX_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        "openarm_left_joint[1-7]",
        "openarm_right_joint[1-7]",
    ],
    hand_joint_names=[],
    target_eef_link_names={
        "left_wrist": "openarm_left_hand",
        "right_wrist": "openarm_right_hand",
    },
    asset_name="robot",
    controller=OPENARM_BIMANUAL_SANDBOX_IK_CONTROLLER_CFG,
)


def _build_openarm_bimanual_sandbox_pipeline():
    """Build the IsaacTeleop retargeting pipeline for the sandbox env.

    Same Se3 + Gripper retargeters as the pick-place env, plus a custom
    DiffDrive retargeter that maps Quest joysticks to wheel velocities.

    Action layout (21-D):
        [upper_body_ik(14), left_gripper(1), right_gripper(1), wheel_vels(4), lift(1)]
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
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, ParameterState
    from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
    from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType, OptionalType
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        NDArrayType,
        DLDataType,
        ControllerInputIndex,
    )

    @dataclass
    class DiffDriveRetargeterConfig:
        wheel_radius: float = WHEEL_RADIUS_M
        track_half: float = WHEEL_TRACK_HALF_M
        max_lin_speed: float = 1  # m/s
        max_ang_speed: float = 5  # rad/s
        deadzone: float = 0.1

    class DiffDriveRetargeter(BaseRetargeter):
        """Maps Quest joysticks to 4 wheel angular velocities (diff-drive).

        - Left stick Y -> forward/back linear velocity
        - Right stick X -> yaw angular velocity
        Output order matches the JointVelocityActionCfg joint regex order:
        ``[BLW, BRW, FRW, FLW]`` resolved alphabetically by Isaac Lab.
        """

        def __init__(self, config: DiffDriveRetargeterConfig, name: str):
            super().__init__(name=name)
            self._config = config

        def input_spec(self):
            return {
                "controller_left": OptionalType(ControllerInput()),
                "controller_right": OptionalType(ControllerInput()),
            }

        def output_spec(self):
            return {
                "wheel_vels": TensorGroupType(
                    "wheel_vels",
                    [NDArrayType("vels", shape=(4,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
            cfg = self._config

            ls_y = 0.0
            rs_x = 0.0
            left = inputs["controller_left"]
            if not left.is_none:
                ls_y = float(left[ControllerInputIndex.THUMBSTICK_Y])
            right = inputs["controller_right"]
            if not right.is_none:
                rs_x = float(right[ControllerInputIndex.THUMBSTICK_X])

            # Deadzone + scale
            ls_y = 0.0 if abs(ls_y) < cfg.deadzone else ls_y
            rs_x = 0.0 if abs(rs_x) < cfg.deadzone else rs_x
            v_fwd = ls_y * cfg.max_lin_speed
            w_yaw = -rs_x * cfg.max_ang_speed  # right on stick = +x = clockwise (negative omega_z)

            # Diff-drive: w_wheel = (v_fwd ± omega * track_half) / radius
            v_left = (v_fwd - w_yaw * cfg.track_half) / cfg.wheel_radius
            v_right = (v_fwd + w_yaw * cfg.track_half) / cfg.wheel_radius

            # Wheel order: BLW, BRW, FLW, FRW (alphabetical regex match order in Isaac Lab).
            wheel_vels = np.array([v_left, v_right, v_left, v_right], dtype=np.float32)
            outputs["wheel_vels"][0] = wheel_vels

    @dataclass
    class LiftRetargeterConfig:
        speed: float = 0.25  # m/s of target movement when squeeze is fully held
        deadzone: float = 0.05
        dt: float = 1.0 / 60.0  # retargeter tick rate (must roughly match teleop loop)
        # Range inverted: initial position is the TOP (0), arms can only travel
        # downward from there (down to -0.30). Right squeeze raises back up
        # toward 0 (the upper limit), left squeeze lowers further toward -0.30.
        lower_limit: float = -0.6   # match USD/URDF lower limit (lowest = -0.6)
        upper_limit: float = 0.0    # initial pos / highest the arms can go
        initial_pos: float = 0.0

    class LiftRetargeter(BaseRetargeter):
        """Integrates Quest grip squeeze triggers into a lift-joint target position.

        - Right SQUEEZE_VALUE -> target position increases (raise)
        - Left SQUEEZE_VALUE  -> target position decreases (lower)
        Target is clamped to [lower_limit, upper_limit] each tick. Output is the
        absolute target position; the lift actuator is PD-controlled so it
        holds against gravity when no squeeze is applied.
        """

        def __init__(self, config: LiftRetargeterConfig, name: str):
            super().__init__(name=name)
            self._config = config
            self._target_pos = config.initial_pos

        def input_spec(self):
            return {
                "controller_left": OptionalType(ControllerInput()),
                "controller_right": OptionalType(ControllerInput()),
            }

        def output_spec(self):
            return {
                "lift_pos": TensorGroupType(
                    "lift_pos",
                    [NDArrayType("p", shape=(1,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
            cfg = self._config
            sq_l = 0.0
            sq_r = 0.0
            left = inputs["controller_left"]
            if not left.is_none:
                sq_l = float(left[ControllerInputIndex.SQUEEZE_VALUE])
            right = inputs["controller_right"]
            if not right.is_none:
                sq_r = float(right[ControllerInputIndex.SQUEEZE_VALUE])
            sq_l = 0.0 if sq_l < cfg.deadzone else sq_l
            sq_r = 0.0 if sq_r < cfg.deadzone else sq_r
            delta = (sq_r - sq_l) * cfg.speed * cfg.dt
            self._target_pos = max(cfg.lower_limit, min(cfg.upper_limit, self._target_pos + delta))
            outputs["lift_pos"][0] = np.array([self._target_pos], dtype=np.float32)

    # ---- Wire the pipeline ----
    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # Same offsets we tuned in the pick-place env.
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=180.0,
        target_offset_pitch=0.0,
        target_offset_yaw=90.0,
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
        target_offset_yaw=90.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

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

    diff_drive = DiffDriveRetargeter(DiffDriveRetargeterConfig(), name="diff_drive")
    connected_diff_drive = diff_drive.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    lift = LiftRetargeter(LiftRetargeterConfig(), name="lift")
    connected_lift = lift.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_grip_elements = ["l_grip"]
    right_grip_elements = ["r_grip"]
    wheel_elements = ["w0", "w1", "w2", "w3"]
    lift_elements = ["lift_v"]

    output_order = (
        left_ee_elements + right_ee_elements + left_grip_elements + right_grip_elements + wheel_elements + lift_elements
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_gripper": left_grip_elements,
            "right_gripper": right_grip_elements,
            "diff_drive": wheel_elements,
            "lift": lift_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_gripper": "scalar",
            "right_gripper": "scalar",
            "diff_drive": "array",
            "lift": "array",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_gripper": connected_left_grip.output("gripper_command"),
            "right_gripper": connected_right_grip.output("gripper_command"),
            "diff_drive": connected_diff_drive.output("wheel_vels"),
            "lift": connected_lift.output("lift_pos"),
        }
    )

    # Store the lift retargeter on a module-level handle so the env's reset
    # event can clear its accumulated target position.
    _LIFT_RETARGETER_HANDLE["instance"] = lift

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_se3, right_se3]


# Holds a reference to the LiftRetargeter once the pipeline is built. Used by
# the reset event below to clear the retargeter's internal target_pos on env
# reset (the retargeter is built once at startup and persists across resets).
_LIFT_RETARGETER_HANDLE: dict = {"instance": None}


def _reset_lift_retargeter(env, env_ids):
    """Reset the LiftRetargeter's accumulated target position on env reset."""
    inst = _LIFT_RETARGETER_HANDLE.get("instance")
    if inst is not None:
        inst._target_pos = inst._config.initial_pos


# In-repo warehouse + package USDs.
_OPENARM_WAREHOUSE_USD = _os.path.join(_OPENARM_ASSETS_DIR, "usds", "warehouse.usd")
_SMALL_PACKAGE_USD = _os.path.join(_OPENARM_ASSETS_DIR, "packages", "SmallPackage", "smallPackage.usdc")
_LARGE_PACKAGE_USD = _os.path.join(_OPENARM_ASSETS_DIR, "packages", "LargePackage", "largePackage.usdc")


def _spawn_package(prim_path: str, usd_path: str, pos: tuple, mass: float = 0.1) -> RigidObjectCfg:
    """Helper to spawn a graspable package at its native USD scale."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class OpenArmBimanualSandboxSceneCfg(InteractiveSceneCfg):
    """Sandbox: warehouse + packing table + graspable packages + wheeled bimanual OpenArm with a lift joint."""

    robot: ArticulationCfg = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Local warehouse USD provides floor / walls / lighting backdrop.
    # rot is xyzw — (0,0,0,1) is identity; (1,0,0,0) would be 180° about X (upside down).
    # kinematic_enabled=True disables physics simulation on every rigid body
    # in the warehouse (walls, racks, etc.) so they're a static backdrop.
    # Collisions stay on so the robot can't drive through walls.
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=UsdFileCfg(
            usd_path=_OPENARM_WAREHOUSE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Packing table in front of the robot (robot faces world +Y after the
    # 90° yaw, so positive Y is forward). rot xyzw identity = upright table.
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.4, -0.3), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Three packages on the packing table (top ≈ 0.74 m, packages settle a bit
    # above to avoid initial interpenetration).
    # Spread packages along the table top (table x is roughly -0.4..0.4, y at 1.4).
    package_a = _spawn_package("{ENV_REGEX_NS}/PackageA", _SMALL_PACKAGE_USD, pos=(-0.35, 1.4, 0.80))
    package_b = _spawn_package("{ENV_REGEX_NS}/PackageB", _SMALL_PACKAGE_USD, pos=(0.00, 1.4, 0.80))
    package_c = _spawn_package("{ENV_REGEX_NS}/PackageC", _LARGE_PACKAGE_USD, pos=(0.35, 1.4, 0.82), mass=0.15)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=2000.0),
    )

    def __post_init__(self):
        # Wheels-active sanitised USD.
        self.robot.spawn.usd_path = OPENARM_BIMANUAL_WHEELS_USD
        # Root link is NOT fixed: the wheels need to support the base on the ground.
        self.robot.spawn.articulation_props.fix_root_link = False
        # Re-enable gravity (HIGH_PD cfg disables it; for driving we need gravity to keep wheels on the ground).
        self.robot.spawn.rigid_props.disable_gravity = False
        # Sit the body so the wheels (at -0.31 from body) are roughly on the ground (wheel radius 0.05).
        self.robot.init_state.pos = (0.0, 0.0, 0.36)
        # Explicit zero defaults for every controlled joint so reset is
        # deterministic and the IK doesn't get a "natural" elbow bend from
        # USD joint defaults.
        self.robot.init_state.joint_pos = {
            **{f"openarm_left_joint{i}": 0.0 for i in range(1, 8)},
            **{f"openarm_right_joint{i}": 0.0 for i in range(1, 8)},
            "openarm_left_finger_joint1": 0.0,
            "openarm_left_finger_joint2": 0.0,
            "openarm_right_finger_joint1": 0.0,
            "openarm_right_finger_joint2": 0.0,
            "lift_joint": 0.0,
        }
        # Robot yawed 90° about +Z so its forward = world +Y (same convention as pick-place env).
        self.robot.init_state.rot = (0.0, 0.0, 0.70710678, 0.70710678)

        # Same G1-class arm actuator overrides as the pick-place env.
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
        self.robot.actuators["openarm_arm"].stiffness = 1000.0
        self.robot.actuators["openarm_arm"].damping = 100.0

        # Pseudo-caster setup: rear wheels do all the driving, front wheels
        # spin freely. This dramatically reduces skid during turns (no front
        # wheels fighting the steering). If turning still feels bad, escalate
        # to true casters (adds vertical pivot joints to URDF/USD).
        self.robot.actuators["driven_wheels"] = ImplicitActuatorCfg(
            joint_names_expr=["BLW", "BRW"],
            velocity_limit_sim=60.0,  # ~3 m/s at r=0.05
            effort_limit_sim=150.0,
            stiffness=0.0,
            damping=400.0,
        )
        # Front wheels: zero stiffness + zero damping = free roll. Supports
        # the chassis but applies no longitudinal force to fight the rears.
        self.robot.actuators["free_wheels"] = ImplicitActuatorCfg(
            joint_names_expr=["FLW", "FRW"],
            velocity_limit_sim=200.0,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        )
        # True-caster vertical pivots on the front wheels. Passive — bearing
        # damping only (prevents flutter), no driving torque. The pivot swivels
        # so the front wheel can align with the direction of motion during
        # turns instead of skidding.
        self.robot.actuators["caster_pivots"] = ImplicitActuatorCfg(
            joint_names_expr=["FLW_pivot", "FRW_pivot"],
            velocity_limit_sim=100.0,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.5,
        )
        # Lift joint — position-controlled (PD). The retargeter integrates the
        # squeeze triggers into a target position; this actuator holds against
        # gravity. Lower limit is 0 so the platform rests on the limit when
        # idle (gravity provides the rest force, PD only does work when raising).
        self.robot.actuators["lift"] = ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.5,
            effort_limit_sim=2000.0,
            stiffness=25000.0,
            damping=1200.0,
        )


@configclass
class ActionsCfg:
    """Action specs. Concat order = declaration order.

    Layout: [upper_body_ik(14), left_gripper(1), right_gripper(1), wheel_vels(4)] = 20-D
    """

    upper_body_ik = OPENARM_BIMANUAL_SANDBOX_IK_ACTION_CFG

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

    wheel_vels = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["BLW", "BRW", "FLW", "FRW"],
        scale=1.0,
    )

    lift = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["lift_joint"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Minimal observations — required by ManagerBasedRLEnvCfg but unused for teleop."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    # Clears the LiftRetargeter's accumulated target position on env reset so
    # the lift joint actually returns to its initial pose instead of jumping
    # back to wherever the user left it.
    reset_lift_retargeter = EventTerm(func=_reset_lift_retargeter, mode="reset")


@configclass
class OpenArmBimanualSandboxEnvCfg(ManagerBasedRLEnvCfg):
    """Sandbox env: drive the wheeled bimanual OpenArm with Quest joysticks while teleop-ing the arms."""

    scene: OpenArmBimanualSandboxSceneCfg = OpenArmBimanualSandboxSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 600.0  # long sandbox session
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        # Wheels-aware URDF for Pink IK (joint-list correspondence with the wheels-active USD).
        self.actions.upper_body_ik.controller.urdf_path = OPENARM_BIMANUAL_WHEELS_URDF
        self.actions.upper_body_ik.controller.mesh_path = OPENARM_BIMANUAL_MESH_ROOT

        # XR anchor follows the robot base, like the G1 locomanip env. The
        # anchor sticks to the body link, so as the robot drives, the user's
        # view and EE world targets translate with it — the arms continue to
        # track Quest-relative motion while the chassis moves.
        # anchor_rot identity. The 180° rotation we tried earlier also flips
        # the Quest controllers' coordinate frame, which broke the IK targets.
        # To face the robot's front at spawn, use Quest's recenter feature
        # (long-press the Meta button) while physically facing the robot.
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, -0.55),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        # Anchor on the lift_platform (not the chassis) so the camera position
        # follows arm height as the lift joint moves up/down. Rotation stays
        # FIXED at anchor_rot identity to match pick-place behavior — Quest
        # recenters the user's physical forward to the robot's forward at
        # session start. (FOLLOW_PRIM_SMOOTHED would inherit the robot's 90°
        # yaw onto the anchor, fighting Quest's recenter.)
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/lift_platform"
        self.xr.fixed_anchor_height = False
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FIXED

        # Wire the retargeting pipeline. xr_cfg must be passed in for the
        # anchor-follows-base settings above to take effect.
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_openarm_bimanual_sandbox_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

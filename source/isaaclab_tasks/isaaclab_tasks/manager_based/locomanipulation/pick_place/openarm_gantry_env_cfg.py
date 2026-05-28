# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual OpenArm gantry sandbox env for Quest 3 teleop.

Uses ``trial_arm_gantry.usd`` — the OpenArm chest+arm assembly mounted on a
4-DOF cartesian carriage (glide X, glide Y, yaw, lift Z). The Quest joysticks
glide and turn the whole robot; the grip triggers raise/lower the chest+arms
via the lift joint; and the Quest controllers teleop the wrists via Pink IK.
"""

import os as _os

import numpy as np
import torch
from dataclasses import dataclass

from isaaclab_teleop import IsaacTeleopCfg, XrAnchorRotationMode, XrCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
)
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import (
    mdp as locomanip_mdp,
)

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

_OPENARM_ASSETS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "openarm_bimanual")
TRIAL_ARM_GANTRY_USD = _os.path.join(_OPENARM_ASSETS_DIR, "usds", "arm_gantry_rigged.usd")
# Hand-authored URDF matching the gantry USD's 22 movable joints (4 carriage +
# 14 arm + 4 finger), for Pink IK's Pinocchio model. Auto-conversion from the
# USD fails on this CAD-derived model, so an explicit URDF is required.
GANTRY_URDF = _os.path.join(_OPENARM_ASSETS_DIR, "urdfs", "arm_gantry.urdf")
GANTRY_MESH_ROOT = _OPENARM_ASSETS_DIR

# Lift joint travel (matches USD/URDF limits): +0.6 = top (initial), -0.6 = lowest.
LIFT_LOWER = -0.3
LIFT_UPPER = 1


# Pink IK controller for the arms. Uses the explicit hand-authored gantry URDF
# (set in __post_init__) whose 22 movable joints match the articulation.
# base_link_frame_name is the URDF arm-mount frame (openarm_body_link0).
OPENARM_GANTRY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
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

OPENARM_GANTRY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
    hand_joint_names=[],
    target_eef_link_names={"left_wrist": "openarm_left_hand", "right_wrist": "openarm_right_hand"},
    asset_name="robot",
    controller=OPENARM_GANTRY_IK_CONTROLLER_CFG,
)


class BodyFrameGantryAction(ActionTerm):
    """Body-relative gantry driving.

    Raw action is ``[v_fwd, v_strafe, w_yaw]`` interpreted in the robot's
    current heading frame. Forward/strafe are rotated by the live ``base_yaw``
    joint angle into the world-axis ``glide_x`` / ``glide_y`` joints, so
    "forward" always means the direction the robot is facing. ``w_yaw`` drives
    ``base_yaw`` directly (it rotates in place since yaw sits after the glides
    in the carriage chain).
    """

    cfg: "BodyFrameGantryActionCfg"

    def __init__(self, cfg: "BodyFrameGantryActionCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._gx_ids, _ = self._asset.find_joints("glide_x")
        self._gy_ids, _ = self._asset.find_joints("glide_y")
        self._yaw_ids, _ = self._asset.find_joints("base_yaw")
        self._raw = torch.zeros((self.num_envs, 3), device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions

    def apply_actions(self) -> None:
        jp = self._asset.data.joint_pos
        if not isinstance(jp, torch.Tensor):
            # Newton/warp backend returns wp.array; convert to a torch view.
            import warp as wp

            jp = wp.to_torch(jp)
        yaw = jp[:, self._yaw_ids[0]]
        c, s = torch.cos(yaw), torch.sin(yaw)
        v_fwd, v_str, w = self._raw[:, 0], self._raw[:, 1], self._raw[:, 2]
        vx = v_fwd * c - v_str * s
        vy = v_fwd * s + v_str * c
        self._asset.set_joint_velocity_target(vx.unsqueeze(-1), joint_ids=self._gx_ids)
        self._asset.set_joint_velocity_target(vy.unsqueeze(-1), joint_ids=self._gy_ids)
        self._asset.set_joint_velocity_target(w.unsqueeze(-1), joint_ids=self._yaw_ids)


@configclass
class BodyFrameGantryActionCfg(ActionTermCfg):
    """Cfg for :class:`BodyFrameGantryAction` (asset_name inherited from base)."""

    class_type: type = BodyFrameGantryAction


def _build_openarm_gantry_pipeline():
    """Retargeting pipeline: joysticks -> gantry glide/yaw, grip triggers -> lift.

    Action layout (4-D): [glide_x_vel, glide_y_vel, yaw_vel, lift_pos]
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.interface.retargeter_core_types import (
        RetargeterIO,
    )
    from isaacteleop.retargeting_engine.interface.tensor_group_type import (
        OptionalType,
        TensorGroupType,
    )
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        DLDataType,
        NDArrayType,
        TransformMatrix,
    )

    @dataclass
    class GantryDriveConfig:
        max_lin_speed: float = 1.0  # m/s for glide_x / glide_y
        max_yaw_speed: float = 1.5  # rad/s for base_yaw
        deadzone: float = 0.25  # generous so small/unintended stick drift is ignored

    class GantryDriveRetargeter(BaseRetargeter):
        """Left stick -> glide X/Y velocity, right stick X -> yaw velocity."""

        def __init__(self, config: GantryDriveConfig, name: str):
            super().__init__(name=name)
            self._config = config

        def input_spec(self):
            return {
                "controller_left": OptionalType(ControllerInput()),
                "controller_right": OptionalType(ControllerInput()),
            }

        def output_spec(self):
            return {
                "gantry_vels": TensorGroupType(
                    "gantry_vels",
                    [NDArrayType("v", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
            cfg = self._config
            lx = ly = rx = 0.0
            left = inputs["controller_left"]
            if not left.is_none:
                lx = float(left[ControllerInputIndex.THUMBSTICK_X])
                ly = float(left[ControllerInputIndex.THUMBSTICK_Y])
            right = inputs["controller_right"]
            if not right.is_none:
                rx = float(right[ControllerInputIndex.THUMBSTICK_X])
            lx = 0.0 if abs(lx) < cfg.deadzone else lx
            ly = 0.0 if abs(ly) < cfg.deadzone else ly
            rx = 0.0 if abs(rx) < cfg.deadzone else rx
            # Left stick Y -> +X glide (forward), left stick X -> +Y glide (strafe).
            vx = ly * cfg.max_lin_speed
            vy = -lx * cfg.max_lin_speed
            vyaw = -rx * cfg.max_yaw_speed
            outputs["gantry_vels"][0] = np.array([vx, vy, vyaw], dtype=np.float32)

    @dataclass
    class LiftConfig:
        speed: float = 0.25
        deadzone: float = 0.05
        dt: float = 1.0 / 60.0
        # Output is a DELTA from the top (default joint pos = LIFT_UPPER).
        # 0 = top; the joint action uses use_default_offset=True so a zero
        # output (pre-teleop OR idle) keeps the lift at the top instead of
        # dropping. Delta ranges from (LIFT_LOWER - LIFT_UPPER) up to 0.
        min_delta: float = LIFT_LOWER - LIFT_UPPER
        max_delta: float = 0.0

    class LiftRetargeter(BaseRetargeter):
        """Right grip squeeze raises (toward top), left grip squeeze lowers.

        Outputs a delta from the top. Combined with use_default_offset=True on
        the lift action, idle/no-input = stay at the top.
        """

        def __init__(self, config: LiftConfig, name: str):
            super().__init__(name=name)
            self._config = config
            self._delta = 0.0  # 0 = top

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
            sq_l = sq_r = 0.0
            left = inputs["controller_left"]
            if not left.is_none:
                sq_l = float(left[ControllerInputIndex.SQUEEZE_VALUE])
            right = inputs["controller_right"]
            if not right.is_none:
                sq_r = float(right[ControllerInputIndex.SQUEEZE_VALUE])
            sq_l = 0.0 if sq_l < cfg.deadzone else sq_l
            sq_r = 0.0 if sq_r < cfg.deadzone else sq_r
            self._delta = max(cfg.min_delta, min(cfg.max_delta, self._delta + (sq_r - sq_l) * cfg.speed * cfg.dt))
            outputs["lift_pos"][0] = np.array([self._delta], dtype=np.float32)

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    gantry = GantryDriveRetargeter(GantryDriveConfig(), name="gantry_drive")
    connected_gantry = gantry.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    lift = LiftRetargeter(LiftConfig(), name="lift")
    connected_lift = lift.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )
    _LIFT_HANDLE["instance"] = lift

    # Wrist IK targets — same tuned offsets as the wheeled/pick-place env.
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=180.0,
            target_offset_pitch=0.0,
            target_offset_yaw=90.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=180.0,
            target_offset_pitch=0.0,
            target_offset_yaw=90.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    left_grip = GripperRetargeter(
        GripperRetargeterConfig(hand_side="left", controller_threshold=0.5), name="left_gripper"
    )
    connected_left_grip = left_grip.connect({"controller_left": controllers.output(ControllersSource.LEFT)})
    right_grip = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right", controller_threshold=0.5), name="right_gripper"
    )
    connected_right_grip = right_grip.connect({"controller_right": controllers.output(ControllersSource.RIGHT)})

    # Action layout MUST match ActionsCfg declaration order:
    # [gantry(3), lift(1), upper_body_ik(14: l_pose7 + r_pose7), left_grip(1), right_grip(1)]
    gantry_elements = ["gx", "gy", "gyaw"]
    lift_elements = ["lift"]
    left_ee_elements = ["l_px", "l_py", "l_pz", "l_qx", "l_qy", "l_qz", "l_qw"]
    right_ee_elements = ["r_px", "r_py", "r_pz", "r_qx", "r_qy", "r_qz", "r_qw"]
    left_grip_elements = ["l_grip"]
    right_grip_elements = ["r_grip"]
    output_order = (
        gantry_elements
        + lift_elements
        + left_ee_elements
        + right_ee_elements
        + left_grip_elements
        + right_grip_elements
    )
    reorderer = TensorReorderer(
        input_config={
            "gantry_drive": gantry_elements,
            "lift": lift_elements,
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_gripper": left_grip_elements,
            "right_gripper": right_grip_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "gantry_drive": "array",
            "lift": "array",
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_gripper": "scalar",
            "right_gripper": "scalar",
        },
    )
    connected = reorderer.connect(
        {
            "gantry_drive": connected_gantry.output("gantry_vels"),
            "lift": connected_lift.output("lift_pos"),
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_gripper": connected_left_grip.output("gripper_command"),
            "right_gripper": connected_right_grip.output("gripper_command"),
        }
    )
    return OutputCombiner({"action": connected.output("output")}), [left_se3, right_se3]


_LIFT_HANDLE: dict = {"instance": None}


def _reset_lift(env, env_ids):
    inst = _LIFT_HANDLE.get("instance")
    if inst is not None:
        inst._delta = 0.0  # back to top


@configclass
class OpenArmGantrySceneCfg(InteractiveSceneCfg):
    """Ground plane + bimanual OpenArm on a 4-DOF cartesian gantry."""

    robot: ArticulationCfg = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        self.robot.spawn.usd_path = TRIAL_ARM_GANTRY_USD
        # cart_base is the articulation root; fix it to world so the carriage
        # joints (not the base) provide all motion.
        self.robot.spawn.articulation_props.fix_root_link = True
        self.robot.spawn.rigid_props.disable_gravity = False
        # cart_base is fixed at this height; the robot hangs off the carriage.
        # Must clear the ground even at full lift descent (-0.6 m), else ground
        # contact pins the glide joints and the robot can't move.
        self.robot.init_state.pos = (0.0, 0.0, 0.65)
        self.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # Hold all arm + finger joints at zero, gantry joints at zero.
        self.robot.init_state.joint_pos = {
            **{f"openarm_left_joint{i}": 0.0 for i in range(1, 8)},
            **{f"openarm_right_joint{i}": 0.0 for i in range(1, 8)},
            "openarm_left_finger_joint.*": 0.0,
            "openarm_right_finger_joint.*": 0.0,
            "glide_x": 0.0,
            "glide_y": 0.0,
            "base_yaw": 0.0,
            "lift_z": LIFT_UPPER,
        }

        # Arm actuators — high velocity/effort/stiffness so the joints chase
        # the Pink IK targets quickly. The asset's spec-sheet velocity limits
        # (~2.2 rad/s) are too slow and make joints look "stuck" (e.g. the
        # right shoulder lagging). Override to G1-class responsiveness.
        self.robot.actuators["openarm_arm"].velocity_limit_sim = 25.0
        self.robot.actuators["openarm_arm"].effort_limit_sim = 400.0
        self.robot.actuators["openarm_arm"].stiffness = 2000.0
        self.robot.actuators["openarm_arm"].damping = 150.0

        # Gantry glide + yaw: velocity-controlled.
        self.robot.actuators["gantry"] = ImplicitActuatorCfg(
            joint_names_expr=["glide_x", "glide_y", "base_yaw"],
            velocity_limit_sim=5.0,
            effort_limit_sim=2000.0,
            stiffness=0.0,
            damping=1500.0,
        )
        # Lift: position-controlled, holds chest+arms (~21 kg payload: chest
        # 5 kg + body_link 16 kg + arm links) against gravity. Very high gain
        # so it doesn't sag to the bottom.
        self.robot.actuators["lift"] = ImplicitActuatorCfg(
            joint_names_expr=["lift_z"],
            velocity_limit_sim=1.0,
            effort_limit_sim=20000.0,
            stiffness=120000.0,
            damping=6000.0,
        )


@configclass
class ActionsCfg:
    """Layout (declaration order = action-tensor order):
    [gantry(3), lift(1), upper_body_ik(14), left_gripper(1), right_gripper(1)] = 20-D.
    """

    # Body-relative: raw action [v_fwd, v_strafe, w_yaw] is rotated by the
    # robot's live yaw into the world-axis glide joints.
    gantry = BodyFrameGantryActionCfg(asset_name="robot")
    # use_default_offset=True: target = default(lift_z=LIFT_UPPER, the top) +
    # retargeter delta. A zero delta (pre-teleop / idle) keeps the lift at the
    # top instead of dropping to 0.
    lift = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["lift_z"],
        scale=1.0,
        use_default_offset=True,
    )
    upper_body_ik = OPENARM_GANTRY_IK_ACTION_CFG
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
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    # Write the default joint positions to the sim on reset so lift_z actually
    # spawns at its default (the top, 1.2) instead of the USD-authored 0.
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_lift = EventTerm(func=_reset_lift, mode="reset")


@configclass
class OpenArmGantryEnvCfg(ManagerBasedRLEnvCfg):
    """Glide the bimanual OpenArm gantry with Quest joysticks; lift arms+chest with grip triggers."""

    scene: OpenArmGantrySceneCfg = OpenArmGantrySceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        # Pink IK uses the hand-authored gantry URDF (auto-conversion from the
        # USD fails on this CAD model). Joint list matches the articulation.
        self.actions.upper_body_ik.controller.urdf_path = GANTRY_URDF
        self.actions.upper_body_ik.controller.mesh_path = GANTRY_MESH_ROOT

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, -0.55),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/cart_lift"
        self.xr.fixed_anchor_height = False
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FIXED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_openarm_gantry_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

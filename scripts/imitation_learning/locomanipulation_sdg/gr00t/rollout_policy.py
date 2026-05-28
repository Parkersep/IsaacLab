# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Disjoint navigation")
parser.add_argument("--task", type=str, help="The Isaac Lab disjoint navigation task to load for data generation.")
parser.add_argument("--dataset", type=str, help="The static manipulation dataset recorded via teleoperation.")
parser.add_argument("--output_file", type=str, help="The file name for the generated output dataset.")
parser.add_argument("--demo", type=str, default="demo_0", help="The demo in the input dataset to use.")
parser.add_argument(
    "--randomize_placement", action="store_true", default=False, help="Randomize placement of obstacles."
)
parser.add_argument("--model_path", type=str, help="The path to the model checkpoint.")
parser.add_argument(
    "--embodiment_tag", type=str, default="new_embodiment", help="The embodiment tag to use for the model."
)
parser.add_argument(
    "--policy_quat_format",
    type=str,
    choices=["xyzw", "wxyz"],
    default="xyzw",
    help="Quaternion order the policy uses: 'xyzw' (current Isaac Lab) or 'wxyz' (legacy). "
    "Converts env observations/actions to match. Default is 'xyzw'.",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="Pick up the steering wheel and place it on the bench",
    help="Language instruction passed to the VLA. Should match one of the TASK_TEMPLATES in "
    "convert_dataset.py. For checkpoints trained on the old single-string label, pass "
    "'Pick up and drop off the object'.",
)
parser.add_argument(
    "--inference_interval",
    type=int,
    default=16,
    help="Number of env steps between policy inferences. Must be <= action horizon "
    "(16 for the default config). Smaller values re-plan more often and reduce compounding "
    "drift at the cost of compute.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from policy import Policy
from scipy.spatial.transform import Rotation as ScipyRotation

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.math import convert_quat

import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env import LocomanipulationSDGEnv
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    merge_occupancy_maps,
)
from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose, place_randomly
from isaaclab_mimic.locomanipulation_sdg.transform_utils import (
    transform_inv,
    transform_mul,
)

from isaaclab_tasks.utils import parse_env_cfg


def _sync_simulation_state(env: "LocomanipulationSDGEnv") -> None:
    """Push USD pose writes into physics, step, and sync buffers back out.

    Mirrors :func:`generate_data.sync_simulation_state` so rollout fixture
    randomization is persisted the same way it was during dataset generation.
    Without this, :meth:`place_randomly` writes the new fixture pose to USD but
    the next env step reverts it (and the camera renders the stale pose).
    """
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def _clone_state(state: dict) -> dict:
    """Deep clone state dict so we do not mutate the episode's stored initial state.

    Args:
        state: Nested dict that may contain torch tensors and other dicts.

    Returns:
        Deep copy of state with tensors cloned and dicts recursively cloned.
    """

    def _clone(val):
        if isinstance(val, torch.Tensor):
            return val.clone()
        if isinstance(val, dict):
            return {k: _clone(v) for k, v in val.items()}
        return val

    return _clone(state)


def build_initial_state_for_replay(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
) -> dict:
    """Build the state dict for reset_to so the scene matches the recording in the current fixture frame.

    The episode stores initial state in env-relative coordinates (world - env_origin). The fixtures
    (tables) are not stored; they are reset to scene defaults by _reset_idx.     So we must transform
    robot and object poses from "where they were relative to the recording's fixture" to "where
    they should be relative to the current scene's start fixture", and pass explicit zero velocities
    so the sim starts at rest. This is the single source of truth for initial state—no post-reset
    projection or sync.

    Returns:
        State dict suitable for env.reset_to(..., is_relative=True).
    """
    state = _clone_state(input_episode_data.get_initial_state())
    load_0 = env.load_input_data(input_episode_data, 0)
    start_fixture_pose = env.get_start_fixture().get_pose()
    recorded_fixture = load_0.fixture_pose.to(env.device)
    if recorded_fixture.dim() == 1:
        recorded_fixture = recorded_fixture.unsqueeze(0)

    env_origins = env.scene.env_origins
    env_id = 0
    if env_origins.dim() == 2:
        env_origin = env_origins[env_id : env_id + 1]
    else:
        env_origin = env_origins.unsqueeze(0).to(env.device)

    def to_world(pose: torch.Tensor) -> torch.Tensor:
        p = pose.clone().to(env.device)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        p = p.clone()
        p[:, :3] += env_origin
        return p

    def to_env_relative(world_pose: torch.Tensor) -> torch.Tensor:
        out = world_pose.clone()
        out[:, :3] -= env_origin
        return out

    # Robot: same relative pose to current start fixture as in recording.
    r_pose = state["articulation"]["robot"]["root_pose"].clone().to(env.device)
    if r_pose.dim() == 1:
        r_pose = r_pose.unsqueeze(0)
    recorded_robot_world = r_pose.clone()
    recorded_robot_world[:, :3] += env_origin
    desired_robot_world = transform_mul(
        start_fixture_pose,
        transform_mul(transform_inv(recorded_fixture), recorded_robot_world),
    )
    state["articulation"]["robot"]["root_pose"] = to_env_relative(desired_robot_world)
    state["articulation"]["robot"]["root_velocity"] = torch.zeros_like(
        state["articulation"]["robot"]["root_velocity"], device=env.device
    )
    state["articulation"]["robot"]["joint_velocity"] = torch.zeros_like(
        state["articulation"]["robot"]["joint_velocity"], device=env.device
    )

    # Object: same.
    o_pose = state["rigid_object"]["object"]["root_pose"].clone().to(env.device)
    if o_pose.dim() == 1:
        o_pose = o_pose.unsqueeze(0)
    recorded_object_world = o_pose.clone()
    recorded_object_world[:, :3] += env_origin
    desired_object_world = transform_mul(
        start_fixture_pose,
        transform_mul(transform_inv(recorded_fixture), recorded_object_world),
    )
    state["rigid_object"]["object"]["root_pose"] = to_env_relative(desired_object_world)
    state["rigid_object"]["object"]["root_velocity"] = torch.zeros_like(
        state["rigid_object"]["object"]["root_velocity"], device=env.device
    )

    return state


# State keys that contain a 7D pose (pos 3 + quat 4). Env always provides quat in XYZW.
_STATE_POSE_KEYS = (
    "state.left_hand_pose",
    "state.right_hand_pose",
    "state.object_pose",
    "state.goal_pose",
    "state.end_fixture_pose",
)


def _to_sdg_relative_pose(target_pose_xyzw: np.ndarray, base_pose_xyzw: np.ndarray) -> np.ndarray:
    """Compute relative pose matching convert_dataset.py's encoding convention.

    Intentionally uses scalar_first=True on XYZW data (same mismatch as training data
    was generated with) so the model receives the same distribution it was trained on.

    Args:
        target_pose_xyzw: World pose [x, y, z, qx, qy, qz, qw] shape (7,).
        base_pose_xyzw: Base pose [x, y, z, qx, qy, qz, qw] shape (7,).

    Returns:
        Relative pose in SDG encoding, shape (7,).
    """
    r_base = ScipyRotation.from_quat(base_pose_xyzw[3:], scalar_first=True)
    r_target = ScipyRotation.from_quat(target_pose_xyzw[3:], scalar_first=True)
    r_rel = r_base.inv() * r_target
    t_rel = r_base.inv().apply(target_pose_xyzw[:3] - base_pose_xyzw[:3])
    return np.concatenate([t_rel, r_rel.as_quat(scalar_first=True)])


def _from_sdg_relative_pose(relative_pose: np.ndarray, base_pose_xyzw: np.ndarray) -> np.ndarray:
    """Inverse of _to_sdg_relative_pose; output numerically equals the original XYZW world pose.

    Args:
        relative_pose: Relative pose in SDG encoding, shape (7,).
        base_pose_xyzw: Base pose [x, y, z, qx, qy, qz, qw] shape (7,).

    Returns:
        World pose numerically identical to original XYZW pose, shape (7,).
    """
    r_base = ScipyRotation.from_quat(base_pose_xyzw[3:], scalar_first=True)
    r_rel = ScipyRotation.from_quat(relative_pose[3:], scalar_first=True)
    r_world = r_base * r_rel
    t_world = r_base.apply(relative_pose[:3]) + base_pose_xyzw[:3]
    return np.concatenate([t_world, r_world.as_quat(scalar_first=True)])


def _convert_pose_quat(pose: torch.Tensor, to_fmt: str) -> torch.Tensor:
    """Convert quaternion part of pose (..., 7) to target format. Env is XYZW.

    Args:
        pose: Pose tensor with last 4 dims as quat (xyzw from env).
        to_fmt: "xyzw" (no-op) or "wxyz" for policy.

    Returns:
        Pose tensor with quat in the requested format.
    """
    if to_fmt == "xyzw":
        return pose
    # to_fmt == "wxyz": env is xyzw -> convert to wxyz for policy
    pos, quat = pose[..., :3], pose[..., 3:7]
    quat_wxyz = convert_quat(quat, to="wxyz")
    return torch.cat([pos, quat_wxyz], dim=-1)


def _convert_action_pose_quats_to_env(action: torch.Tensor, policy_quat_format: str) -> None:
    """Convert policy action pose quaternions (columns 0:7 and 7:14) to XYZW for env in-place.

    Args:
        action: Action tensor of shape (..., 32) with pose quats in columns 3:7 and 10:14.
        policy_quat_format: "xyzw" (no-op) or "wxyz" (convert to xyzw for env).
    """
    if policy_quat_format == "xyzw":
        return
    # Policy output is WXYZ; env expects XYZW
    for start in (0, 7):
        quat = action[..., start + 3 : start + 7]
        action[..., start + 3 : start + 7] = convert_quat(quat, to="xyzw")


def setup_navigation_scene(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    approach_distance: float,
    randomize_placement: bool = True,
) -> tuple[OccupancyMap | None, RelativePose]:
    """Set up occupancy map and base goal for policy rollout (no path planning).

    Args:
        env: The locomanipulation SDG environment.
        input_episode_data: Input episode data used to compute base goal from initial state.
        approach_distance: Unused; kept for API compatibility with generate_data.
        randomize_placement: Whether to randomize end fixture and obstacle placement.

    Returns:
        Tuple of (occupancy_map, base_goal). First element is None; base_goal is the goal
        pose for the policy relative to the end fixture.
    """
    # Create base occupancy map
    occupancy_map = merge_occupancy_maps(
        [
            OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
            env.get_start_fixture().get_occupancy_map(),
        ]
    )

    # Randomize fixture placement if enabled
    fixtures = [env.get_end_fixture()] + env.get_obstacle_fixtures()
    for fixture in fixtures:
        if randomize_placement:
            place_randomly(fixture, occupancy_map.buffered_meters(0.7), num_iter=5000)
            # Sync each placement so subsequent fixtures see the updated poses
            # and so the write is not reverted by the next env.step.
            _sync_simulation_state(env)
        occupancy_map = merge_occupancy_maps([occupancy_map, fixture.get_occupancy_map()])

    # Compute goal poses from initial state (for policy's base_goal; robot/object already set by reset_to).
    initial_state = env.load_input_data(input_episode_data, 0)
    base_goal = RelativePose(
        relative_pose=transform_mul(transform_inv(initial_state.fixture_pose), initial_state.base_pose),
        parent=env.get_end_fixture(),
    )

    # Flush physics + render so the camera image and scene buffers reflect the
    # randomized fixture poses before the policy's first inference.
    if randomize_placement:
        _sync_simulation_state(env)
        env.sim.render()
        env.obs_buf = env.observation_manager.compute(update_history=True)

    return None, base_goal


def build_model_input(
    env: LocomanipulationSDGEnv,
    base_goal: RelativePose,
    policy_quat_format: str = "xyzw",
    task_description: str = "Pick up the steering wheel and place it on the bench",
):
    """Build GR00T model input dict and dummy action from current env state.

    Poses are expressed relative to the robot base. State pose quats are converted
    to policy format (xyzw or wxyz) if needed.

    Args:
        env: The locomanipulation SDG environment.
        base_goal: Goal pose (e.g. from setup_navigation_scene).
        policy_quat_format: "xyzw" or "wxyz" for state pose quaternions.
        task_description: Language instruction for the VLA. Must match one of the
            TASK_TEMPLATES in convert_dataset.py used to train the checkpoint.

    Returns:
        Tuple of (model_input dict, dummy_action tensor) for the policy.
    """
    obs = env.obs_buf
    left_hand_pose = torch.cat([obs["policy"]["left_eef_pos"], obs["policy"]["left_eef_quat"]], dim=-1)
    right_hand_pose = torch.cat([obs["policy"]["right_eef_pos"], obs["policy"]["right_eef_quat"]], dim=-1)
    left_hand_joint_positions = obs["policy"]["hand_joint_state"][:, 0:7]
    right_hand_joint_positions = obs["policy"]["hand_joint_state"][:, 7:14]

    base_pose = env.get_base().get_pose()
    object_pose = env.get_object().get_pose()
    goal_pose = base_goal.get_pose()
    end_fixture_pose = env.get_end_fixture().get_pose()

    # Use the same scipy scalar_first=True convention as convert_dataset.py so state poses
    # match the training distribution (round-trip consistent despite the XYZW/WXYZ mismatch).
    base_pose_np = base_pose.cpu().numpy().squeeze()  # (7,) XYZW

    def _rel(pose_tensor: torch.Tensor) -> torch.Tensor:
        """Convert (1, 7) world pose to (1, 7) SDG-relative pose matching training encoding."""
        pose_np = pose_tensor.cpu().numpy().squeeze()  # (7,)
        rel_np = _to_sdg_relative_pose(pose_np, base_pose_np).astype(np.float32)
        return torch.from_numpy(rel_np).unsqueeze(0)

    # Build state tensors (torch) before converting to numpy.
    state_tensors = {
        "state.left_hand_pose": _rel(left_hand_pose),
        "state.right_hand_pose": _rel(right_hand_pose),
        "state.left_hand_joint_positions": left_hand_joint_positions,
        "state.right_hand_joint_positions": right_hand_joint_positions,
        "state.object_pose": _rel(object_pose),
        "state.goal_pose": _rel(goal_pose),
        "state.end_fixture_pose": _rel(end_fixture_pose),
    }
    # SDG scipy encoding already matches training distribution; no quat-format conversion needed.

    # Convert to numpy arrays in the format Gr00tSimPolicyWrapper expects:
    # video: np.uint8 (B, T, H, W, C), state: np.float32 (B, T, D), language: tuple[str] (B,)
    video = obs["policy"]["robot_pov_cam"]  # (B, H, W, C) torch uint8
    model_input = {
        "video.ego_view": video.unsqueeze(1).cpu().numpy().astype(np.uint8),
        "annotation.human.action.task_description": (task_description,),  # language instruction (B,)
    }
    for key, val in state_tensors.items():
        model_input[key] = val.unsqueeze(1).cpu().numpy().astype(np.float32)

    dummy_action = torch.zeros(1, 32)
    dummy_action[:, :28] = torch.cat(
        [left_hand_pose, right_hand_pose, left_hand_joint_positions, right_hand_joint_positions], dim=1
    )
    dummy_action[:, 31] = 0.8

    return model_input, dummy_action


def eval_policy(
    env: LocomanipulationSDGEnv,
    policy: Policy,
    input_episode_data: EpisodeData,
    randomize_placement: bool = True,
    policy_quat_format: str = "xyzw",
    task_description: str = "Pick up the steering wheel and place it on the bench",
    inference_interval: int = 16,
) -> None:
    """Run policy rollout in the environment with state machine and recording-based initial state.

    Resets env to the episode initial state, sets up navigation scene (occupancy map and goal),
    then steps the environment using policy actions at a fixed inference interval. Handles
    quaternion format conversion between env (xyzw) and policy (xyzw or wxyz).

    Args:
        env: The locomanipulation SDG environment.
        policy: The GR00T policy wrapper.
        input_episode_data: Episode data for initial state and goal.
        randomize_placement: Whether to randomize fixture placement.
        policy_quat_format: Quaternion format expected by the policy ("xyzw" or "wxyz").
    """
    obs, _ = env.reset_to(
        state=_clone_state(input_episode_data.get_initial_state()),
        env_ids=torch.tensor([0], device=env.device),
        is_relative=False,
    )
    occupancy_map, base_goal = setup_navigation_scene(
        env, input_episode_data, approach_distance=0.5, randomize_placement=randomize_placement
    )

    step = 0
    action_idx = 0

    while simulation_app.is_running() and not simulation_app.is_exiting():
        if step % inference_interval == 0:
            model_input, dummy_action = build_model_input(env, base_goal, policy_quat_format, task_description)
            action_dict, _ = policy.policy.get_action(model_input)
            # action_dict values are np.float32 (B, T, D) — take first batch, convert to torch
            action_buffer = torch.cat([torch.from_numpy(v[0]) for v in action_dict.values()], dim=-1)
            action_idx = 0

        if step < 0:
            env.step(dummy_action)
        else:
            action = action_buffer[action_idx].clone()  # (32,)
            # Decode EEF poses from SDG-relative back to world frame.
            # convert_dataset.py encodes action[t+k] relative to base[t+k], so we must decode
            # using the CURRENT base pose (which is base[t+k] at step k), not a pose captured
            # at inference time. Using a stale inference-time pose causes EEF targets to land
            # in the wrong world frame whenever the base has moved (e.g. during navigation).
            current_base_pose_np = env.get_base().get_pose().cpu().numpy().squeeze()
            for pose_start in (0, 7):
                rel_np = action[pose_start : pose_start + 7].numpy()
                world_np = _from_sdg_relative_pose(rel_np, current_base_pose_np)
                action[pose_start : pose_start + 7] = torch.from_numpy(world_np.astype(np.float32))

            action[28:31] = action[28:31] * 1.0
            _, _, reset_terminated, reset_time_outs, _ = env.step(action.unsqueeze(0))
            if reset_terminated.any() or reset_time_outs.any():
                print("Episode ended, resetting rollout state")
                env.reset_to(
                    state=_clone_state(input_episode_data.get_initial_state()),
                    env_ids=torch.tensor([0], device=env.device),
                    is_relative=False,
                )
                occupancy_map, base_goal = setup_navigation_scene(
                    env,
                    input_episode_data,
                    approach_distance=0.5,
                    randomize_placement=randomize_placement,
                )
                policy.policy.reset()
                step = 0
                action_idx = 0
                continue

        step += 1
        action_idx += 1


if __name__ == "__main__":
    with torch.no_grad():
        env_name = args_cli.task.split(":")[-1] if args_cli.task is not None else None
        if env_name is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        policy = Policy(model_path=args_cli.model_path, embodiment_tag=args_cli.embodiment_tag)

        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
        env_cfg.sim.device = args_cli.device
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.output_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.output_file)
        # Remove the SDG output recorder — it requires _locomanipulation_sdg_output_data
        # which is only populated during data generation, not policy rollout.
        if hasattr(env_cfg.recorders, "record_pre_step_locomanipulation_sdg_output_data"):
            env_cfg.recorders.record_pre_step_locomanipulation_sdg_output_data = None

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        input_dataset_file_handler = HDF5DatasetFileHandler()
        input_dataset_file_handler.open(args_cli.dataset)
        input_episode_data = input_dataset_file_handler.load_episode(args_cli.demo, args_cli.device)

        eval_policy(
            env=env,
            policy=policy,
            input_episode_data=input_episode_data,
            randomize_placement=args_cli.randomize_placement,
            policy_quat_format=args_cli.policy_quat_format,
            task_description=args_cli.task_description,
            inference_interval=args_cli.inference_interval,
        )

        env.reset()
        env.close()

        simulation_app.close()

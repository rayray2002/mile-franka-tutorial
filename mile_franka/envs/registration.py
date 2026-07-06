from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from mile_franka.config import FR3_DOWN_QUAT, MULTIPANDA_DOWN_QUAT, StackTaskConfig
from mile_franka.envs.fake_backend import FakeRobotBackend, FakeWorld, WorldPoseSource
from mile_franka.envs.franka_env import FrankaEnv

FAKE_ENV_ID = "Franka-Stack-Fake-v0"
SIM_ENV_ID = "Franka-Stack-Sim-v0"
REAL_ENV_ID = "Franka-Stack-Real-v0"

# Default temporal window for the Franka path: 10 frames @ 10 Hz = 1.0 s, long enough to
# disambiguate descend / hold / lift at the grasp (the 4-frame/0.4 s window aliased them).
# The MetaWorld path keeps its own FrameStack(4) — do not reuse this constant there.
FRANKA_FRAME_STACK = 10


def _build_fake_env(config: Optional[StackTaskConfig] = None,
                    obs_include_bottom_z: bool = False) -> FrankaEnv:
    config = config if config is not None else StackTaskConfig()
    world = FakeWorld(config)
    return FrankaEnv(FakeRobotBackend(world), WorldPoseSource(world), config, mode="sim",
                     obs_include_bottom_z=obs_include_bottom_z)


def _build_sim_env(config: Optional[StackTaskConfig] = None,
                   obs_include_bottom_z: bool = False) -> FrankaEnv:
    """Multipanda MuJoCo-sim env. Imports ROS lazily (only when this id is made)."""
    from mile_franka.envs.ros_backend import MultipandaRosBackend
    from mile_franka.pose.mujoco_gt import MujocoGtPoseSource

    # Align with the MJCF stacking scene: 5 cm cubes (half-extent 0.025), floor at world z=0.
    # 300-step horizon: the scripted policy's 7-phase sequence needs ~150-250 steps against the
    # real impedance controller (which tracks more slowly than the fake backend).
    config = config if config is not None else StackTaskConfig(
        cube_size=0.05,
        table_z=0.0,
        workspace_low=np.array([0.40, -0.18, 0.02], dtype=np.float32),
        workspace_high=np.array([0.75, 0.18, 0.40], dtype=np.float32),
        max_steps=1_000,
    )
    backend = MultipandaRosBackend(
        config,
        apply_sim_gains=True,
        reset_controller_target_on_reset=True,
        # NOTE: move_to_start_example_controller is NOT in the multipanda sim controller config
        # (franka_bringup/config/sim/single_sim_controllers.yaml) -- it exists only in the real
        # config -- so joint-space homing via that controller is unavailable in sim. The sim
        # instead homes via the Cartesian impedance controller (prime->activate captures the EE,
        # then _home() ramps to the fixed home with DOWN_QUAT); with SIM_STACKING_GAINS this
        # holds the wrist down well enough that scripted/expert rollouts stack reliably. The
        # known-wrist-down-joint-config approach (config.Q_HOME / move_to_start) is the REAL path
        # only (real config defines move_to_start_example_controller). See the phase-b spec §5.8.
        move_to_start_on_reset=False,
        env_step_period_s=0.1,
        home_steps=50,
        # Settle to ~1.5cm; the scripted/expert rollouts re-home via their descent so a tight
        # start isn't critical -- cap the budget modestly to keep 100-episode collection fast.
        home_settle_tol=0.015,
        home_settle_timeout_s=4.0,
        down_quat=MULTIPANDA_DOWN_QUAT,
    )
    return FrankaEnv(backend, MujocoGtPoseSource(backend), config, mode="sim",
                     obs_include_bottom_z=obs_include_bottom_z)


def _build_real_env(config: Optional[StackTaskConfig] = None,
                    obs_include_bottom_z: bool = False) -> FrankaEnv:
    """Real FR3 env with AprilTag object poses + multipanda_ros2 Cartesian controller.

    Requires the hucebot multipanda_ros2 controller docker running on the FR3 control PC,
    the RealSense D415 + apriltag_ros launched (``make apriltag-up``), and a valid
    ``config/camera_calib.yaml`` from the calibration capture script. The operator must
    be present — this env moves the real arm on reset (joint-space home via
    move_to_start, then Cartesian-impedance home) and on every step.
    """
    import rclpy

    from mile_franka.envs.ros_backend import MultipandaRosBackend
    from mile_franka.pose.apriltag import AprilTagPoseSource
    from mile_franka.pose.calibration import load_camera_calibration
    from mile_franka.pose.canonical import CanonicalCubePoseSource, TablePlaneCanonicalizer

    config = config if config is not None else StackTaskConfig(
        cube_size=0.05,
        table_z=0.0,                        # table surface at base origin (verify on hardware)
        workspace_low=np.array([0.35, -0.25, 0.02], dtype=np.float32),
        workspace_high=np.array([0.75, 0.25, 0.35], dtype=np.float32),
        max_steps=10_000,                    # human-paced (operator-gated reset)
    )

    if not rclpy.ok():
        # No rclpy SIGINT handler: its default shuts down the context on Ctrl-C, which
        # kills the node mid-eval (the operator Ctrl-Cs to end an episode), so the next
        # reset() fails with "rcl node's context is invalid". NO lets Ctrl-C raise a
        # normal KeyboardInterrupt the script handles, keeping the context alive. This is
        # the first init in the real path (runs before the backend's), so it wins.
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # Separate node from the backend's internal node so the AprilTag tf2 listener has its
    # own spinner domain. Name must not collide with the backend's "mile_franka_backend".
    node = rclpy.create_node("mile_franka_real_pose")

    real_stack = os.environ.get("MILE_REAL_STACK", "multipanda").lower()
    use_fr3_pose = real_stack in ("fr3", "franka_ros2", "fr3_pose")
    # The calibration static tf and the tag tf are rooted at this frame; the fr3 stack
    # publishes fr3_link0 (panda_link0 is absent), so the pose source MUST look tags up
    # under the same root the backend and apriltag launch use, or every tf lookup fails.
    base_frame = "fr3_link0" if use_fr3_pose else "panda_link0"

    # On real hardware the controller has its own tuned gains, so default to native.
    # MILE_APPLY_SIM_GAINS=1 is only for validating against the lab sim.
    apply_sim_gains = os.environ.get("MILE_APPLY_SIM_GAINS", "0").lower() in ("1", "true", "yes")

    # Gripper action namespace differs by stack: the franka_ros2 gripper node comes up as
    # `franka_gripper` (empty namespace) -> /franka_gripper/grasp, while this lab's
    # multipanda_ros2 bringup names it `panda_gripper` -> /panda_gripper/grasp (confirmed via
    # `ros2 action list` against realtime_franka_humble). MILE_GRASP_ACTION overrides either.
    grasp_action = os.environ.get(
        "MILE_GRASP_ACTION",
        "/franka_gripper/grasp" if use_fr3_pose else "/panda_gripper/grasp")
    # Action name differs by stack; override with MILE_ERROR_RECOVERY_ACTION.
    error_recovery_action = os.environ.get(
        "MILE_ERROR_RECOVERY_ACTION",
        "/franka_control/error_recovery" if use_fr3_pose else "/error_recovery")
    # MILE_CONTROLLER overrides the default. See docs/bringup-reference.md.
    controller_name = os.environ.get(
        "MILE_CONTROLLER",
        "custom_cartesian_impedance_controller")

    backend = MultipandaRosBackend(
        config,
        sim=False,                           # real Franka — no mujoco_ros services
        base_frame=base_frame,
        randomize_on_reset=False,            # cannot teleport real cubes
        grasp_action=grasp_action,
        controller_name=controller_name,
        # Do NOT use move_to_start on the FR3: the controller-mode round-trip
        # (impedance Effort -> move_to_start CartesianPose -> back to Effort)
        # trips communication_constraints_violation in libfranka ("Cannot perform
        # this operation while another control or read operation is running") when
        # the CartesianPose motion generator is still active.  Instead the
        # Cartesian-impedance controller activates at whatever orientation the arm
        # is already at, and _home() ramps orientation under impedance control
        # (weaker rot_stiff=25 but converges over the settling budget when the
        # start orientation is close to down, which it is between episodes).
        move_to_start_on_reset=False,
        reset_controller_target_on_reset=True,
        apply_sim_gains=apply_sim_gains,
        env_step_period_s=0.1,
        home_steps=50,
        # Settle to ~1.5cm before an episode starts. Real/native gains converge slower than the
        # sim, so allow a larger budget; the loop exits early once within tolerance.
        home_settle_tol=0.015,
        home_settle_timeout_s=8.0,
        error_recovery_action=error_recovery_action,
        down_quat=(FR3_DOWN_QUAT if use_fr3_pose else MULTIPANDA_DOWN_QUAT),
    )
    # Load the eye-to-hand calibration so AprilTagPoseSource can compose
    # camera→tag (from apriltag_ros) with camera→base directly in Python,
    # bypassing the need for a camera→base static transform in tf2.  This
    # avoids CycloneDDS+iceoryx shared-memory exhaustion silently dropping the
    # one-shot /tf_static publisher when it starts after initial DDS discovery.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _calib_path = os.environ.get("MILE_CAMERA_CALIB",
                                 os.path.join(_repo_root, "config", "camera_calib.yaml"))
    _camera_to_base = None
    if os.path.exists(_calib_path):
        _camera_to_base = load_camera_calibration(_calib_path).camera_to_base
    else:
        print(f"[mile_franka] {_calib_path} not found — camera→base tf2 chain required; "
              "run calibrate_camera.py or set MILE_CAMERA_CALIB")
    # Orientation canonicalization only: the reduced observation drops the cube quaternions,
    # but CanonicalCubePoseSource still stamps the training quaternion so any consumer reading
    # full poses (e.g. a scripted intervener via privileged_frame) sees the training convention.
    # Vertical canonicalization is now ADAPTIVE (z_offset=0 here): the env applies
    # TablePlaneCanonicalizer, which measures the table height from the resting cube each step
    # instead of assuming a fixed +cube_size/2, so an unknown real table height is corrected.
    pose_source = CanonicalCubePoseSource(
        AprilTagPoseSource(half_edge=0.025, node=node, base_frame=base_frame,
                           camera_to_base=_camera_to_base),
        z_offset=0.0)
    z_canonicalizer = TablePlaneCanonicalizer(train_rest_z=config.cube_size / 2.0)
    return FrankaEnv(backend, pose_source, config, mode="real",
                     obs_include_bottom_z=obs_include_bottom_z,
                     z_canonicalizer=z_canonicalizer)


def make_franka_env(env: FrankaEnv, frame_stack: int = FRANKA_FRAME_STACK) -> gym.Env:
    """Apply FrameStack + FlattenObservation to a FrankaEnv (matches the MetaWorld path)."""
    return FlattenObservation(FrameStack(env, frame_stack))


def register_franka_envs() -> None:
    """Register Franka gym ids (idempotent)."""
    if FAKE_ENV_ID not in gym.registry:
        gym.register(id=FAKE_ENV_ID, entry_point=_build_fake_env)
    if SIM_ENV_ID not in gym.registry:
        gym.register(id=SIM_ENV_ID, entry_point=_build_sim_env)
    if REAL_ENV_ID not in gym.registry:
        gym.register(id=REAL_ENV_ID, entry_point=_build_real_env)

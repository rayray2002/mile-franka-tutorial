"""RobotBackend backed by hucebot's multipanda_ros2 MuJoCo sim (or real hardware).

EE position comes from /cartesian_impedance/cartesian_pos_curr (O_T_EE, panda_link0 frame).
Sim cube poses come from mujoco_ros get_body_state (world frame, flip ±xy into base).
All ROS imports are lazy so the package imports with no ROS installed.
See docs/bringup-reference.md for topic names, FR3 quirks, and iceoryx workaround notes.
"""
from __future__ import annotations

import os
import re
import time
import subprocess
from typing import Optional

import numpy as np

from mile_franka.config import DOWN_QUAT, StackTaskConfig
from mile_franka.envs.backend import RobotBackend

EQUILIBRIUM_TOPIC = "/cartesian_impedance/equilibrium_pose"  # PoseStamped command (sim==real)
EE_CURR_TOPIC = "/cartesian_impedance/cartesian_pos_curr"  # O_T_EE PoseStamped, panda_link0 (sim==real)
CONTROLLER_NAME = "custom_cartesian_impedance_controller"  # sim==real
# Sim-only mujoco_ros services (not present on the real graph; gated by sim=True).
GET_BODY_STATE_SRV = "/get_body_state"
SET_BODY_STATE_SRV = "/set_body_state"
SET_PAUSE_SRV = "/set_pause"
GRASP_ACTION = "/panda_gripper_sim_node/grasp"  # sim gripper; real (multipanda) = /panda_gripper/grasp
EE_BODY = "panda_hand"  # used only if EE_CURR_TOPIC is unavailable (sim fallback)
# move_to_start exists ONLY in the real controller config (not the sim config); used for the
# real wrist-down joint-space home. See config.Q_HOME and envs/registration.py.
MOVE_TO_START_CONTROLLER = "move_to_start_example_controller"
ENV_STEP_PERIOD_S = 0.1  # ~10 Hz Python env command/observation step
SIM_STACKING_GAINS = {
    "pos_stiff": 4000.0,
    "translational_clip": 0.5,
    "translational_Ki": 0.0,
    "rot_stiff": 800.0,
    "rotational_Ki": 0.0,
    "ns_stiff_q1_to_4": 0.0,
    "ns_stiff_q5_to_7": 0.0,
}


class MultipandaRosBackend(RobotBackend):
    """RobotBackend over a live multipanda_ros2 node (MuJoCo sim or hardware)."""

    def __init__(self, config: Optional[StackTaskConfig] = None,
                 sim: bool = True,
                 base_frame: str = "panda_link0", ee_body: str = EE_BODY,
                 randomize_on_reset: bool = True,
                 controller_name: str = CONTROLLER_NAME,
                 move_to_start_controller: str = MOVE_TO_START_CONTROLLER,
                 grasp_action: str = GRASP_ACTION,
                 activate_controller_on_reset: bool = True,
                 apply_sim_gains: bool = False,
                 reset_controller_target_on_reset: bool = False,
                 move_to_start_on_reset: bool = False,
                 env_step_period_s: float = ENV_STEP_PERIOD_S,
                 move_to_start_hold_s: float = 8.0,
                 home_steps: int = 80,
                 home_settle_tol: float = 0.015,
                 home_settle_timeout_s: float = 6.0,
                 setpoint_substep_m: float = 0.003,
                 error_recovery_action: Optional[str] = None,
                 down_quat: Optional[np.ndarray] = None):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from franka_msgs.action import Grasp
        from rclpy.action import ActionClient
        try:
            # Move = "position the fingers to a width" (no grasp force, moves either
            # direction). This is the correct primitive to OPEN: franka_gripper's Grasp
            # only clamps inward with force and reports success even when it closes, so
            # grasp(0.08) physically *closes* the gripper. Some sim gripper nodes lack
            # Move; we fall back to Grasp for opening there (sim honors the width).
            from franka_msgs.action import Move as _Move
        except Exception:
            _Move = None
        try:
            from franka_msgs.action import ErrorRecovery as _ErrorRecovery
        except Exception:  # older franka_msgs without the action — recovery just no-ops
            _ErrorRecovery = None

        # sim=True joins the multipanda MuJoCo graph (mujoco_ros services for pause +
        # ground-truth body poses); sim=False is the real-FR3 path: no mujoco_ros, the world
        # is never paused or teleported, and object poses come from an external source
        # (AprilTag) rather than get_body_state. Every mujoco-only call below is gated on this.
        self.sim = bool(sim)
        if not self.sim and randomize_on_reset:
            raise ValueError("randomize_on_reset is sim-only (cannot teleport real cubes); "
                             "the real env must place cubes via an operator-gated reset")

        self.config = config if config is not None else StackTaskConfig()
        self.down_quat = np.asarray(down_quat if down_quat is not None else DOWN_QUAT,
                                    dtype=np.float32)
        self.base_frame = base_frame
        self.ee_body = ee_body
        self.randomize_on_reset = randomize_on_reset
        self.controller_name = controller_name
        self.move_to_start_controller = move_to_start_controller
        self.activate_controller_on_reset = activate_controller_on_reset
        self.reset_controller_target_on_reset = reset_controller_target_on_reset
        self.move_to_start_on_reset = move_to_start_on_reset
        self.env_step_period_s = float(env_step_period_s)
        self.move_to_start_hold_s = float(move_to_start_hold_s)
        self.home_steps = int(home_steps)
        self.home_settle_tol = float(home_settle_tol)
        self.home_settle_timeout_s = float(home_settle_timeout_s)
        self.setpoint_substep_m = float(setpoint_substep_m)
        self._controller_activated_by_backend = False

        if not rclpy.ok():
            # Ctrl-C raises KeyboardInterrupt rather than tearing down the rclpy context between episodes.
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._rclpy = rclpy
        self._node = rclpy.create_node("mile_franka_backend")
        self._PoseStamped = PoseStamped
        self._Grasp = Grasp
        self._Move = _Move

        self._pub = self._node.create_publisher(PoseStamped, EQUILIBRIUM_TOPIC, 10)
        self._grasp = ActionClient(self._node, Grasp, grasp_action)
        # Move action shares the gripper namespace ("…/grasp" -> "…/move"). Used to OPEN.
        move_action = grasp_action.rsplit("/grasp", 1)[0] + "/move"
        self._move = (ActionClient(self._node, _Move, move_action)
                      if _Move is not None else None)

        # Error-recovery action (real only). After a motion reflex the FrankaHardwareInterface
        # drops the command interfaces until a franka_msgs/ErrorRecovery goal is sent, which
        # otherwise makes the next reset's controller switch fail. No client => recovery no-ops.
        self._ErrorRecovery = _ErrorRecovery
        self._error_recovery = None
        if not self.sim and error_recovery_action and _ErrorRecovery is not None:
            self._error_recovery = ActionClient(
                self._node, _ErrorRecovery, error_recovery_action)

        # mujoco_ros services (pause + GT body poses) exist only in the sim graph.
        self._get_body = self._set_body = self._set_pause = None
        self._SetBodyState = self._SetPause = None
        if self.sim:
            from mujoco_ros_msgs.srv import GetBodyState, SetBodyState, SetPause
            self._SetBodyState = SetBodyState
            self._SetPause = SetPause
            self._get_body = self._node.create_client(GetBodyState, GET_BODY_STATE_SRV)
            self._set_body = self._node.create_client(SetBodyState, SET_BODY_STATE_SRV)
            self._set_pause = self._node.create_client(SetPause, SET_PAUSE_SRV)
            required_clients = [(self._get_body, GET_BODY_STATE_SRV),
                                (self._set_body, SET_BODY_STATE_SRV),
                                (self._set_pause, SET_PAUSE_SRV)]
            for client, name in required_clients:
                if not client.wait_for_service(timeout_sec=15.0):
                    raise RuntimeError(f"multipanda service {name} not available")

        # Subscribe to the controller's EE pose (O_T_EE, already in panda_link0 frame).
        # Cached so get_ee_position() stays non-blocking.
        self._ee_curr_pose: Optional[np.ndarray] = None
        self._ee_curr_quat: Optional[np.ndarray] = None  # xyzw, base frame
        self._node.create_subscription(
            PoseStamped, EE_CURR_TOPIC,
            self._ee_curr_callback, 1)

        self._gripper_width = float(self.config.gripper_open_width)
        self._gripper_closed = False  # last commanded state
        # Always ensure the controllers we depend on are actually loaded before reset()
        # tries to activate them. On a slow cold boot the launch-time spawners can lose
        # the race against controller_manager and die (exit 1), leaving the controllers
        # never loaded -- self-heal that here rather than racing param-list.
        self._wait_for_controller_node()
        if apply_sim_gains:
            self._apply_controller_gains(SIM_STACKING_GAINS)

    # --- EE pose cache ---------------------------------------------------------
    def _ee_curr_callback(self, msg) -> None:
        """Cache the controller's current EE pose (panda_link0 frame, no flip needed)."""
        p = msg.pose.position
        self._ee_curr_pose = np.array([p.x, p.y, p.z], dtype=np.float32)
        o = msg.pose.orientation
        self._ee_curr_quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float32)

    def _spin_once_for_ee(self, timeout: float = 0.5) -> None:
        """Spin briefly to let the subscription callback fire and populate _ee_curr_pose."""
        deadline = time.time() + timeout
        while self._ee_curr_pose is None and time.time() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

    # --- rclpy helpers ---------------------------------------------------------
    def _call(self, client, request, timeout: float = 5.0):
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError("multipanda service call timed out")
        return future.result()

    def _run_ros2(self, args: list[str], *, timeout: float = 10.0,
                  check: bool = True, no_iceoryx: bool = False) -> subprocess.CompletedProcess:
        # Disable iceoryx shared-memory for CLI subprocesses in sim; it breaks node discovery.
        # See docs/bringup-reference.md for details.
        env = None
        if no_iceoryx or self.sim:
            env = {**os.environ, "CYCLONEDDS_URI": ""}
        try:
            result = subprocess.run(
                ["ros2", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            # A `ros2 control`/`ros2 param` CLI that hangs to the timeout almost always means
            # there is no controller_manager on the DDS graph -- i.e. the controller/sim (or the
            # real franka_control2 bringup) is not running. Surface that instead of an opaque
            # TimeoutExpired traceback out of backend construction. check=False callers
            # (e.g. _list_controllers) get a synthetic failed result so they can retry/degrade.
            if check:
                raise RuntimeError(
                    f"ros2 {' '.join(args)} timed out after {timeout:.0f}s -- is the "
                    "controller running on this DDS graph? (start the sim with `make sim-up`, "
                    "or the real Franka bringup, before the real env)") from exc
            return subprocess.CompletedProcess(args, returncode=124, stdout="",
                                               stderr="timed out")
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"ros2 {' '.join(args)} failed: {detail}")
        return result

    def _run_param(self, verb: str, rest: list[str], *, timeout: float = 20.0,
                   check: bool = False) -> subprocess.CompletedProcess:
        """Run `ros2 param <verb> ...`. In sim, bypass the iceoryx-poisoned CLI daemon."""
        flags = ["--no-daemon", "--spin-time", "5"] if self.sim else []
        return self._run_ros2(["param", verb, *flags, *rest],
                              timeout=timeout, check=check, no_iceoryx=self.sim)

    def _list_controllers(self, timeout: float = 10.0):
        """Return {controller_name: state} from controller_manager, or None if unreachable.

        Queries controller_manager (`ros2 control list_controllers`) rather than the
        controller's own param node: a loaded controller and a never-loaded one are
        indistinguishable from `ros2 param list`, but controller_manager is the
        authoritative source and is up as soon as mujoco_server's ros2_control plugin is.
        """
        result = self._run_ros2(["control", "list_controllers"], timeout=timeout, check=False)
        if result.returncode != 0:
            return None
        # `ros2 control list_controllers` colorizes its output with ANSI escapes; strip them
        # so the controller name (parts[0]) and state (parts[-1]) parse cleanly.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        states: dict[str, str] = {}
        for line in clean.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                states[parts[0]] = parts[-1].lower()
        return states

    def _load_controller(self, controller: str, state: str) -> None:
        """Load a controller into controller_manager at the given lifecycle state.

        The launch file no longer spawns controllers (they were removed because the
        spawners reliably lose the race against controller_manager's ~20 s warm-up).
        controller_manager already knows each controller's type from
        sim_stacking_controllers.yaml, so loading it here is the primary (idempotent) path.
        """
        self._run_ros2(
            ["control", "load_controller", controller, "--set-state", state],
            timeout=45.0, check=False)

    def _wait_for_controller_node(self, timeout: float = 60.0) -> None:
        """Make the controllers we depend on deterministically ready.

        The mujoco_ros2_control plugin creates the controller_manager node, but its service
        callbacks can take ~20 s to become responsive on a cold boot. Rather than racing with
        launch-time spawners (which timeout and kill the launch), wait here for
        controller_manager to answer, then load any missing required controller via
        controller_manager (idempotent), and finally confirm the param interface is
        addressable so a subsequent `ros2 param set` of the gains will not race.
        """
        deadline = time.time() + timeout
        last = "controller_manager not reachable"
        # 1. Wait for controller_manager to answer, then ensure required controllers loaded.
        while time.time() < deadline:
            states = self._list_controllers()
            if states is not None:
                if states.get("joint_state_broadcaster") != "active":
                    # The broadcaster is often loaded-but-'unconfigured' (the launch loads it
                    # but its spawner loses the controller_manager warm-up race). Two facts make
                    # a single set-state active insufficient: load_controller --set-state is a
                    # no-op once the controller is already loaded, and you CANNOT activate
                    # directly from 'unconfigured'. So load it only if truly missing, then step
                    # the lifecycle explicitly: configure (-> inactive) then activate. Both
                    # steps are idempotent for the already-inactive case. (The activate CLI may
                    # report a 10 s timeout while the switch still succeeds — check=False, and
                    # the loop's ready-check re-reads the authoritative state next iteration.)
                    if "joint_state_broadcaster" not in states:
                        self._load_controller("joint_state_broadcaster", "active")
                    if (self._list_controllers() or {}).get("joint_state_broadcaster") != "active":
                        self._set_named_controller_state("joint_state_broadcaster", "inactive")
                        self._set_named_controller_state("joint_state_broadcaster", "active")
                if self.controller_name not in states:
                    self._load_controller(self.controller_name, "inactive")
                # reset() activates move_to_start when move_to_start_on_reset; load it
                # here so it is ready when reset() needs to switch to it.
                if (self.move_to_start_on_reset
                        and self.move_to_start_controller not in states):
                    self._load_controller(self.move_to_start_controller, "inactive")
                states = self._list_controllers() or {}
                ready = (self.controller_name in states
                         and states.get("joint_state_broadcaster") == "active"
                         and (not self.move_to_start_on_reset
                              or self.move_to_start_controller in states))
                if ready:
                    break
            last = "controller_manager up but %s not loaded" % self.controller_name
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"controller {self.controller_name} not loadable within {timeout:.0f}s: {last}")
        # 2. Confirm the controller's parameter interface is addressable, in-process over
        #    the backend's existing DDS session. A daemonless `ros2 param list` here costs
        #    ~15 s (full discovery per CLI call); waiting on the param service is instant.
        from rcl_interfaces.srv import ListParameters
        probe = self._node.create_client(
            ListParameters, f"/{self.controller_name}/list_parameters")
        if probe.wait_for_service(timeout_sec=max(1.0, deadline - time.time())):
            return
        raise RuntimeError(
            f"controller {self.controller_name} loaded but param node not addressable "
            f"within {timeout:.0f}s")

    def _apply_controller_gains(self, gains: dict[str, float]) -> None:
        """Apply the sim-validated stacking gains to the loaded controller.

        Sets all gains in one in-process call to /<controller>/set_parameters over the
        backend's own (already-discovered) node, NOT `ros2 param set` subprocesses. Each
        daemonless CLI call pays ~10 s of DDS discovery, so 7 gains cost ~75 s of startup;
        the in-process batch reuses the live DDS session and completes in <1 s.
        """
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        client = self._node.create_client(
            SetParameters, f"/{self.controller_name}/set_parameters")
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                f"{self.controller_name}/set_parameters unavailable; cannot apply gains")
        req = SetParameters.Request()
        for name, value in gains.items():
            req.parameters.append(Parameter(
                name=name,
                value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                     double_value=float(value))))
        result = self._call(client, req, timeout=10.0)
        failed = [n for n, r in zip(gains, result.results) if not r.successful]
        if failed:
            raise RuntimeError(f"failed to set controller gains: {failed}")

    def _activate_controller(self) -> None:
        """Activate so the controller captures the current EE pose as its desired pose."""
        if not self.activate_controller_on_reset or self._controller_activated_by_backend:
            return
        result = self._run_ros2(
            ["control", "set_controller_state", self.controller_name, "active"],
            timeout=10.0,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 and "active" not in output:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to activate {self.controller_name}: {detail}")
        self._controller_activated_by_backend = True

    def _reset_controller_target(self) -> None:
        """Discard a stale equilibrium target before re-homing.

        Real: re-publish current EE pose (no mode switch — avoids libfranka reflex on Effort controller).
        Sim: cycle inactive→active (clean, no firmware constraint). See docs/bringup-reference.md.
        """
        import rclpy as _rclpy
        if not _rclpy.ok():
            # rclpy was shut down (Ctrl-C SIGINT handler in a previous episode calls
            # rclpy.shutdown()). The upcoming _home(activate_controller=True) will
            # re-activate the controller which captures the current EE as equilibrium,
            # so skipping the re-seed here is harmless — it just means the first ramp
            # step may start from a stale target rather than the current pose.
            return
        states = self._list_controllers() or {}
        if states.get(self.controller_name) != "active":
            return
        if self.sim:
            self._set_controller_state("inactive", check=False)
            return
        self._ee_curr_pose = None
        self._spin_once_for_ee()
        cur = self.get_ee_position()
        quat = (self._ee_curr_quat.copy() if self._ee_curr_quat is not None
                else np.asarray(self.down_quat, dtype=np.float32))
        self._publish_pose(cur, quat)

    def _set_controller_state(self, state: str, *, check: bool = False) -> None:
        self._set_named_controller_state(self.controller_name, state, check=check)

    def _set_named_controller_state(self, controller: str, state: str,
                                    *, check: bool = False) -> None:
        result = self._run_ros2(
            ["control", "set_controller_state", controller, state],
            timeout=10.0,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if check and result.returncode != 0 and state not in output:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to set {controller} {state}: {detail}")
        if controller == self.controller_name:
            self._controller_activated_by_backend = state == "active" and result.returncode == 0

    def _recover_from_errors(self) -> None:
        """Send a franka ErrorRecovery goal to clear a reflex, if a recovery server exists.

        After a motion reflex (e.g. communication_constraints_violation) the
        FrankaHardwareInterface keeps the cartesian_pose_command interfaces unavailable
        until an error-recovery is performed; without this every subsequent reset's
        controller switch is rejected. No-op (short wait) when no server is on the graph.
        """
        import rclpy as _rclpy
        if not _rclpy.ok():
            return  # rclpy was shut down by a previous Ctrl-C; can't send recovery goal
        if self._error_recovery is None:
            return
        if not self._error_recovery.wait_for_server(timeout_sec=2.0):
            return  # no recovery server (e.g. sim, or wrong action name) — nothing to do
        goal = self._ErrorRecovery.Goal()
        future = self._error_recovery.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return
        result_future = handle.get_result_async()
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=5.0)

    def _move_to_start(self) -> None:
        self._set_controller_state("inactive", check=False)
        self._set_named_controller_state(self.move_to_start_controller, "active", check=True)
        time.sleep(self.move_to_start_hold_s)
        self._set_named_controller_state(self.move_to_start_controller, "inactive", check=False)

    # sim reports poses in the `world` frame; the robot base (panda_link0) is at the world
    # origin but rotated 180 deg about z (verified: link0 quat xyzw = [0,0,1,0]). The whole
    # MILE env works in the base frame (matches the equilibrium-pose command frame and the
    # scripted policy's +x-forward convention), so transform every read into base and every
    # body-set back into world. The transform is its own inverse (180 about z).
    @staticmethod
    def _flip_xy_pos(p):
        return np.array([-p[0], -p[1], p[2]], dtype=np.float32)

    @staticmethod
    def _world_quat_to_base(q):  # xyzw world -> base, premultiply conj([0,0,1,0])
        x, y, z, w = q
        return np.array([y, -x, -w, z], dtype=np.float32)

    def get_body_pose(self, name: str) -> np.ndarray:
        """Return base-frame [x,y,z,qx,qy,qz,qw] for a sim body (via get_body_state)."""
        from mujoco_ros_msgs.srv import GetBodyState

        req = GetBodyState.Request()
        req.name = name
        res = self._call(self._get_body, req)
        if res is None or not res.success:
            raise RuntimeError(f"get_body_state failed for {name!r}")
        p = res.state.pose.pose.position
        o = res.state.pose.pose.orientation
        pos = self._flip_xy_pos([p.x, p.y, p.z])
        quat = self._world_quat_to_base([o.x, o.y, o.z, o.w])
        return np.concatenate([pos, quat]).astype(np.float32)

    def _unpause(self, attempts: int = 5) -> None:
        """Unpause the sim, verifying it took (clock advances) instead of fire-and-forget."""
        preq = self._SetPause.Request()
        preq.paused = False
        for _ in range(attempts):
            res = self._call(self._set_pause, preq)
            if res is not None and getattr(res, "success", True):
                return
            time.sleep(0.5)
        raise RuntimeError("failed to unpause sim via /set_pause (clock would stay frozen)")

    # --- RobotBackend ----------------------------------------------------------
    def reset(self, np_random: np.random.Generator) -> None:
        # Clear any FR3 reflex first: a tripped reflex (e.g. from the previous episode)
        # leaves the command interfaces unavailable, so the controller switch below would
        # be rejected. Recovery restores them before we deactivate/re-home.
        if not self.sim:
            self._recover_from_errors()
        if self.reset_controller_target_on_reset:
            self._reset_controller_target()

        # Unpause the (paused-on-boot) sim. The sim ignores SetPause until its ros2_control
        # plugin is fully up, so verify the response and retry rather than silently leaving
        # the clock frozen (a frozen clock means step() commands never take effect).
        # Real hardware has no pause concept — skip.
        if self.sim:
            self._unpause()

        self._gripper_closed = True  # force an open command even if our proxy is stale
        self.open_gripper_blocking()  # verify + retry: the real gripper drops some opens
        if self.move_to_start_on_reset:
            self._move_to_start()

        if self.randomize_on_reset:
            c = self.config
            lo = c.workspace_low[:2] + c.reset_margin
            hi = c.workspace_high[:2] - c.reset_margin
            while True:
                bottom_xy = np_random.uniform(lo, hi)
                top_xy = np_random.uniform(lo, hi)
                if np.linalg.norm(bottom_xy - top_xy) >= c.reset_min_separation:
                    break
            rest_z = c.table_z + c.cube_size / 2.0
            self._set_body_pose_verified("bottom_cube", [*bottom_xy, rest_z])
            self._set_body_pose_verified("top_cube", [*top_xy, rest_z])
        self._home(activate_controller=True)

    def _home(self, activate_controller: bool = False) -> float:
        """Drive the arm to a consistent reachable start pose and settle to a tolerance.

        The multipanda custom Cartesian controller sets its desired pose to the current EE
        pose in `on_activate()`. Cycling inactive->active is therefore the stack-native way
        to discard a stale equilibrium target. After activation, command home through the
        controller; do not use MuJoCo `/reset` for the arm in ROS 2, because the generic
        initial-joint loader is NYI and the ros2_control plugin reset is a no-op.

        Settling is convergence-based, not a fixed sleep: keep commanding home until the EE
        is within `home_settle_tol` of it, or `home_settle_timeout_s` elapses. This adapts to
        whatever gains are active -- a stiff (sim) controller exits in ~1s, a soft one uses
        more of the budget -- instead of a magic hold time that under-settled the soft case
        (the multipanda impedance controller takes ~6s to converge under SIM_STACKING_GAINS).
        Returns the final home error (m) so callers can warn if it never converged.
        """
        c = self.config
        home = np.array([0.45, 0.0, c.table_z + 0.33], dtype=np.float32)
        if activate_controller:
            self._activate_controller()

        start = self.get_ee_position()
        # Capture the current EE orientation so we can ramp orientation too (get_ee_position
        # above already forced a fresh cartesian_pos_curr read, populating _ee_curr_quat).
        start_quat = (self._ee_curr_quat.copy() if self._ee_curr_quat is not None
                      else np.asarray(self.down_quat, dtype=np.float32))
        target_quat = np.asarray(self.down_quat, dtype=np.float32)
        if float(np.dot(start_quat, target_quat)) < 0.0:
            target_quat = -target_quat  # shortest-arc interpolation
        # Ramp the equilibrium pose from the current EE to home in small position AND
        # orientation increments for BOTH controllers, pacing each increment one tick apart
        # so cartesian_pose_target_controller's smoothstep finishes (returns to v=0) before
        # the next setpoint -> the commanded velocity stays continuous (same first principle
        # as set_equilibrium_pose). Size the number of increments by whichever is larger:
        # position hops <= setpoint_substep_m, or orientation hops <= ~5 deg.
        dist = float(np.linalg.norm(home - start))
        ang = 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(start_quat, target_quat))))))
        n = max(5,
                int(np.ceil(dist / max(self.setpoint_substep_m, 1e-4))),
                int(np.ceil(ang / np.radians(5.0))))
        settle = max(0.03, self.env_step_period_s)  # must be >= controller target_duration_s
        for i in range(1, n + 1):
            alpha = float(i) / n
            target = (1.0 - alpha) * start + alpha * home
            quat = (1.0 - alpha) * start_quat + alpha * target_quat
            quat = quat / (np.linalg.norm(quat) + 1e-9)
            self._publish_pose(target, quat)
            self._rclpy.spin_once(self._node, timeout_sec=0.02)
            time.sleep(settle)

        deadline = time.time() + self.home_settle_timeout_s
        err = float("inf")
        while time.time() < deadline:
            self._publish_pose(home, self.down_quat)
            self._ee_curr_pose = None          # force a fresh read for the convergence check
            self._spin_once_for_ee()
            time.sleep(0.03)
            if self._ee_curr_pose is not None:
                err = float(np.linalg.norm(self._ee_curr_pose - home))
                if err < self.home_settle_tol:
                    break
        if err >= self.home_settle_tol:
            print(f"[MultipandaRosBackend] home did not settle within "
                  f"{self.home_settle_timeout_s:.1f}s: error {err*100:.1f}cm "
                  f"(tol {self.home_settle_tol*100:.1f}cm)")
        return err

    def _set_body_pose(self, name: str, position) -> None:
        """position is base-frame; the service sets in world, so flip x,y back to world."""
        from mujoco_ros_msgs.msg import BodyState

        world = self._flip_xy_pos(position)
        req = self._SetBodyState.Request()
        bs = BodyState()
        bs.name = name
        bs.pose.header.frame_id = "world"
        bs.pose.pose.position.x = float(world[0])
        bs.pose.pose.position.y = float(world[1])
        bs.pose.pose.position.z = float(world[2])
        bs.pose.pose.orientation.w = 1.0
        req.state = bs
        req.set_pose = True
        req.set_twist = False
        req.set_mass = False
        req.reset_qpos = False
        res = self._call(self._set_body, req)
        if res is not None and not getattr(res, "success", True):
            raise RuntimeError(
                f"SetBodyState for {name!r} to world={world.tolist()} failed: "
                f"{getattr(res, 'status_message', 'unknown error')}"
            )

    def _set_body_pose_verified(self, name: str, position,
                                tol: float = 0.015,
                                attempts: int = 5) -> None:
        """Set a body pose and verify get_body_state has converged before reset returns."""
        target = np.asarray(position, dtype=np.float32)
        last = None
        for _ in range(attempts):
            self._set_body_pose(name, target)
            time.sleep(0.2)
            last = self.get_body_pose(name)[:3]
            if float(np.linalg.norm(last - target)) < tol:
                return
        raise RuntimeError(
            f"failed to reset {name} near {target.tolist()}; last pose was "
            f"{None if last is None else last.tolist()}"
        )

    def _publish_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        msg = self._PoseStamped()
        msg.header.frame_id = self.base_frame  # = panda_link0
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self._pub.publish(msg)

    def set_equilibrium_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        target = np.asarray(position, dtype=np.float32)
        deadline = time.time() + self.env_step_period_s

        if self.controller_name == "cartesian_pose_target_controller":
            # Smoothstep controller: one small setpoint per tick keeps velocity continuous.
            # See docs/bringup-reference.md for the substep-clamping rationale.
            start = self.get_ee_position()
            delta = target - start
            dist = float(np.linalg.norm(delta))
            if dist > self.setpoint_substep_m:
                target = start + delta * (self.setpoint_substep_m / dist)
            self._ee_curr_pose = None  # invalidate so next get_ee_position() waits for fresh data
            self._publish_pose(target, orientation)
            while time.time() < deadline:
                self._rclpy.spin_once(
                    self._node, timeout_sec=min(0.01, deadline - time.time()))
            return

        # Keep the policy/env contract at 10 Hz: one policy action maps to one Cartesian
        # target. The Cartesian controller tracks and holds that setpoint internally at its
        # own control frequency until the next policy tick.
        self._ee_curr_pose = None  # invalidate so next get_ee_position() waits for fresh data
        self._publish_pose(target, orientation)
        while time.time() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=min(0.01, deadline - time.time()))

    def _set_gripper(self, command: float, wait_result: bool = False,
                     force: bool = False) -> Optional[bool]:
        """Drive the gripper. Returns True/False (action reported success) when
        ``wait_result`` is set, else None.

        CLOSE uses the Grasp action (clamp inward with force to hold the cube). OPEN uses
        the Move action — Grasp cannot open: it only clamps inward and reports success even
        when it ends fully closed, so ``grasp(0.08)`` physically *closes* the real gripper
        (observed: ``open -> success=True`` yet the fingers shut). Move positions the
        fingers to a width with no grasp force and moves outward to open. Sim gripper nodes
        without a Move server fall back to Grasp (sim honors the commanded width)."""
        want_closed = command > 0
        c = self.config
        if want_closed == self._gripper_closed and not force:
            return True  # no state change; already where we want to be
        self._gripper_closed = want_closed
        grasp_width = min(c.gripper_open_width, max(c.gripper_closed_width, 0.95 * c.cube_size))
        self._gripper_width = grasp_width if want_closed else c.gripper_open_width

        if not want_closed and self._move is not None and \
                self._move.wait_for_server(timeout_sec=2.0):
            goal = self._Move.Goal()
            goal.width = float(self._gripper_width)
            goal.speed = 0.1
            client, label = self._move, "open(move)"
        else:
            # Close, or open with no Move server (sim) -> Grasp. Wide epsilon so the grasp
            # reports success across the cube-width range.
            if not self._grasp.wait_for_server(timeout_sec=2.0):
                return False
            goal = self._Grasp.Goal()
            goal.width = float(self._gripper_width)
            goal.speed = 0.1
            goal.force = 80.0   # increased from 40 N — cube was slipping under light grip
            goal.epsilon.inner = 0.08
            goal.epsilon.outer = 0.08
            client = self._grasp
            label = "close(grasp)" if want_closed else "open(grasp-fallback)"

        future = client.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
        if not wait_result or not future.done():
            return None
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            print(f"[gripper] {label} REJECTED (width={goal.width:.3f})", flush=True)
            return False
        result_future = goal_handle.get_result_async()
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=4.0)
        ok = False
        if result_future.done() and result_future.result() is not None:
            res = result_future.result().result
            ok = bool(getattr(res, "success", True))
        return ok

    def open_gripper_blocking(self, attempts: int = 3) -> bool:
        """Open the gripper (Move action) and confirm it reported success, retrying if a
        goal is dropped. Reset calls this so an episode always starts with an open gripper."""
        for attempt in range(1, attempts + 1):
            # force=True so it re-sends even though _gripper_closed is already False
            if self._set_gripper(-1.0, wait_result=True, force=True):
                return True
            print(f"[gripper] reset open attempt {attempt}/{attempts} not confirmed; "
                  "retrying", flush=True)
            time.sleep(0.5)
        print("[gripper] WARNING: could not confirm gripper open on reset", flush=True)
        return False

    def set_gripper(self, command: float) -> None:
        # Wait for both close and open to complete — without waiting on close the EE
        # moves before the gripper has physically shut, causing the cube to slip.
        self._set_gripper(command, wait_result=True)

    def get_ee_position(self) -> np.ndarray:
        """Return the controller's O_T_EE position (panda_link0 frame).

        Reads from /cartesian_impedance/cartesian_pos_curr, which is already in the same
        frame as the equilibrium_pose commands. Falls back to get_body_state("panda_hand")
        minus the hand-to-flange offset if the topic has not published yet.
        """
        if self._ee_curr_pose is None:
            self._spin_once_for_ee()
        if self._ee_curr_pose is not None:
            return self._ee_curr_pose.copy()
        # Sim fallback: panda_hand is ~0.103 m above the controller EE frame. Real has no
        # get_body_state — the O_T_EE topic is the only EE source, so require it.
        if self.sim:
            return self.get_body_pose(self.ee_body)[:3] - np.array([0.0, 0.0, 0.103],
                                                                    dtype=np.float32)
        raise RuntimeError(
            f"EE pose not received on {EE_CURR_TOPIC}; required on real (no get_body_state "
            "fallback). Check the controller is publishing O_T_EE.")

    def get_gripper_width(self) -> float:
        return float(self._gripper_width)

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        # Shut down the rclpy context so the next env construction can reinitialise cleanly.
        # The init guard (rclpy.ok() check in __init__) handles the restart side.
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

"""HTC Vive teleop device for the France-lab (Inria hucebot) real-robot setup.

Talks to SteamVR directly via the `openvr` package (lazy import, so the module loads
fine on machines without SteamVR running -- e.g. the home-lab SpaceMouse setup) to poll
a hand-held Vive controller's pose and buttons. Mirrors JoystickDevice: injectable
reader, segment-toggle intervention, button-edge ghost/bounce filtering.

Unlike SpaceMouse/Joystick (which report a proportional deflection every poll), the
Vive controller reports an absolute pose. dx/dy/dz are therefore the controller's
frame-to-frame displacement while inside an intervention segment, so the operator's
hand motion maps onto the EE delta (times translation_scale), and lifting/repositioning
the controller between segments never produces a jump.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from mile_franka.teleop.base import TeleopDevice, TeleopReading


class ViveDevice(TeleopDevice):
    """Maps an HTC Vive wand controller to a 4-DoF [dx, dy, dz, gripper] action.

    Segment mode: grip_button is a toggle -- squeeze once to enter an intervention
    segment (dx/dy/dz then track the controller's own motion), squeeze again to exit.
    Outside a segment the robot runs under the policy and nothing is recorded.

    Gripper: menu_button press toggles between open (-1) and close (+1) -- the same
    button the France-lab's ROS teleop node uses for the franka gripper.

    done/discard: trackpad click signals episode end; trigger signals discard (the
    trigger is otherwise unused here -- entering a segment already resets the position
    reference, which is all the reference ROS node uses its trigger clutch for).

    Args:
        translation_scale: unitless gain applied to the controller's real-world
            frame-to-frame displacement (1.0 = 1:1 meters).
        deadband: per-axis displacement (meters) below which it is treated as zero,
            to reject tracker jitter while the hand is held still.
        clutch_button, gripper_button, done_button, discard_button: keys into the
            controller-inputs dict returned by the reader (see `_default_reader`).
        hold_confirm_s / debounce_s: ghost/bounce filters, see JoystickDevice.
        reader: zero-arg callable returning an object with `.position` (3,) and
            `.inputs` (dict of the button names above -> bool/float). Defaults to an
            openvr-backed reader over the first discovered Vive controller.
    """

    def __init__(self, translation_scale: float = 1.0, deadband: float = 0.003,
                 clutch_button: str = "grip_button", gripper_button: str = "menu_button",
                 done_button: str = "trackpad_pressed", discard_button: str = "trigger",
                 hold_confirm_s: float = 0.05, debounce_s: float = 0.3,
                 reader: Optional[Callable[[], object]] = None):
        self.translation_scale = translation_scale
        self.deadband = deadband
        self.clutch_button = clutch_button
        self.gripper_button = gripper_button
        self.done_button = done_button
        self.discard_button = discard_button
        self.hold_confirm_s = hold_confirm_s
        self.debounce_s = debounce_s
        self._in_segment: bool = False
        self._gripper_state: float = -1.0  # open
        self._last_position: Optional[np.ndarray] = None
        self._prev_buttons: Dict[str, bool] = {}
        self._press_start: Dict[str, float] = {}
        self._triggered_this_press: Dict[str, bool] = {}
        self._last_trigger: Dict[str, float] = {}
        self._reader = reader if reader is not None else self._default_reader()

    def reset(self) -> None:
        """Reset stateful control flags on episode reset."""
        self._in_segment = False
        self._last_position = None

    def sync_gripper_state(self, closed: bool) -> None:
        """Sync gripper state from the robot on segment entry (called by TeleopIntervener)."""
        self._gripper_state = 1.0 if closed else -1.0

    @staticmethod
    def _default_reader() -> Callable[[], object]:
        import openvr  # lazy: only needed with real SteamVR/hardware

        vr = openvr.init(openvr.VRApplication_Other)
        device_index = None
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
                device_index = i
                break
        if device_index is None:
            raise RuntimeError(
                "No Vive controller found -- check SteamVR is running and the "
                "controller is powered on and tracked.")

        class _Snapshot:
            __slots__ = ("position", "inputs")

        def _read() -> object:
            poses = vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
            pose = poses[device_index]
            if pose.bPoseIsValid:
                m = pose.mDeviceToAbsoluteTracking
                position = np.array([m[0][3], m[1][3], m[2][3]], dtype=np.float32)
            else:
                position = np.zeros(3, dtype=np.float32)

            _, state = vr.getControllerState(device_index)
            # ulButtonPressed bit layout per the OpenVR EVRButtonId enum: bit 1 = menu
            # (System button is bit 0, reserved by SteamVR), bit 2 = grip, bit 32 = trackpad.
            inputs = {
                "trigger": state.rAxis[1].x,
                "menu_button": bool(state.ulButtonPressed >> 1 & 1),
                "grip_button": bool(state.ulButtonPressed >> 2 & 1),
                "trackpad_pressed": bool(state.ulButtonPressed >> 32 & 1),
            }

            snap = _Snapshot()
            snap.position = position
            snap.inputs = inputs
            return snap

        return _read

    @staticmethod
    def _button(inputs: Dict, name: str) -> bool:
        return float(inputs.get(name, 0)) >= 0.5

    def _confirmed_edge(self, inputs: Dict, name: str, now: float) -> bool:
        """True once per physical press, after hold_confirm_s hold and debounce_s lockout.

        Ghost filter: a press that is released before hold_confirm_s is silently dropped.
        Bounce filter: after an accepted trigger, the button is locked for debounce_s.
        Only one trigger fires per continuous press (released->re-pressed resets the timer).
        """
        cur = self._button(inputs, name)
        prev = self._prev_buttons.get(name, False)
        self._prev_buttons[name] = cur

        if not prev and cur:
            self._press_start[name] = now
            self._triggered_this_press[name] = False

        if prev and not cur:
            self._press_start.pop(name, None)

        if not cur or self._triggered_this_press.get(name, True):
            return False

        held_s = now - self._press_start.get(name, now)
        since_last = now - self._last_trigger.get(name, -999.0)

        if held_s >= self.hold_confirm_s and since_last >= self.debounce_s:
            self._last_trigger[name] = now
            self._triggered_this_press[name] = True
            return True

        return False

    def read(self) -> TeleopReading:
        now = time.monotonic()
        s = self._reader()
        position = np.asarray(getattr(s, "position", np.zeros(3)), dtype=np.float32)
        inputs = getattr(s, "inputs", {})

        clutch_edge = self._confirmed_edge(inputs, self.clutch_button, now)
        gripper_edge = self._confirmed_edge(inputs, self.gripper_button, now)
        done = self._confirmed_edge(inputs, self.done_button, now)
        discard = self._confirmed_edge(inputs, self.discard_button, now)

        if discard:
            self._in_segment = False
            self._last_position = None

        if clutch_edge:
            self._in_segment = not self._in_segment
            # Reset the reference on entry so the first frame of a segment never jumps.
            self._last_position = position.copy() if self._in_segment else None

        intervene = self._in_segment
        if intervene:
            delta = (position - self._last_position) * self.translation_scale
            delta[np.abs(delta) < self.deadband] = 0.0
            self._last_position = position.copy()
        else:
            delta = np.zeros(3, dtype=np.float32)

        if gripper_edge:
            self._gripper_state = 1.0 if self._gripper_state < 0 else -1.0

        action = np.array([delta[0], delta[1], delta[2], self._gripper_state], dtype=np.float32)
        return TeleopReading(action=action, intervene=intervene, done=done, discard=discard)

    def close(self) -> None:
        try:
            import openvr

            openvr.shutdown()
        except Exception:
            pass

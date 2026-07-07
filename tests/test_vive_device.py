import numpy as np
from mile_franka.teleop.vive import ViveDevice


def _dev(frames):
    """ViveDevice whose reader yields successive (position, inputs) snapshots.

    frames: sequence of (position_xyz, inputs_dict) tuples.
    """
    frames = list(frames)
    state = {"i": 0}

    def reader():
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1
        position, inputs = frames[i]

        class S:
            pass
        s = S()
        s.position = np.asarray(position, dtype=np.float32)
        s.inputs = inputs
        return s

    return ViveDevice(reader=reader, hold_confirm_s=0.0, debounce_s=0.0)


def _no_buttons():
    return {"trigger": 0.0, "menu_button": False, "grip_button": False,
            "trackpad_pressed": False}


def test_idle_produces_zero_action_and_no_intervene():
    dev = _dev([(np.zeros(3), _no_buttons())])
    r = dev.read()
    assert r.intervene is False
    assert np.allclose(r.action[:3], 0.0)


def test_grip_toggles_intervention_segment():
    b_on = {**_no_buttons(), "grip_button": True}
    b_off = _no_buttons()
    frames = [(np.zeros(3), f) for f in [b_off, b_on, b_off, b_on, b_off]]
    dev = _dev(frames)
    assert dev.read().intervene is False    # idle
    assert dev.read().intervene is True     # grip -> enter segment
    assert dev.read().intervene is True     # still in segment
    assert dev.read().intervene is False    # grip again -> exit segment
    assert dev.read().intervene is False


def test_segment_entry_has_zero_delta_then_tracks_motion():
    b_on = {**_no_buttons(), "grip_button": True}
    b_off = _no_buttons()
    frames = [
        (np.array([0.0, 0.0, 0.0]), b_on),   # enter segment at p0
        (np.array([0.1, 0.0, 0.0]), b_off),  # moved +0.1 in x, still in segment
        (np.array([0.1, 0.05, 0.0]), b_off),  # moved +0.05 in y
    ]
    dev = _dev(frames)
    r0 = dev.read()
    assert np.allclose(r0.action[:3], 0.0)   # no jump on entry
    r1 = dev.read()
    assert abs(r1.action[0] - 0.1) < 1e-6
    assert abs(r1.action[1] - 0.0) < 1e-6
    r2 = dev.read()
    assert abs(r2.action[0] - 0.0) < 1e-6
    assert abs(r2.action[1] - 0.05) < 1e-6


def test_exiting_segment_resets_reference_no_jump_on_reentry():
    # grip_button is a toggle: each True frame is a fresh press (rising edge), so an
    # intervening False frame is required between presses to re-arm the edge detector.
    b_on = {**_no_buttons(), "grip_button": True}
    b_off = _no_buttons()
    frames = [
        (np.array([0.0, 0.0, 0.0]), b_on),    # press: enter segment at x=0
        (np.array([0.2, 0.0, 0.0]), b_off),   # still in segment, tracking motion
        (np.array([0.2, 0.0, 0.0]), b_on),    # press: exit segment at x=0.2
        (np.array([0.5, 0.0, 0.0]), b_off),   # out of segment; free to drift to x=0.5
        (np.array([0.5, 0.0, 0.0]), b_on),    # press: re-enter at x=0.5 (far from x=0.2)
        (np.array([0.55, 0.0, 0.0]), b_off),  # move +0.05
    ]
    dev = _dev(frames)
    dev.read()  # enter
    dev.read()  # track
    dev.read()  # exit
    dev.read()  # drift while out
    r_reentry = dev.read()
    assert np.allclose(r_reentry.action[:3], 0.0)  # no jump despite absolute-position gap
    r_move = dev.read()
    assert abs(r_move.action[0] - 0.05) < 1e-6


def test_deadband_zeros_small_jitter():
    b_on = {**_no_buttons(), "grip_button": True}
    b_off = _no_buttons()
    frames = [
        (np.array([0.0, 0.0, 0.0]), b_on),
        (np.array([0.001, 0.0, 0.0]), b_off),  # 1mm jitter, below default 3mm deadband
    ]
    dev = _dev(frames)
    dev.read()
    r = dev.read()
    assert r.action[0] == 0.0


def test_menu_button_toggles_gripper():
    b_menu = {**_no_buttons(), "menu_button": True}
    b_off = _no_buttons()
    dev = _dev([(np.zeros(3), f) for f in [b_off, b_menu, b_off, b_menu, b_off]])
    assert dev.read().action[3] == -1.0   # starts open
    assert dev.read().action[3] == 1.0    # menu -> close
    assert dev.read().action[3] == 1.0    # held value persists
    assert dev.read().action[3] == -1.0   # menu -> open


def test_trackpad_and_trigger_signal_done_and_discard():
    b_done = {**_no_buttons(), "trackpad_pressed": True}
    b_discard = {**_no_buttons(), "trigger": 1.0}
    assert _dev([(np.zeros(3), b_done)]).read().done is True
    assert _dev([(np.zeros(3), b_discard)]).read().discard is True


def test_discard_exits_segment():
    b_on = {**_no_buttons(), "grip_button": True}
    b_discard = {**_no_buttons(), "trigger": 1.0}
    dev = _dev([(np.zeros(3), b_on), (np.zeros(3), b_discard)])
    assert dev.read().intervene is True
    r = dev.read()
    assert r.discard is True
    assert r.intervene is False


def test_reset_clears_segment_and_reference():
    b_on = {**_no_buttons(), "grip_button": True}
    dev = _dev([(np.array([1.0, 2.0, 3.0]), b_on)])
    assert dev.read().intervene is True
    dev.reset()
    assert dev._in_segment is False
    assert dev._last_position is None


def test_sync_gripper_state():
    dev = _dev([(np.zeros(3), _no_buttons())])
    dev.sync_gripper_state(closed=True)
    assert dev._gripper_state == 1.0

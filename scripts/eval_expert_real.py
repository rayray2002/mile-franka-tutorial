#!/usr/bin/env python3
"""Run the scripted expert policy on the real FR3 (Franka-Stack-Real-v0) and report
success rate.

Same purpose as ``eval_base_policy_real.py``, but drives ``ScriptedStackPolicy``
(the ground-truth-pose state machine from ``mile_franka/policies/scripted.py``,
also used for ``make collect-expert`` in sim) instead of a learned SB3 policy.
Useful for validating the AprilTag pose pipeline / calibration against a
deterministic controller, or for capturing expert rollouts on real hardware.

**Hardware prerequisites:** identical to ``eval_base_policy_real.py`` —
- hucebot multipanda_ros2 controller docker running on the FR3 control PC
- Our container on the same DDS graph (``make up`` with ``network_mode: host``)
- ``make apriltag-up`` running (RealSense/webcam + AprilTag detection + calibration static tf)
- Valid ``config/camera_calib.yaml`` from the calibration capture script
- Operator present — this script moves the real arm on every reset

**Safety:** the scripted policy has never been validated against real hardware
before this script existed — treat it as an unverified controller. The operator
stands at the robot, watches every episode, and hits the physical emergency stop
if anything goes wrong. The arm moves to a known joint home on each reset, then
to a Cartesian home above the workspace; the policy only commands small
(<=5.5 cm) delta actions. ``--max_steps`` caps each episode well below the env's
human-paced 10 000-step ceiling so a stuck state machine can't run unattended.

**Usage (in-container):**

    python3 scripts/eval_expert_real.py --episodes 5
    python3 scripts/eval_expert_real.py --episodes 3 --max_steps 400 --mediocre true
"""
import argparse
import sys

import numpy as np

from mile_franka.envs.registration import register_franka_envs, make_franka_env
from mile_franka.policies.scripted import ScriptedStackPolicy


def _confirm(prompt: str) -> None:
    """Print a prompt and wait for the operator to press Enter (or Ctrl-C to abort)."""
    print()
    print(f"  >>> {prompt}")
    print("  >>> Press Enter to continue, Ctrl-C to abort.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the scripted expert policy on the real FR3.")
    ap.add_argument("--env_name", default="Franka-Stack-Real-v0")
    ap.add_argument("--episodes", type=int, default=5,
                    help="number of evaluation episodes")
    ap.add_argument("--max_steps", type=int, default=600,
                    help="per-episode step cap (env default is a human-paced 10 000; "
                         "this keeps an unattended-looking stall bounded)")
    ap.add_argument("--mediocre", type=lambda s: s.lower() != "false", default=False,
                    help="use the mediocre (noisy) variant instead of the expert (default False = expert)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    register_franka_envs()
    import gymnasium as gym
    env = make_franka_env(gym.make(args.env_name))
    unwrapped = env.unwrapped
    max_t = min(args.max_steps, unwrapped.config.max_steps)
    policy = ScriptedStackPolicy(unwrapped.config, mediocre=args.mediocre)
    rng = np.random.default_rng(args.seed)

    print(f"Policy    : ScriptedStackPolicy(mediocre={args.mediocre})")
    print(f"Env       : {args.env_name}")
    print(f"Episodes  : {args.episodes}")
    print(f"Max steps : {max_t}")
    _confirm("ARM WILL MOVE on reset. Clear the workspace and stand clear.")

    successes, steps_used = [], []
    for ep in range(args.episodes):
        # --- Reset (moves the arm: joint home -> activate Cartesian -> Cartesian home) ---
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{args.episodes}")
        _confirm(f"Episode {ep+1}: about to RESET (arm will move to home).")

        env.reset(seed=args.seed + ep)
        policy.reset(rng)
        ee0 = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
        print("Reset complete — arm is at Cartesian home.")
        print(f"  home EE     : [{ee0[0]:.3f} {ee0[1]:.3f} {ee0[2]:.3f}]")
        print(f"  action_scale: {unwrapped.config.action_scale} m/tick")

        # --- Place cubes ---
        _confirm("Place the cubes in the workspace, then press Enter.")
        print("Running scripted policy — watch the arm and be ready on the e-stop.")

        # --- Run policy ---
        success = 0
        try:
            for t in range(max_t):
                frame = np.asarray(unwrapped.privileged_frame(), dtype=np.float32)
                action = policy.act(frame)
                ee_before = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                ee_after = np.asarray(unwrapped.backend.get_ee_position(), dtype=np.float32)
                act = np.asarray(action, dtype=np.float32).reshape(4)
                cmd_cm = float(np.linalg.norm(act[:3]) * unwrapped.config.action_scale * 100.0)
                moved_cm = float(np.linalg.norm(ee_after - ee_before) * 100.0)
                print(f"  t={t:3d} act=[{act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f} g={act[3]:+.3f}] "
                      f"cmd={cmd_cm:4.1f}cm moved={moved_cm:4.2f}cm "
                      f"ee=[{ee_after[0]:.3f} {ee_after[1]:.3f} {ee_after[2]:.3f}] "
                      f"rew={reward:+.2f} succ={info.get('success')}")
                if info.get("success") or terminated or truncated:
                    success = int(info.get("success", 0))
                    steps_used.append(t + 1)
                    break
            else:
                print(f"  episode hit --max_steps={max_t} without success or termination")
                steps_used.append(max_t)
        except KeyboardInterrupt:
            print("Episode interrupted by operator (Ctrl-C).")
            success = 0
            steps_used.append(0)

        successes.append(success)
        status = "SUCCESS" if success else "no stack"
        print(f"Episode {ep}: {status}  steps={steps_used[-1]}")

    # --- Summary ---
    rate = float(np.mean(successes))
    mean_steps = np.mean(steps_used) if steps_used else 0
    print(f"\n{'='*50}")
    print(f"REAL FR3 SCRIPTED-EXPERT EVAL")
    print(f"  Success rate : {rate:.2f} ({sum(successes)}/{args.episodes})")
    print(f"  Mean steps   : {mean_steps:.0f}")
    print(f"  Mediocre     : {args.mediocre}")

    env.close()


if __name__ == "__main__":
    main()

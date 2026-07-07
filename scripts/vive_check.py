#!/usr/bin/env python3
"""Vive controller sanity check: print live position/buttons. Blocks until Ctrl-C.

The whole Vive path is blocked until this prints nonzero deltas when the controller is
moved during a segment. Run inside the container via `make vive-check`. Use `--fake` to
exercise the read path with no SteamVR/controller attached.
"""
import argparse
import time

from mile_franka.teleop.vive import ViveDevice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="use a synthetic reader (no hardware) to smoke-test the read path")
    args = parser.parse_args()

    if args.fake:
        class Snap:
            position = [0.1, 0.0, 0.0]
            inputs = {"trigger": 0.0, "menu_button": True, "grip_button": True,
                      "trackpad_pressed": False}
        dev = ViveDevice(reader=lambda: Snap())
        r = dev.read()
        print(f"[fake] action={r.action} intervene={r.intervene} done={r.done}")
        return

    # grip=segment toggle, menu=gripper toggle, trackpad=done, trigger=discard
    dev = ViveDevice()
    print("Vive: grip=segment  menu=gripper  trackpad=done  trigger=discard  Ctrl-C to stop")
    prev_intervene = False
    try:
        while True:
            r = dev.read()
            if r.intervene != prev_intervene:
                print(f"  SEGMENT {'ON ' if r.intervene else 'OFF'}", flush=True)
                prev_intervene = r.intervene
            if r.intervene:
                dx, dy, dz = r.action[0], r.action[1], r.action[2]
                gripper = "close" if r.action[3] > 0 else "open"
                print(f"  dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} gripper={gripper}")
            if r.done:
                print("  DONE")
            if r.discard:
                print("  DISCARD")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()


if __name__ == "__main__":
    main()

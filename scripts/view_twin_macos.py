#!/usr/bin/env python3
"""macOS VNC launcher for MuJoCo-based targets (view-twin, sim-gui).

Docker Desktop on macOS puts containers in a Linux VM, so container ports are
unreachable at Mac localhost directly.  This script works around it by:
  1. Running a viewer command inside the container on Xvfb :99 (software GL works).
  2. Serving the display via x11vnc on the container's localhost.
  3. Tunnelling VNC over docker-exec stdin/stdout → Mac localhost:5900.
  4. Opening macOS Screen Sharing (vnc://localhost:5900) automatically.

Usage (via Makefile targets, or directly):
    python3 scripts/view_twin_macos.py [extra args forwarded to view_cubes_mujoco]
    python3 scripts/view_twin_macos.py --viewer-cmd 'bash scripts/sim_gui.sh' \\
                                        --pkill-pattern franka_sim_stacking
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time

PORT = 5900
CONTAINER = "mile_sim"
VNC_PASS = "mile"

# bash /dev/tcp bridge — runs inside the container via docker exec -i.
# Opens a bidirectional TCP connection to x11vnc and bridges stdin↔socket↔stdout.
# bash built-in /dev/tcp avoids nc/socat dependencies and buffering issues.
_INNER_TUNNEL = f"exec 3<>/dev/tcp/127.0.0.1/{PORT} && {{ cat <&3 & cat >&3; wait; }}"


def _start_container_services(viewer_cmd: str):
    cmd = (
        "source scripts/in_container_env.sh && "
        f"DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe {viewer_cmd} & "
        f"x11vnc -display :99 -passwd {VNC_PASS} -forever -rfbport {PORT} -quiet"
    )
    subprocess.Popen(
        ["docker", "exec", "-d", CONTAINER, "bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_x11vnc(timeout: float) -> bool:
    """Poll until x11vnc is accepting connections inside the container.

    Returns True once the container's 127.0.0.1:PORT is listening, or False if
    the timeout elapses.  We probe with a real TCP connect (nc -z) so we only
    proceed once x11vnc has actually bound the RFB port — a fixed sleep can race
    ahead of x11vnc startup (env sourcing alone takes ~3.7s) and cause macOS
    Screen Sharing to hit a dead port and report "connection failed".
    """
    deadline = time.monotonic() + timeout
    probe = f"(echo >/dev/tcp/127.0.0.1/{PORT}) 2>/dev/null"
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "bash", "-lc", probe],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"[view-twin] x11vnc is ready (after {attempt} probe(s)).")
            return True
        remaining = int(deadline - time.monotonic())
        print(f"[view-twin] Waiting for x11vnc to bind :{PORT} "
              f"(attempt {attempt}, {remaining}s left)...")
        time.sleep(1)
    return False


def _relay(recv, send, flush=None):
    try:
        while True:
            d = recv(65536)
            if not d:
                break
            send(d)
            if flush:
                flush()
    except Exception:
        pass


def _handle_vnc_client(conn):
    proc = subprocess.Popen(
        ["docker", "exec", "-i", CONTAINER, "bash", "-c", _INNER_TUNNEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    threading.Thread(
        target=_relay,
        args=(conn.recv, proc.stdin.write, proc.stdin.flush),
        daemon=True,
    ).start()
    threading.Thread(
        target=_relay,
        args=(proc.stdout.read, conn.sendall),
        daemon=True,
    ).start()


def _vnc_tunnel():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(5)
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_vnc_client, args=(conn,), daemon=True).start()
        except Exception:
            break


def _make_cleanup(pkill_pattern: str):
    def _cleanup(*_):
        try:
            for pattern in ("x11vnc", pkill_pattern):
                try:
                    subprocess.run(
                        ["docker", "exec", CONTAINER, "pkill", "-f", pattern],
                        capture_output=True,
                        timeout=5,
                        # start_new_session isolates the child so Ctrl+C SIGINT doesn't
                        # kill the docker process before it can send pkill to the container
                        start_new_session=True,
                    )
                except Exception:
                    pass
        finally:
            os._exit(0)
    return _cleanup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--viewer-cmd",
        default="python3 scripts/view_cubes_mujoco.py",
        help="Shell command to run in the container as the viewer (default: view_cubes_mujoco.py)",
    )
    ap.add_argument(
        "--pkill-pattern",
        default="view_cubes_mujoco",
        help="Pattern passed to pkill when cleaning up the viewer process",
    )
    ap.add_argument(
        "--ready-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for x11vnc to start listening before giving up (default: 30)",
    )
    args, extra = ap.parse_known_args()

    viewer_cmd = args.viewer_cmd
    if extra:
        viewer_cmd += " " + " ".join(extra)

    cleanup = _make_cleanup(args.pkill_pattern)
    # Install the SAME handler for SIGINT and SIGTERM so cleanup runs no matter
    # how we're asked to stop. Relying on Python's default KeyboardInterrupt is
    # fragile here: when launched via `make`, make puts the recipe shell and this
    # process in one foreground process group; Ctrl+C delivers SIGINT to the
    # group, but the interpreter's default handler can be starved while we're
    # blocked spawning `docker exec` subprocesses. An explicit C-level signal
    # handler fires immediately and calls os._exit(0), which is unmaskable.
    signal.signal(signal.SIGTERM, lambda *_: cleanup())
    signal.signal(signal.SIGINT, lambda *_: cleanup())

    print(f"[view-twin] Starting viewer + x11vnc in container (Xvfb :99)...")
    _start_container_services(viewer_cmd)

    print(f"[view-twin] Waiting for x11vnc to become ready "
          f"(timeout {args.ready_timeout:.0f}s)...")
    if not _wait_for_x11vnc(args.ready_timeout):
        print(
            f"[view-twin] ERROR: x11vnc did not start listening on :{PORT} "
            f"inside '{CONTAINER}' within {args.ready_timeout:.0f}s.\n"
            f"           Check that the container is running and x11vnc is "
            f"installed. Not opening Screen Sharing.",
            file=sys.stderr,
        )
        cleanup()
        return

    tunnel_thread = threading.Thread(target=_vnc_tunnel, daemon=True)
    tunnel_thread.start()

    print(f"[view-twin] VNC ready — opening Screen Sharing → vnc://localhost:{PORT}")
    subprocess.Popen(["open", f"vnc://:{VNC_PASS}@localhost:{PORT}"])

    # Block on a stop Event with short timeouts instead of `while True: sleep(1)`.
    # A long sleep() can swallow the interrupt on macOS when signals arrive while
    # C-level subprocess machinery holds the GIL; Event.wait() with a short
    # timeout keeps returning to the interpreter so the SIGINT handler above can
    # fire promptly. Pressing Enter also quits, which works through `make` even
    # if Screen Sharing has stolen keyboard focus intermittently.
    stop = threading.Event()

    def _watch_enter():
        try:
            sys.stdin.readline()
        except Exception:
            pass
        stop.set()

    threading.Thread(target=_watch_enter, daemon=True).start()
    print("[view-twin] Press Ctrl+C (or Enter) in this terminal to quit.")

    try:
        while not stop.wait(timeout=0.2):
            pass
    except KeyboardInterrupt:
        pass
    print("\n[view-twin] Stopping — cleaning up...")
    cleanup()


if __name__ == "__main__":
    main()

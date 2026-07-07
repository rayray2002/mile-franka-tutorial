#!/usr/bin/env python3
"""MJPEG-stream wrapper for view_cubes_mujoco on macOS Docker Desktop.

Renders the digital twin to the container's internal Xvfb (:99) using software
OpenGL (which works fine there), grabs the screen with ffmpeg x11grab, and serves
an MJPEG stream at http://localhost:8086 so you can view it in any browser.

Usage (via `make view-twin` on macOS, or manually inside the container):
    python3 scripts/view_twin_stream.py [args forwarded to view_cubes_mujoco.py]
    TWIN_STREAM_PORT=9000 python3 scripts/view_twin_stream.py
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time

PORT = int(os.environ.get("TWIN_STREAM_PORT", "8086"))
XVFB_DISPLAY = ":99"
RATE = int(os.environ.get("TWIN_FPS", "20"))
GRAB_SIZE = "1920x1080"


def _start_viewer(extra_args):
    env = os.environ.copy()
    env["DISPLAY"] = XVFB_DISPLAY
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    return subprocess.Popen(
        [sys.executable, "scripts/view_cubes_mujoco.py"] + extra_args,
        env=env,
    )


def _start_ffmpeg():
    return subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "x11grab",
            "-r", str(RATE),
            "-video_size", GRAB_SIZE,
            "-i", f"{XVFB_DISPLAY}.0",
            "-vf", "format=yuvj420p",
            "-vcodec", "mjpeg", "-q:v", "4",
            "-f", "image2pipe", "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )


def _frame_reader(ffmpeg_proc, latest_frame, lock):
    """Read JPEG frames from ffmpeg stdout and cache the most recent one."""
    buf = b""
    raw = ffmpeg_proc.stdout
    while True:
        chunk = raw.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            s = buf.find(b"\xff\xd8")
            if s == -1:
                break
            e = buf.find(b"\xff\xd9", s + 2)
            if e == -1:
                break
            with lock:
                latest_frame[0] = buf[s:e + 2]
            buf = buf[e + 2:]


def _handle_client(conn, latest_frame, lock):
    try:
        conn.settimeout(5.0)
        conn.recv(4096)  # consume the HTTP GET request
        conn.settimeout(None)
        conn.sendall(
            b"HTTP/1.0 200 OK\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        )
        while True:
            with lock:
                frame = latest_frame[0]
            if frame:
                msg = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
                conn.sendall(msg)
            time.sleep(1.0 / RATE)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve(ffmpeg_proc):
    latest_frame = [None]
    lock = threading.Lock()

    threading.Thread(
        target=_frame_reader, args=(ffmpeg_proc, latest_frame, lock), daemon=True
    ).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(10)
    srv.settimeout(1.0)

    print(f"\n  [view-twin] Open in browser → http://localhost:{PORT}\n", flush=True)

    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if ffmpeg_proc.poll() is not None:
                break
            continue
        threading.Thread(
            target=_handle_client, args=(conn, latest_frame, lock), daemon=True
        ).start()


def main():
    extra_args = sys.argv[1:]

    viewer = _start_viewer(extra_args)
    time.sleep(3)  # wait for GLFW window to open on Xvfb

    ffmpeg = _start_ffmpeg()

    procs = [viewer, ffmpeg]

    def _cleanup(sig, frame):
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        _serve(ffmpeg)
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

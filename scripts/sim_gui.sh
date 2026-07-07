#!/usr/bin/env bash
# Launch the multipanda MuJoCo stacking sim with a LIVE GLFW window on the host display.
# Mirror of sim_up.sh but: no Xvfb, host DISPLAY, hardware GL (no llvmpipe override).
# Host prerequisite (once per login): `xhost +local:root`.
set -e
source /home/user/mile-code/scripts/in_container_env.sh

# Live window via X11. On Linux with a real host display, unset software-GL overrides so the
# NVIDIA stack is used.  On macOS Docker (Xvfb :99), keep software GL so GLFW can open a window.
: "${DISPLAY:?DISPLAY must be set (pass the host display via make sim-gui)}"
if [ "${DISPLAY}" = ":99" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
else
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
fi

# 0. Kill any prior sim launch so re-runs don't leave duplicate ROS2 nodes.
pkill -f 'franka_sim_stacking' 2>/dev/null || true
sleep 2

# 1. Inject the cube scene beside franka_description's panda.xml (relative includes need this).
FD=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('franka_description'))")
DEST="$FD/mujoco/franka"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_scene.xml   "$DEST/"
cp /home/user/mile-code/mile_franka/assets/mujoco/stacking_objects.xml "$DEST/"
SCENE="$DEST/stacking_scene.xml"

# 2. (No Xvfb — we render to the host display.)

# 3. Launch the wrapper (cubes + inactive move-to-start / cartesian-impedance controllers).
exec ros2 launch /home/user/mile-code/mile_franka/launch/franka_sim_stacking.launch.py \
    scene_path:="$SCENE"

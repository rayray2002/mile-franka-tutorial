#!/usr/bin/env bash
# Common in-container environment for the MILE/multipanda sim. `source` me before any verb.
# `bash -lc` does NOT source ~/.bashrc reliably, so set the sim's runtime deps explicitly
# (see the franka-sim-verified-bringup note).
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/user/Libraries/libfranka/lib:/home/user/Libraries/mujoco/lib"
source /opt/ros/humble/setup.bash
source /home/user/humble_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CycloneDDS iceoryx shared-memory transport (see config/cyclonedds.xml).
# iox-roudi must be running before any ROS node starts.
# docker-compose boots it, but docker exec sessions need a guard.
# On macOS Docker Desktop, POSIX ACLs on /dev/shm are unsupported — iox-roudi crashes
# immediately (MEPOO__SEGMENT_COULD_NOT_APPLY_POSIX_RIGHTS_TO_SHARED_MEMORY).  We detect
# this and fall back to UDP-only CycloneDDS (no shared memory).
_CYCLONEDDS_URI=file:///home/user/mile-code/config/cyclonedds.xml
if ! pgrep -x iox-roudi > /dev/null 2>&1; then
    LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
        /opt/ros/humble/bin/iox-roudi -c /home/user/mile-code/config/roudi_config.toml \
        > /tmp/iox-roudi.log 2>&1 &
    sleep 1  # let RouDi init its shared-memory segments
    if ! pgrep -x iox-roudi > /dev/null 2>&1; then
        echo "[in_container_env] iox-roudi failed (no POSIX ACL support — macOS Docker?); falling back to UDP DDS" >&2
        _CYCLONEDDS_URI=file:///home/user/mile-code/config/cyclonedds_no_shm.xml
    fi
fi
export CYCLONEDDS_URI=$_CYCLONEDDS_URI
export DISPLAY="${DISPLAY:-:99}"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
cd /home/user/mile-code

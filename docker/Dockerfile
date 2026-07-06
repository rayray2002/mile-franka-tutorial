# Single MILE-on-multipanda image: hucebot base + the pinned MILE training stack + MetaWorld.
# Python mujoco is 2.3.7 (MetaWorld-v2 native); the Franka sim uses mujoco_ros (C++).
FROM hucebot:franka-humble
USER root

# Headless X server (Xvfb) for MuJoCo GLFW rendering (software GL via llvmpipe) + ffmpeg for
# x11grab video capture (eval/collect/grasp-demo record the headless display). mujoco_ros' GLFW
# init needs a reachable X display even headless; without it the sim node dies on boot and the
# whole controller graph wedges. Verify both landed so the image can never ship without them.
RUN apt-get update -qq && apt-get install -y --no-install-recommends xvfb ffmpeg \
        ros-humble-realsense2-camera \
        ros-humble-usb-cam \
        ros-humble-apriltag-ros \
 && rm -rf /var/lib/apt/lists/* \
 && command -v Xvfb >/dev/null && command -v ffmpeg >/dev/null \
 || { echo "FATAL: Xvfb/ffmpeg not installed" >&2; exit 1; }

# Install the MILE layer. CPU torch wheel keeps the image lean; the MLP trains on CPU.
# (For a CUDA box, swap the index URL and add `gpus: all` in docker-compose.yml.)
COPY docker/requirements-mile.txt /tmp/requirements-mile.txt

# MetaWorld is installed from a local copy rather than the git URL. Its
# setup.py calls setup() with no arguments, relying on pyproject.toml for
# metadata. Modern pip (≥24) rejects the built wheel from a VCS URL because
# the isolated-build setuptools reports "unknown" instead of "metaworld".
# Installing from a local path with a pre-installed known-good setuptools
# and --no-build-isolation works around both issues.
ARG MW_COMMIT=c822f28f582ba1ad49eb5dcf61016566f28003ba
RUN pip install --no-cache-dir 'setuptools>=61.0.0,<75' wheel \
 && mkdir -p /tmp/metaworld-src \
 && curl -sSL https://github.com/Farama-Foundation/Metaworld/archive/${MW_COMMIT}.tar.gz \
    | tar xz -C /tmp/metaworld-src --strip-components=1 \
 && pip install --no-cache-dir --no-build-isolation /tmp/metaworld-src \
 && python3 -c "import metaworld; print('metaworld OK')" \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements-mile.txt \
 && python3 -c "import metaworld; print('metaworld OK (post-deps)')" \
 && rm -rf /tmp/metaworld-src

# Bake the repo for shipping; in lab dev a bind mount (docker-compose.yml) overlays the
# same path, so `pip install -e .` keeps working against live host edits.
COPY . /home/user/mile-code
WORKDIR /home/user/mile-code
RUN pip install --no-cache-dir -e .

# Fail the build early if the deps don't coexist.
# ROS Python packages live under /opt/ros/humble and the user workspace; source both
# setup scripts so they are on PYTHONPATH for this non-interactive RUN step.
RUN bash -c "source /opt/ros/humble/setup.bash \
 && source /home/user/humble_ws/install/setup.bash \
 && python3 -c \"import rclpy, torch, stable_baselines3, imitation, gymnasium, metaworld, mile, mile_franka; \
from mujoco_ros_msgs.srv import GetBodyState; \
assert gymnasium.__version__=='0.29.1', gymnasium.__version__; \
import mujoco; assert mujoco.__version__.startswith('2.3'), mujoco.__version__; print('deps OK')\""

CMD ["sleep", "infinity"]

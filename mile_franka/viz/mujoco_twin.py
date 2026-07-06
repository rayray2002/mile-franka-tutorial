"""Pure/isolated helpers for the live MuJoCo digital twin.

Kept free of ROS and the viewer so the non-trivial logic (quaternion order,
joint-name->qpos mapping, scene injection) is unit-testable without hardware.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import List, Sequence, Tuple

import numpy as np

from mile_franka.pose.base import Pose

_POSITION_TAG_RE = re.compile(r"<position\b[^>]*>")
_KV_ATTR_RE = re.compile(r'\s+kv="[^"]*"')


def pose_to_freejoint_qpos(pose: Pose) -> np.ndarray:
    """Pack a base-frame Pose into MuJoCo free-joint qpos order.

    Pose.orientation is (qx, qy, qz, qw); a MuJoCo free joint stores
    qpos = [x, y, z, qw, qx, qy, qz].
    """
    x, y, z = (float(v) for v in pose.position)
    qx, qy, qz, qw = (float(v) for v in pose.orientation)
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)


def base_pose_to_world_qpos(pose: Pose, base_xpos: Sequence[float],
                            base_xquat_wxyz: Sequence[float]) -> np.ndarray:
    """Transform a robot-base-frame Pose into world free-joint qpos.

    AprilTag/GT cube poses are expressed in the robot base frame (panda_link0),
    but a MuJoCo free joint stores world coordinates. The franka MJCF places
    panda_link0 with a non-identity world transform (quat (wxyz) [0,0,0,1] =
    180 deg about Z), so writing base coords straight to qpos renders the cubes
    rotated about the base -- behind the arm and mirrored. Compose the base
    body's world transform (xpos, xquat in MuJoCo wxyz order) with the pose so
    the cube lands where the real cube is relative to the arm.
    """
    from scipy.spatial.transform import Rotation

    base_xpos = np.asarray(base_xpos, dtype=np.float64).reshape(3)
    w, x, y, z = (float(v) for v in base_xquat_wxyz)
    R_world_base = Rotation.from_quat([x, y, z, w])  # scipy uses xyzw order

    p_base = np.asarray(pose.position, dtype=np.float64).reshape(3)
    world_pos = base_xpos + R_world_base.apply(p_base)

    R_tag = Rotation.from_quat(np.asarray(pose.orientation, dtype=np.float64))
    qx, qy, qz, qw = (R_world_base * R_tag).as_quat()  # xyzw
    return np.array([world_pos[0], world_pos[1], world_pos[2],
                     qw, qx, qy, qz], dtype=np.float64)


def joint_writes(model, names: Sequence[str],
                 positions: Sequence[float]) -> List[Tuple[int, float]]:
    """Map (joint name, position) pairs to (qpos address, value).

    `model` is a mujoco.MjModel. Joint names absent from the model are skipped,
    so a mismatched arm_id or namespaced names degrade gracefully (arm stays at
    rest) instead of erroring. mujoco is imported lazily so this module loads
    without it for the pure tests.
    """
    import mujoco  # twin viewer runs on mujoco 2.3.7 in the tutorial image (see Dockerfile)

    writes: List[Tuple[int, float]] = []
    for name, value in zip(names, positions):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        writes.append((int(model.jnt_qposadr[jid]), float(value)))
    return writes


def _franka_description_franka_dir() -> str:
    """Path to franka_description's mujoco/franka dir (where panda.xml + assets live).

    Lazy ament import so the module loads without ROS for the pure tests.
    """
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("franka_description"),
                        "mujoco", "franka")


def _write_2_3_7_compatible_panda(src_panda: str, dest_panda: str) -> None:
    """Copy franka_description's panda.xml with <position> actuators' `kv`
    attribute stripped, and write it to dest_panda.

    MuJoCo only added `kv` (velocity damping) to <position> actuators in 3.0;
    the twin viewer runs pip mujoco==2.3.7 (pinned for MetaWorld-v2 compat, see
    docker/requirements-mile.txt) and fails to parse newer franka_description
    builds that set it ("Schema violation: unrecognized attribute: 'kv'"). The
    real sim (mujoco_ros, C++) uses a newer MuJoCo and loads the untouched
    panda.xml directly (see scripts/sim_up.sh) -- this sanitized copy only
    feeds the Python twin viewer. <velocity> actuators keep their own `kv`
    (gain, valid since 2.3.7); only <position>'s newer damping term is dropped.
    """
    with open(src_panda) as f:
        txt = f.read()
    patched = _POSITION_TAG_RE.sub(lambda m: _KV_ATTR_RE.sub("", m.group(0)), txt)
    with open(dest_panda, "w") as f:
        f.write(patched)


def resolve_scene_path() -> str:
    """Copy the stacking scene + objects beside franka_description's panda.xml and
    return the scene path.

    The scene uses relative includes (panda.xml, meshdir="assets"), so it must sit
    in that directory at load time. scripts/sim_up.sh does this for the sim; the
    real-robot path never runs sim_up.sh, so the twin repeats the injection here.

    The twin's copy of the scene points at a version of panda.xml sanitized for
    mujoco 2.3.7 (see _write_2_3_7_compatible_panda) instead of the original, so
    the sim's copies of stacking_scene.xml/panda.xml (loaded by a different,
    newer MuJoCo build) are left untouched.
    """
    dest = _franka_description_franka_dir()
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "assets", "mujoco")
    shutil.copy(os.path.join(src_dir, "stacking_objects.xml"),
                os.path.join(dest, "stacking_objects.xml"))

    panda_compat = "panda_mujoco_2_3_7.xml"
    _write_2_3_7_compatible_panda(os.path.join(dest, "panda.xml"),
                                   os.path.join(dest, panda_compat))

    with open(os.path.join(src_dir, "stacking_scene.xml")) as f:
        scene_txt = f.read()
    scene_txt = scene_txt.replace('<include file="panda.xml"/>',
                                  f'<include file="{panda_compat}"/>')
    scene_dest = os.path.join(dest, "stacking_scene_twin.xml")
    with open(scene_dest, "w") as f:
        f.write(scene_txt)
    return scene_dest

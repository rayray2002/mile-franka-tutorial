"""AprilTag object poses from apriltag_ros via tf2.

apriltag_ros publishes a tf frame per detected tag; with the calibration extrinsics,
tf2 yields base→tag directly. cube_center_pose() offsets along tag −z (into the face,
toward the cube center; apriltag +z points out toward the camera).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from mile_franka.pose.base import Pose


def cube_center_pose(translation: Sequence[float], quat_xyzw: Sequence[float],
                     half_edge: float) -> Pose:
    """Cube-center Pose from a base-frame tag transform + half the cube edge.

    The offset is applied along the tag's -z axis (into the tag, toward the cube
    center), matching the AprilTag convention where +z points out of the tag
    toward the camera.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    center = t - half_edge * R[:, 2]
    return Pose(tuple(float(v) for v in center),
                tuple(float(v) for v in quat_xyzw))


def _compose_camera_to_base(camera_to_base: np.ndarray,
                             cam_trans: Tuple[float, ...],
                             cam_quat_xyzw: Tuple[float, ...]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Compose ``camera_to_base @ cam_T_tag`` → base-frame (translation, quat_xyzw).

    ``camera_to_base`` is the 4×4 eye-to-hand extrinsics (camera frame in base frame).
    ``cam_trans``, ``cam_quat_xyzw`` are the tag pose in the camera optical frame (from
    apriltag_ros via tf2).  Returns the tag pose in the base frame, ready for
    ``cube_center_pose()``.
    """
    t_cam = np.asarray(cam_trans, dtype=np.float64).reshape(3)
    R_cam = Rotation.from_quat(np.asarray(cam_quat_xyzw, dtype=np.float64)).as_matrix()
    cam_T_tag = np.eye(4, dtype=np.float64)
    cam_T_tag[:3, :3] = R_cam
    cam_T_tag[:3, 3] = t_cam

    base_T_tag = camera_to_base @ cam_T_tag
    base_trans = tuple(float(v) for v in base_T_tag[:3, 3])
    base_quat = tuple(float(v) for v in Rotation.from_matrix(base_T_tag[:3, :3]).as_quat())
    return base_trans, base_quat


import time
from typing import Callable, Dict, Optional, Tuple

from mile_franka.envs.fake_backend import BOTTOM_CUBE, TOP_CUBE
from mile_franka.pose.base import ObjectPoseSource

# tag36h11 ids from scripts/generate_cube_tags.py: 0 = bottom, 1 = top. apriltag_ros names
# each tag's tf frame "<family>:<id>" (configurable in config/apriltag.yaml).
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.043  # measured black-border edge of the mounted tag (m); matches config/apriltag.yaml
CUBE_TAG_IDS = {BOTTOM_CUBE: 0, TOP_CUBE: 1}
CUBE_TAG_FRAMES = {name: f"{TAG_FAMILY}:{i}" for name, i in CUBE_TAG_IDS.items()}
CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"

# A tf lookup: (base_frame, tag_frame) -> (translation[3], quat_xyzw[4][, age_sec]); raises
# if no tf yet. The optional 3rd element is the transform's age (s) from its tf stamp; the
# real tf2 lookup supplies it so cached/stale transforms are detectable. Injected test
# lookups may return just the 2-tuple, in which case the read is treated as fresh.
TfLookup = Callable[[str, str], Tuple[Tuple[float, ...], ...]]


class AprilTagPoseSource(ObjectPoseSource):
    """Cube poses (base frame) from apriltag_ros tf, via an injectable tf lookup.

    Holds the last good pose when a tag drops out (occlusion is routine during a
    stack), but tracks the *true* age of each pose from the tf stamp so callers
    can tell a held/stale pose from a fresh one. `max_age` is the freshness bound
    used by `is_stale`; it does not abort — the pose is always returned.
    """

    def __init__(self, half_edge: float, base_frame: str = "panda_link0",
                 tf_lookup: Optional[TfLookup] = None, node=None,
                 max_age: float = 0.5,
                 camera_to_base: Optional[np.ndarray] = None):
        """half_edge: half the cube edge (m). tf_lookup injectable for tests; if None a tf2
        buffer/listener is built on `node` (a live rclpy node) lazily. max_age: seconds beyond
        which a held pose is reported stale by is_stale().

        camera_to_base: optional 4×4 eye-to-hand extrinsics (camera frame in base frame).
        When provided, the pose source looks tags up from ``CAMERA_OPTICAL_FRAME`` and composes
        with this transform, bypassing the need for a camera→base static transform in tf2.
        When None, the pose source looks up from ``base_frame`` directly (requires the
        calibration static transform to be published to tf2)."""
        self.half_edge = float(half_edge)
        self.base_frame = base_frame
        self.max_age = float(max_age)
        self._tf_lookup = tf_lookup
        self._node = node
        self._camera_to_base = camera_to_base
        # name -> (pose, observed_at_wall_s): observed_at is the wall time the pose was
        # actually *seen* (read time minus the transform's age), not the read time.
        self._last: Dict[str, Tuple[Pose, float]] = {}

    def _lookup(self) -> TfLookup:
        if self._tf_lookup is not None:
            return self._tf_lookup
        self._tf_lookup = _build_tf2_lookup(self._node)
        return self._tf_lookup

    def get_pose(self, name: str) -> Pose:
        frame = CUBE_TAG_FRAMES[name]
        try:
            if self._camera_to_base is not None:
                # Look up from camera optical frame, then compose with the eye-to-hand
                # calibration.  This bypasses the need for a camera→base static transform
                # in tf2 (which can be lost to CycloneDDS+iceoryx shared-memory
                # exhaustion when the publisher starts after initial DDS discovery).
                result = self._lookup()(CAMERA_OPTICAL_FRAME, frame)
                cam_trans, cam_quat = result[0], result[1]
                age = float(result[2]) if len(result) > 2 else 0.0
                base_trans, base_quat = _compose_camera_to_base(
                    self._camera_to_base, cam_trans, cam_quat)
                pose = cube_center_pose(base_trans, base_quat, self.half_edge)
            else:
                result = self._lookup()(self.base_frame, frame)
                translation, quat = result[0], result[1]
                # Real lookup returns the transform's age; injected 2-tuples are fresh.
                age = float(result[2]) if len(result) > 2 else 0.0
                pose = cube_center_pose(translation, quat, self.half_edge)
            # Record when the pose was observed, backdating by the transform age so a
            # silently-cached tf (Time() returns the latest, up to ~10s old) ages correctly.
            self._last[name] = (pose, time.time() - max(age, 0.0))
        except LookupError:
            pass  # tag not visible this frame; fall through to cached
        if name not in self._last:
            raise RuntimeError(f"AprilTag for {name!r} ({frame}) never seen; "
                               "check the camera/apriltag node and tag visibility")
        return self._last[name][0]

    def pose_age(self, name: str) -> Optional[float]:
        entry = self._last.get(name)
        return None if entry is None else time.time() - entry[1]

    def is_stale(self, name: str) -> bool:
        age = self.pose_age(name)
        return age is None or age > self.max_age

    def last_seen(self, name: str) -> Optional[float]:
        entry = self._last.get(name)
        return None if entry is None else entry[1]


def _build_tf2_lookup(node) -> TfLookup:
    """Default tf2-backed lookup. Imported lazily so the module loads without ROS."""
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener

    buffer = Buffer(cache_time=Duration(seconds=10))
    TransformListener(buffer, node)

    # Prime the listener: give DDS discovery enough time to connect to both /tf
    # and /tf_static publishers. The /tf_static latched message (camera→base
    # calibration from apriltag-up or view-twin) needs discovery to complete
    # before the first lookup. 40 spins × 50ms = 2s.
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.05)

    def lookup(base_frame: str, tag_frame: str):
        # Brief spin so the TransformListener can deposit a pending /tf or
        # /tf_static message before we query.  When the caller already drains
        # (view_cubes_mujoco.py drain loop) this returns immediately (~2 ms);
        # for callers that don't drain it still catches the racing tf message.
        rclpy.spin_once(node, timeout_sec=0.002)
        from rclpy.time import Time
        try:
            tf = buffer.lookup_transform(base_frame, tag_frame, Time())
        except Exception as exc:  # tf2 raises LookupException/ExtrapolationException
            raise LookupError(tag_frame) from exc
        tr = tf.transform.translation
        rot = tf.transform.rotation
        # lookup_transform(..., Time()) returns the latest buffered transform even when
        # the tag is no longer visible (the buffer caches ~10s). Report its age from the
        # tf stamp so the pose source can flag a held/stale pose during inference.
        age = (node.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
        return ((tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w), age)

    return lookup

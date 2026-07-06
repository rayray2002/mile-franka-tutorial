import numpy as np
import pytest

from mile_franka.pose.base import Pose
from mile_franka.viz.mujoco_twin import (
    base_pose_to_world_qpos, pose_to_freejoint_qpos)


def test_base_pose_to_world_qpos_identity_base():
    # When the robot base sits at the world origin with identity orientation,
    # base-frame coords pass through unchanged.
    pose = Pose(position=[0.44, -0.13, 0.04], orientation=[0.0, 0.0, 0.0, 1.0])
    q = base_pose_to_world_qpos(pose, base_xpos=[0, 0, 0],
                                base_xquat_wxyz=[1, 0, 0, 0])
    np.testing.assert_allclose(q[:3], [0.44, -0.13, 0.04], atol=1e-6)


def test_base_pose_to_world_qpos_applies_base_rotation():
    # The franka MJCF places panda_link0 with quat (wxyz) [0,0,0,1] = 180 deg
    # about Z, so the base frame is rotated 180 deg relative to world. A cube in
    # front of the robot (base +x) must render in front of the rotated arm
    # (world -x), with x and y both negated.
    pose = Pose(position=[0.442, -0.129, 0.038], orientation=[0.0, 0.0, 0.0, 1.0])
    q = base_pose_to_world_qpos(pose, base_xpos=[0, 0, 0],
                                base_xquat_wxyz=[0, 0, 0, 1])
    np.testing.assert_allclose(q[:3], [-0.442, 0.129, 0.038], atol=1e-6)


def test_pose_to_freejoint_qpos_reorders_quat_to_wxyz():
    # Pose.orientation is (qx, qy, qz, qw); MuJoCo free joints store (x,y,z, qw,qx,qy,qz).
    pose = Pose(position=[0.1, 0.2, 0.3], orientation=[0.0, 0.0, 0.7071, 0.7071])
    q = pose_to_freejoint_qpos(pose)
    assert q.shape == (7,)
    np.testing.assert_allclose(q[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(q[3:], [0.7071, 0.0, 0.0, 0.7071], atol=1e-6)


def test_joint_writes_maps_known_and_skips_unknown():
    mujoco = pytest.importorskip("mujoco")
    from mile_franka.viz.mujoco_twin import joint_writes

    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.1 0.1 0.1"/>
          <body name="b2" pos="0 0 0.3">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.1 0.1 0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    writes = joint_writes(model, ["j1", "ghost", "j2"], [0.5, 9.9, -0.3])

    by_adr = {adr: val for adr, val in writes}
    a1 = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j1")])
    a2 = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j2")])
    assert by_adr[a1] == 0.5
    assert by_adr[a2] == -0.3
    assert len(writes) == 2  # unknown "ghost" skipped


def test_resolve_scene_path_copies_assets(tmp_path, monkeypatch):
    import mile_franka.viz.mujoco_twin as mt

    dest = tmp_path / "mujoco" / "franka"
    dest.mkdir(parents=True)
    # Stand in for the real franka_description location (avoids needing ament/ROS).
    monkeypatch.setattr(mt, "_franka_description_franka_dir", lambda: str(dest))
    # franka_description ships panda.xml alongside its mujoco/franka assets; stub
    # one with a <position kv=...> actuator, as newer builds emit.
    (dest / "panda.xml").write_text(
        '<mujoco model="panda">\n'
        '  <actuator>\n'
        '    <position name="panda_act_pos1" joint="panda_joint1" kp="10" kv="1"/>\n'
        '    <velocity name="panda_act_vel1" joint="panda_joint1" kv="1"/>\n'
        '  </actuator>\n'
        '</mujoco>\n'
    )

    scene = mt.resolve_scene_path()

    assert scene == str(dest / "stacking_scene_twin.xml")
    assert (dest / "stacking_scene_twin.xml").exists()
    assert (dest / "stacking_objects.xml").exists()

    # Position actuator's `kv` (unsupported pre-3.0) is stripped; kp and the
    # velocity actuator's own kv are left alone.
    compat_panda = (dest / "panda_mujoco_2_3_7.xml").read_text()
    assert 'name="panda_act_pos1" joint="panda_joint1" kp="10"/>' in compat_panda
    assert 'name="panda_act_vel1" joint="panda_joint1" kv="1"/>' in compat_panda

    scene_txt = (dest / "stacking_scene_twin.xml").read_text()
    assert '<include file="panda_mujoco_2_3_7.xml"/>' in scene_txt
    assert '<include file="panda.xml"/>' not in scene_txt

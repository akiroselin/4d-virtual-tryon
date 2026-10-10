from pathlib import Path

import torch

from aigc_fashion.bvh_motion import BVHMotion
from aigc_fashion.unified_cloth4d_scene import CapturedWalkingSE3BodyTrajectory


MOTION_PATH = Path(__file__).parents[1] / "assets" / "mocap" / "ezgitek_walk1.bvh"


def test_bvh_walk_has_ordered_channels_and_real_duration():
    motion = BVHMotion.from_file(MOTION_PATH)

    assert motion.frame_count == 292
    assert motion.frame_time_s == 0.017
    assert motion.duration_s > 4.0
    assert [joint.name for joint in motion.joints[:5]] == [
        "Hips",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftUpLeg",
    ]
    assert motion.channel("Hips", "Zposition")[-1] > motion.channel("Hips", "Zposition")[0]


def test_captured_walk_retargets_root_and_limb_signals():
    trajectory = CapturedWalkingSE3BodyTrajectory(
        torch.zeros(3),
        str(MOTION_PATH),
        step_duration_s=2.0,
        steps=2,
        step_length_m=0.06,
    )
    reference = torch.tensor(
        [[0.0, -0.10, 0.10], [0.10, -0.30, 0.10]], dtype=torch.float32
    )

    start, start_translation = trajectory.pose(0.0, reference)
    end, end_translation = trajectory.pose(4.0, reference)
    assert torch.allclose(start, torch.eye(3), atol=1.0e-5)
    assert torch.linalg.norm(end_translation - start_translation) > 0.10
    assert torch.linalg.norm(end[:, 2] - start[:, 2]) > 0.02
    left, right = trajectory._leg_motion_angles(1.0, reference)
    assert abs(float(left)) > 0.02 or abs(float(right)) > 0.02
    left_arm, right_arm = trajectory._arm_motion_angles(1.0, reference)
    assert abs(float(left_arm)) > 0.01 or abs(float(right_arm)) > 0.01
    assert abs(float(left_arm + right_arm)) < 1.0e-6
    sampled_arm = torch.stack(
        [
            trajectory._arm_motion_angles(float(time), reference)[0]
            for time in torch.linspace(0.0, 4.0, 17)
        ]
    )
    assert float(sampled_arm.abs().max()) > 0.08
    assert trajectory.capture_summary()["retargeted_channels"]["arms"] == (
        "{Left,Right}Arm.Xrotation"
    )


def test_captured_body_rigid_branches_preserve_segment_length():
    trajectory = CapturedWalkingSE3BodyTrajectory(
        torch.zeros(3),
        str(MOTION_PATH),
        step_duration_s=2.0,
        steps=2,
        step_length_m=0.06,
    )
    reference = torch.tensor(
        [[0.10, -0.35, 0.10], [0.10, -0.50, 0.10]], dtype=torch.float32
    )
    start = trajectory.transform_body_points(reference, 0.0)
    middle = trajectory.transform_body_points(reference, 1.0)
    start_length = torch.linalg.norm(start[1] - start[0])
    middle_length = torch.linalg.norm(middle[1] - middle[0])
    assert torch.isclose(start_length, middle_length, rtol=1.0e-4, atol=1.0e-5)


def test_captured_walk_has_monotone_stride_and_alternating_knee_flexion():
    trajectory = CapturedWalkingSE3BodyTrajectory(
        torch.zeros(3),
        str(MOTION_PATH),
        step_duration_s=2.0,
        steps=1,
        step_length_m=0.22,
        knee_flex_rad=torch.deg2rad(torch.tensor(38.0)).item(),
    )
    reference = torch.zeros((1, 3))

    translations = torch.stack(
        [trajectory.pose(float(time), reference)[1] for time in torch.linspace(0, 2, 9)]
    )
    assert torch.all(torch.diff(translations[:, 2]) >= -1.0e-7)
    assert torch.isclose(translations[4, 2], torch.tensor(0.11), atol=1.0e-4)
    assert torch.isclose(translations[-1, 2], torch.tensor(0.22), atol=1.0e-4)

    knee_angles = torch.stack(
        [
            torch.stack(trajectory._knee_motion_angles(float(time), reference))
            for time in torch.linspace(0, 2, 33)
        ]
    )
    assert torch.all(knee_angles >= 0.0)
    assert torch.all(knee_angles.max(dim=0).values > 0.30)

    transforms, nodes, edges, joints = trajectory.lie_message_graph(0.5, reference)
    assert transforms.shape == (8, 4, 4)
    assert nodes.shape == (8, 8)
    assert edges.shape == (2, 7)
    assert joints.shape == (7, 4)
    assert "knee flexion" in trajectory.capture_summary()["retargeted_channels"]["legs"]

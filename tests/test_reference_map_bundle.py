import torch

from aigc_fashion.hypernetwork_kirchhoff_love import (
    build_kirchhoff_love_shell_geometry,
)
from aigc_fashion.reference_map_bundle import ReferenceMapFiberBundle


def _bundle() -> tuple[ReferenceMapFiberBundle, torch.Tensor]:
    rest = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    material = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] * 2
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    geometry = build_kirchhoff_love_shell_geometry(
        material,
        faces,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        reference_positions=rest,
    )
    return ReferenceMapFiberBundle.from_shell_geometry(geometry, rest), rest


def test_rigid_rotation_changes_quaternion_not_material_metric() -> None:
    bundle, rest = _bundle()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    base = bundle.sample(rest)
    moved = bundle.sample(rest @ rotation.T + torch.tensor([2.0, -3.0, 0.4]))
    assert torch.allclose(moved.metric, base.metric, atol=1.0e-6)
    assert torch.allclose(moved.metric_change, torch.zeros_like(moved.metric_change), atol=1.0e-6)
    assert not torch.allclose(moved.orientation_quaternion, base.orientation_quaternion)
    assert torch.allclose(
        torch.linalg.vector_norm(moved.orientation_quaternion, dim=-1),
        torch.ones(len(moved.orientation_quaternion)),
        atol=1.0e-6,
    )


def test_pure_stretch_changes_metric_while_quaternion_frame_stays_fixed() -> None:
    bundle, rest = _bundle()
    stretched = rest * torch.tensor([1.2, 0.85, 1.0])
    base = bundle.sample(rest)
    changed = bundle.sample(stretched)
    assert torch.allclose(
        changed.orientation_quaternion, base.orientation_quaternion, atol=1.0e-6
    )
    assert float(changed.metric_change.abs().max()) > 0.1
    assert torch.equal(bundle.material_coordinates, bundle.material_coordinates.clone())


def test_reference_map_report_marks_material_memory_immutable() -> None:
    bundle, rest = _bundle()
    report = bundle.report(rest)
    assert report.material_map_immutable
    assert report.face_count == 2
    assert report.sheet_count == 2
    assert report.maximum_metric_change < 1.0e-6

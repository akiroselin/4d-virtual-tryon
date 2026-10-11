from types import SimpleNamespace

import torch

from aigc_fashion.anny_cloth4d_simulation import (
    AnnyCloth4DConfig,
    _scene_graph_tensors,
)
from aigc_fashion.behavioral_phase import BehavioralPhaseTrajectory
from aigc_fashion.spatiotemporal_scene_graph import (
    HierarchyAwareSpatiotemporalAttention,
)
from aigc_fashion.unified_cloth4d_scene import WalkingSE3BodyTrajectory


def _geometry() -> SimpleNamespace:
    return SimpleNamespace(
        panel_names=["front_body", "back_body"],
        panel_ids=torch.tensor([0, 0, 1, 1]),
        body_anchor_positions=torch.tensor(
            [[-0.1, 0.2, 0.0], [0.1, 0.2, 0.0], [0.0, -0.2, 0.0]]
        ),
        panel_adjacency=((0, 1), (1, 0)),
    )


def test_behavior_is_time_varying_and_attached_to_person_body_hierarchy() -> None:
    profile = BehavioralPhaseTrajectory(
        personality_ocean=(0.2, 0.1, 0.8, -0.1, 0.3),
        emotion_vad_base=(0.1, 0.2, 0.0),
        emotion_vad_amplitude=(0.2, 0.5, 0.1),
        emotion_frequency_hz=1.0,
    )
    config = AnnyCloth4DConfig(behavior_profile=profile)
    previous = torch.tensor(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, -0.2, 0.0], [0.1, -0.2, 0.0]]
    )
    current = previous + torch.tensor([0.0, 0.0, 0.01])
    scene = _scene_graph_tensors(previous, current, _geometry(), 0.25, config)
    object_count = 5
    current_root = object_count
    current_body = object_count + 1
    current_garment = object_count + 2
    current_panel = object_count + 3

    assert torch.allclose(
        scene["scalar_features"][current_root, 8:13],
        torch.tensor(profile.personality_ocean),
    )
    assert torch.allclose(
        scene["scalar_features"][current_body, 13:16],
        torch.tensor([0.3, 0.7, 0.1]),
        atol=1e-6,
    )
    edges = {tuple(edge) for edge in scene["edge_index"].T.tolist()}
    assert (current_body, current_root) in edges
    assert (current_garment, current_body) in edges
    assert (current_panel, current_garment) in edges
    assert (current_panel, current_body) in edges
    assert torch.equal(
        scene["levels"][current_root : current_panel + 1],
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )


def test_three_hop_attention_propagates_personality_to_panel_phase_context() -> None:
    torch.manual_seed(211)
    previous = torch.tensor(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, -0.2, 0.0], [0.1, -0.2, 0.0]]
    )
    current = previous.clone()
    neutral = _scene_graph_tensors(
        previous,
        current,
        _geometry(),
        0.1,
        AnnyCloth4DConfig(),
    )
    expressive = _scene_graph_tensors(
        previous,
        current,
        _geometry(),
        0.1,
        AnnyCloth4DConfig(
            behavior_profile=BehavioralPhaseTrajectory(
                personality_ocean=(0.9, 0.2, 0.8, 0.1, -0.2),
                emotion_vad_base=(0.6, 0.8, 0.5),
            )
        ),
    )
    model = HierarchyAwareSpatiotemporalAttention(16, depth=3, hidden_dim=12)

    def evaluate(scene: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(
            scene["scalar_features"],
            scene["positions"],
            scene["times"],
            scene["levels"],
            scene["edge_index"],
            scene["relation_ids"],
        ).scalar_features

    neutral_features = evaluate(neutral)
    expressive_features = evaluate(expressive)
    current_front_panel = 5 + 3
    assert not torch.allclose(
        expressive_features[current_front_panel],
        neutral_features[current_front_panel],
    )


def test_behavior_changes_body_control_port_before_cloth_contact() -> None:
    expressive_profile = BehavioralPhaseTrajectory(
        personality_ocean=(0.5, 0.1, 0.9, 0.2, -0.3),
        emotion_vad_base=(0.7, 0.8, 0.6),
    )
    neutral = WalkingSE3BodyTrajectory(torch.zeros(3), step_duration_s=0.2)
    expressive = WalkingSE3BodyTrajectory(
        torch.zeros(3),
        step_duration_s=0.2,
        behavior_profile=expressive_profile,
    )
    reference = torch.zeros(3)
    _neutral_rotation, neutral_translation = neutral.pose(0.08, reference)
    _expressive_rotation, expressive_translation = expressive.pose(0.08, reference)
    port = expressive_profile.body_control_port(0.08)

    assert 0.65 <= port.cadence_scale <= 1.45
    assert 0.70 <= port.stride_scale <= 1.40
    assert 0.70 <= port.sway_scale <= 1.40
    assert 0.60 <= port.gesture_scale <= 1.60
    assert not torch.allclose(expressive_translation, neutral_translation)
    assert expressive_translation[2] > neutral_translation[2]

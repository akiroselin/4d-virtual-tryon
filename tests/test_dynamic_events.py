from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigc_fashion.dynamic_events import (  # noqa: E402
    ActiveSide,
    DynamicTopologyEventHandler,
    EventKind,
    EventSurface,
    EventSurfaceKind,
    FoldGraphTracker,
    FoldObservation,
    IntersectionContourObservation,
    IntersectionContourTracker,
    ZeroCrossingEventDetector,
)


def test_contact_zero_crossing_is_localized_and_requests_restart() -> None:
    detector = ZeroCrossingEventDetector()
    detector.register(
        EventSurface(
            "body:v12",
            EventSurfaceKind.BODY_CONTACT,
            ActiveSide.BELOW,
        )
    )
    assert not detector.observe(0.0, {"body:v12": 0.01}).restart_required
    batch = detector.observe(0.1, {"body:v12": -0.01})
    assert batch.restart_required
    assert batch.earliest_time == pytest.approx(0.05)
    assert batch.events[0].kind is EventKind.BODY_CONTACT_ENTER
    assert batch.events[0].localized
    assert detector.is_active("body:v12")

    exit_batch = detector.observe(0.2, {"body:v12": 0.03})
    assert exit_batch.events[0].kind is EventKind.BODY_CONTACT_EXIT
    assert exit_batch.events[0].time == pytest.approx(0.125)
    assert not detector.is_active("body:v12")


def test_hysteresis_confirms_root_without_chatter() -> None:
    detector = ZeroCrossingEventDetector()
    detector.register(
        EventSurface(
            "vf:1:8",
            EventSurfaceKind.SELF_VERTEX_FACE,
            ActiveSide.BELOW,
            hysteresis=0.002,
        )
    )
    detector.observe(0.0, {"vf:1:8": 0.01})
    assert not detector.observe(0.1, {"vf:1:8": -0.001}).events
    assert not detector.observe(0.2, {"vf:1:8": 0.001}).events
    batch = detector.observe(0.3, {"vf:1:8": -0.003})
    assert len(batch.events) == 1
    assert batch.events[0].kind is EventKind.SELF_VERTEX_FACE_ENTER
    assert batch.events[0].time == pytest.approx(0.225)


def test_ccd_can_report_a_crossing_hidden_between_equal_sign_endpoints() -> None:
    detector = ZeroCrossingEventDetector()
    detector.register(
        EventSurface(
            "ee:4:9",
            EventSurfaceKind.SELF_EDGE_EDGE,
            ActiveSide.BELOW,
        )
    )
    detector.observe(0.0, {"ee:4:9": 0.01})
    batch = detector.record_external_root(
        "ee:4:9",
        time=0.04,
        active=True,
        bracket=(0.0, 0.1),
        sample_values=(0.01, 0.02),
    )
    assert batch.events[0].kind is EventKind.SELF_EDGE_EDGE_ENTER
    assert batch.events[0].payload["reason"] == "ccd_root"
    assert batch.earliest_time == pytest.approx(0.04)
    assert detector.is_active("ee:4:9")


def test_fold_graph_birth_and_death_do_not_change_material_mesh() -> None:
    tracker = FoldGraphTracker(threshold=0.7, hysteresis=0.05)
    low = FoldObservation("crease-a", "front", ("region-1", "region-2"), 0.4)
    high = FoldObservation(
        "crease-a",
        "front",
        ("region-1", "region-2"),
        0.9,
        endpoint_ids=("crease-end-1", "crease-end-2"),
        occlusion_front_region="region-2",
    )
    assert not tracker.update(0.0, [low]).events
    born = tracker.update(1.0, [high])
    assert born.events[0].kind is EventKind.FOLD_BIRTH
    assert "crease-a" in tracker.active_creases

    died = tracker.update(2.0, [])
    assert died.events[0].kind is EventKind.FOLD_DEATH
    assert not tracker.active_creases


def _contour(
    face_pairs: set[tuple[int, int]],
    *,
    contour_nodes: set[int],
    enclosed_nodes: set[int] | None = None,
    closed: bool = False,
    nested: bool = False,
    length: float = 0.1,
) -> IntersectionContourObservation:
    return IntersectionContourObservation(
        layer_ids=("shirt", "jacket"),
        face_pairs=frozenset(face_pairs),
        contour_nodes=frozenset(contour_nodes),
        enclosed_nodes=frozenset(enclosed_nodes or set()),
        length=length,
        closed=closed,
        nested=nested,
    )


def test_intersection_contours_are_matched_updated_and_retired() -> None:
    tracker = IntersectionContourTracker(match_threshold=0.2)
    first = _contour({(1, 7), (2, 8)}, contour_nodes={1, 2}, length=0.3)
    birth = tracker.update(0.0, [first])
    assert birth.events[0].kind is EventKind.INTERSECTION_CONTOUR_BIRTH
    contour_id = birth.events[0].event_id
    assert tracker.non_repulsive_nodes == frozenset({1, 2})

    moved = _contour({(2, 8), (3, 9)}, contour_nodes={2, 3}, length=0.2)
    update = tracker.update(0.1, [moved])
    assert update.events[0].kind is EventKind.INTERSECTION_CONTOUR_UPDATE
    assert update.events[0].event_id == contour_id

    death = tracker.update(0.2, [])
    assert death.events[0].kind is EventKind.INTERSECTION_CONTOUR_DEATH
    assert not tracker.active_contours


def test_contourcraft_non_repulsive_policy_drops_nested_contours() -> None:
    tracker = IntersectionContourTracker()
    outer = _contour(
        {(1, 7)},
        contour_nodes={1, 2},
        enclosed_nodes={10, 11},
        closed=True,
    )
    nested = _contour(
        {(20, 30)},
        contour_nodes={20, 21},
        enclosed_nodes={99},
        closed=True,
        nested=True,
    )
    tracker.update(0.0, [outer, nested])
    assert tracker.non_repulsive_nodes == frozenset({10, 11})


def test_facade_rejects_fold_surface_registration() -> None:
    handler = DynamicTopologyEventHandler()
    with pytest.raises(ValueError, match="FoldGraphTracker"):
        handler.register_surface(
            EventSurface(
                "fold:manual",
                EventSurfaceKind.FOLD_CONFIDENCE,
                ActiveSide.ABOVE,
                threshold=0.7,
            )
        )

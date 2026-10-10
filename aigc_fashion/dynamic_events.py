from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping


class ActiveSide(str, Enum):
    """Which side of an event surface represents an active constraint."""

    BELOW = "below"
    ABOVE = "above"


class EventSurfaceKind(str, Enum):
    BODY_CONTACT = "body_contact"
    GARMENT_CONTACT = "garment_contact"
    SELF_VERTEX_FACE = "self_vertex_face"
    SELF_EDGE_EDGE = "self_edge_edge"
    FOLD_CONFIDENCE = "fold_confidence"


class EventKind(str, Enum):
    BODY_CONTACT_ENTER = "body_contact_enter"
    BODY_CONTACT_EXIT = "body_contact_exit"
    GARMENT_CONTACT_ENTER = "garment_contact_enter"
    GARMENT_CONTACT_EXIT = "garment_contact_exit"
    SELF_VERTEX_FACE_ENTER = "self_vertex_face_enter"
    SELF_VERTEX_FACE_EXIT = "self_vertex_face_exit"
    SELF_EDGE_EDGE_ENTER = "self_edge_edge_enter"
    SELF_EDGE_EDGE_EXIT = "self_edge_edge_exit"
    FOLD_BIRTH = "fold_birth"
    FOLD_DEATH = "fold_death"
    INTERSECTION_CONTOUR_BIRTH = "intersection_contour_birth"
    INTERSECTION_CONTOUR_UPDATE = "intersection_contour_update"
    INTERSECTION_CONTOUR_DEATH = "intersection_contour_death"


_TRANSITIONS = {
    (EventSurfaceKind.BODY_CONTACT, True): EventKind.BODY_CONTACT_ENTER,
    (EventSurfaceKind.BODY_CONTACT, False): EventKind.BODY_CONTACT_EXIT,
    (EventSurfaceKind.GARMENT_CONTACT, True): EventKind.GARMENT_CONTACT_ENTER,
    (EventSurfaceKind.GARMENT_CONTACT, False): EventKind.GARMENT_CONTACT_EXIT,
    (EventSurfaceKind.SELF_VERTEX_FACE, True): EventKind.SELF_VERTEX_FACE_ENTER,
    (EventSurfaceKind.SELF_VERTEX_FACE, False): EventKind.SELF_VERTEX_FACE_EXIT,
    (EventSurfaceKind.SELF_EDGE_EDGE, True): EventKind.SELF_EDGE_EDGE_ENTER,
    (EventSurfaceKind.SELF_EDGE_EDGE, False): EventKind.SELF_EDGE_EDGE_EXIT,
    (EventSurfaceKind.FOLD_CONFIDENCE, True): EventKind.FOLD_BIRTH,
    (EventSurfaceKind.FOLD_CONFIDENCE, False): EventKind.FOLD_DEATH,
}


@dataclass(frozen=True)
class EventSurface:
    """A scalar event function sampled by the geometry or perception layer.

    Body, vertex-face, and edge-edge gaps normally use ``BELOW`` with a zero
    threshold. Fold confidence normally uses ``ABOVE`` with a learned threshold.
    Hysteresis confirms a transition away from zero while preserving the time of
    the first zero crossing.
    """

    surface_id: str
    kind: EventSurfaceKind
    active_side: ActiveSide
    threshold: float = 0.0
    hysteresis: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.surface_id:
            raise ValueError("surface_id must not be empty")
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not isfinite(self.hysteresis) or self.hysteresis < 0.0:
            raise ValueError("hysteresis must be finite and non-negative")


@dataclass(frozen=True)
class DynamicEvent:
    kind: EventKind
    event_id: str
    time: float
    bracket: tuple[float, float]
    localized: bool
    active: bool
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EventBatch:
    """Events returned to an adaptive solver at one observation boundary."""

    events: tuple[DynamicEvent, ...]

    @property
    def restart_required(self) -> bool:
        return bool(self.events)

    @property
    def earliest_time(self) -> float | None:
        if not self.events:
            return None
        return min(event.time for event in self.events)


@dataclass(frozen=True)
class _PendingTransition:
    active: bool
    time: float
    bracket: tuple[float, float]
    values: tuple[float, float]
    localized: bool


@dataclass
class _SurfaceState:
    time: float
    oriented_value: float
    raw_value: float
    active: bool
    pending: _PendingTransition | None = None


class ZeroCrossingEventDetector:
    """Stateful zero-crossing detector for adaptive ODE and NDF sampling.

    Values are oriented internally so that ``oriented_value <= 0`` always means
    active. The detector localizes a root linearly inside the sampled bracket.
    A production DOPRI5 adapter should then re-evaluate the dense interpolant and
    refine that root before accepting the event.
    """

    def __init__(self, *, emit_initial_events: bool = True) -> None:
        self.emit_initial_events = emit_initial_events
        self._surfaces: dict[str, EventSurface] = {}
        self._states: dict[str, _SurfaceState] = {}

    def register(self, surface: EventSurface) -> None:
        existing = self._surfaces.get(surface.surface_id)
        if existing is not None and existing != surface:
            raise ValueError(f"surface {surface.surface_id!r} is already registered differently")
        self._surfaces[surface.surface_id] = surface

    def is_active(self, surface_id: str) -> bool:
        state = self._states.get(surface_id)
        return bool(state and state.active)

    @property
    def active_surface_ids(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, state in self._states.items() if state.active))

    def observe(self, time: float, values: Mapping[str, float]) -> EventBatch:
        if not isfinite(time):
            raise ValueError("time must be finite")
        events: list[DynamicEvent] = []
        for surface_id in sorted(values):
            if surface_id not in self._surfaces:
                raise KeyError(f"unregistered event surface: {surface_id}")
            raw_value = float(values[surface_id])
            if not isfinite(raw_value):
                raise ValueError(f"event value for {surface_id!r} must be finite")
            event = self._observe_one(self._surfaces[surface_id], time, raw_value)
            if event is not None:
                events.append(event)
        events.sort(key=lambda event: (event.time, event.kind.value, event.event_id))
        return EventBatch(tuple(events))

    def record_external_root(
        self,
        surface_id: str,
        *,
        time: float,
        active: bool,
        bracket: tuple[float, float],
        sample_values: tuple[float, float],
        reason: str = "ccd_root",
    ) -> EventBatch:
        """Record a root found by CCD or a dense-output root solver.

        This path is required when a primitive crosses and separates inside one
        adaptive step, leaving event values with the same sign at both endpoints.
        """

        if surface_id not in self._surfaces:
            raise KeyError(f"unregistered event surface: {surface_id}")
        if not isfinite(time) or not all(isfinite(value) for value in bracket + sample_values):
            raise ValueError("external root data must be finite")
        if bracket[0] > time or time > bracket[1] or bracket[0] > bracket[1]:
            raise ValueError("external root time must lie inside an ordered bracket")
        state = self._states.get(surface_id)
        if state is not None:
            if time <= state.time:
                raise ValueError(f"time for {surface_id!r} must increase strictly")
            if state.active == active:
                return EventBatch(())
        surface = self._surfaces[surface_id]
        self._states[surface_id] = _SurfaceState(
            time=time,
            oriented_value=0.0,
            raw_value=surface.threshold,
            active=active,
        )
        event = _surface_event(
            surface,
            active=active,
            time=time,
            bracket=bracket,
            localized=True,
            values=sample_values,
            reason=reason,
        )
        return EventBatch((event,))

    def _observe_one(self, surface: EventSurface, time: float, raw_value: float) -> DynamicEvent | None:
        oriented = _orient_value(surface, raw_value)
        state = self._states.get(surface.surface_id)
        if state is None:
            active = oriented <= (-surface.hysteresis if surface.hysteresis > 0.0 else 0.0)
            self._states[surface.surface_id] = _SurfaceState(time, oriented, raw_value, active)
            if active and self.emit_initial_events:
                return _surface_event(
                    surface,
                    active=True,
                    time=time,
                    bracket=(time, time),
                    localized=False,
                    values=(raw_value, raw_value),
                    reason="initial_active_state",
                )
            return None
        if time <= state.time:
            raise ValueError(f"time for {surface.surface_id!r} must increase strictly")

        pending = state.pending
        if not state.active:
            if state.oriented_value > 0.0 and oriented <= 0.0:
                pending = _make_pending(
                    True,
                    state.time,
                    time,
                    state.oriented_value,
                    oriented,
                    state.raw_value,
                    raw_value,
                )
            elif oriented > 0.0:
                pending = None
            confirmed = oriented <= (-surface.hysteresis if surface.hysteresis > 0.0 else 0.0)
            if confirmed:
                event = _event_from_pending(surface, pending, True, state, time, raw_value)
                self._states[surface.surface_id] = _SurfaceState(time, oriented, raw_value, True)
                return event
        else:
            if state.oriented_value < 0.0 and oriented >= 0.0:
                pending = _make_pending(
                    False,
                    state.time,
                    time,
                    state.oriented_value,
                    oriented,
                    state.raw_value,
                    raw_value,
                )
            elif oriented < 0.0:
                pending = None
            confirmed = oriented >= (surface.hysteresis if surface.hysteresis > 0.0 else 0.0)
            if confirmed:
                event = _event_from_pending(surface, pending, False, state, time, raw_value)
                self._states[surface.surface_id] = _SurfaceState(time, oriented, raw_value, False)
                return event

        self._states[surface.surface_id] = _SurfaceState(
            time=time,
            oriented_value=oriented,
            raw_value=raw_value,
            active=state.active,
            pending=pending,
        )
        return None


def _orient_value(surface: EventSurface, value: float) -> float:
    centered = value - surface.threshold
    return centered if surface.active_side is ActiveSide.BELOW else -centered


def _make_pending(
    active: bool,
    t0: float,
    t1: float,
    q0: float,
    q1: float,
    raw0: float,
    raw1: float,
) -> _PendingTransition:
    root_time, localized = _linear_zero(t0, t1, q0, q1)
    return _PendingTransition(active, root_time, (t0, t1), (raw0, raw1), localized)


def _linear_zero(t0: float, t1: float, q0: float, q1: float) -> tuple[float, bool]:
    denominator = q1 - q0
    if denominator == 0.0:
        return t1, False
    alpha = -q0 / denominator
    if alpha < 0.0 or alpha > 1.0:
        return t1, False
    return t0 + alpha * (t1 - t0), True


def _event_from_pending(
    surface: EventSurface,
    pending: _PendingTransition | None,
    active: bool,
    previous: _SurfaceState,
    time: float,
    raw_value: float,
) -> DynamicEvent:
    if pending is None or pending.active != active:
        return _surface_event(
            surface,
            active=active,
            time=time,
            bracket=(previous.time, time),
            localized=False,
            values=(previous.raw_value, raw_value),
            reason="transition_confirmed_without_observed_root",
        )
    return _surface_event(
        surface,
        active=active,
        time=pending.time,
        bracket=pending.bracket,
        localized=pending.localized,
        values=pending.values,
        reason="zero_crossing",
    )


def _surface_event(
    surface: EventSurface,
    *,
    active: bool,
    time: float,
    bracket: tuple[float, float],
    localized: bool,
    values: tuple[float, float],
    reason: str,
) -> DynamicEvent:
    payload = dict(surface.metadata)
    payload.update(
        {
            "surface_kind": surface.kind.value,
            "threshold": surface.threshold,
            "hysteresis": surface.hysteresis,
            "sample_values": values,
            "reason": reason,
        }
    )
    return DynamicEvent(
        kind=_TRANSITIONS[(surface.kind, active)],
        event_id=surface.surface_id,
        time=time,
        bracket=bracket,
        localized=localized,
        active=active,
        payload=payload,
    )


@dataclass(frozen=True)
class FoldObservation:
    """One Liu-style visible crease connecting two perceived fabric regions."""

    crease_id: str
    panel_id: str
    region_ids: tuple[str, str]
    confidence: float
    endpoint_ids: tuple[str, str] | None = None
    occlusion_front_region: str | None = None


class FoldGraphTracker:
    """Tracks births and deaths in a visible fold graph without remeshing cloth."""

    def __init__(self, threshold: float = 0.7, hysteresis: float = 0.05) -> None:
        if hysteresis < 0.0:
            raise ValueError("hysteresis must be non-negative")
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.detector = ZeroCrossingEventDetector(emit_initial_events=True)
        self._known: dict[str, FoldObservation] = {}
        self._active: dict[str, FoldObservation] = {}

    @property
    def active_creases(self) -> Mapping[str, FoldObservation]:
        return dict(self._active)

    def update(self, time: float, observations: Iterable[FoldObservation]) -> EventBatch:
        current = {observation.crease_id: observation for observation in observations}
        for crease_id, observation in current.items():
            if not isfinite(observation.confidence):
                raise ValueError(f"fold confidence for {crease_id!r} must be finite")
            if crease_id not in self._known:
                self.detector.register(
                    EventSurface(
                        surface_id=f"fold:{crease_id}",
                        kind=EventSurfaceKind.FOLD_CONFIDENCE,
                        active_side=ActiveSide.ABOVE,
                        threshold=self.threshold,
                        hysteresis=self.hysteresis,
                        metadata={"crease_id": crease_id, "panel_id": observation.panel_id},
                    )
                )
            self._known[crease_id] = observation

        values = {
            f"fold:{crease_id}": current.get(crease_id, replace(previous, confidence=0.0)).confidence
            for crease_id, previous in self._known.items()
        }
        batch = self.detector.observe(time, values)
        for event in batch.events:
            crease_id = str(event.payload["crease_id"])
            if event.kind is EventKind.FOLD_BIRTH:
                self._active[crease_id] = current.get(crease_id, self._known[crease_id])
            elif event.kind is EventKind.FOLD_DEATH:
                self._active.pop(crease_id, None)
        for crease_id in tuple(self._active):
            if crease_id in current:
                self._active[crease_id] = current[crease_id]
        return batch


@dataclass(frozen=True)
class IntersectionContourObservation:
    """A connected DCD intersection contour between one or two garment layers."""

    layer_ids: tuple[str, str]
    face_pairs: frozenset[tuple[int, int]]
    contour_nodes: frozenset[int]
    enclosed_nodes: frozenset[int] = frozenset()
    length: float = 0.0
    closed: bool = False
    nested: bool = False

    def __post_init__(self) -> None:
        if len(self.layer_ids) != 2:
            raise ValueError("layer_ids must contain exactly two entries")
        if not isfinite(self.length) or self.length < 0.0:
            raise ValueError("contour length must be finite and non-negative")


@dataclass(frozen=True)
class _TrackedContour:
    contour_id: str
    observation: IntersectionContourObservation


class IntersectionContourTracker:
    """Tracks ContourCraft-style contour components across sampled states."""

    def __init__(self, match_threshold: float = 0.2) -> None:
        if match_threshold < 0.0 or match_threshold > 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        self.match_threshold = match_threshold
        self._active: dict[str, _TrackedContour] = {}
        self._next_id = 1

    @property
    def active_contours(self) -> Mapping[str, IntersectionContourObservation]:
        return {key: tracked.observation for key, tracked in self._active.items()}

    @property
    def non_repulsive_nodes(self) -> frozenset[int]:
        nodes: set[int] = set()
        for tracked in self._active.values():
            contour = tracked.observation
            if contour.nested:
                continue
            if contour.closed:
                nodes.update(contour.enclosed_nodes)
            else:
                nodes.update(contour.contour_nodes)
        return frozenset(nodes)

    def update(self, time: float, observations: Iterable[IntersectionContourObservation]) -> EventBatch:
        if not isfinite(time):
            raise ValueError("time must be finite")
        current = list(observations)
        matches = self._match(current)
        matched_existing = {contour_id for contour_id, _ in matches}
        matched_current = {index for _, index in matches}
        events: list[DynamicEvent] = []
        next_active: dict[str, _TrackedContour] = {}

        for contour_id, index in matches:
            old = self._active[contour_id].observation
            new = current[index]
            next_active[contour_id] = _TrackedContour(contour_id, new)
            if old != new:
                events.append(_contour_event(EventKind.INTERSECTION_CONTOUR_UPDATE, contour_id, time, new))

        for contour_id, tracked in self._active.items():
            if contour_id not in matched_existing:
                events.append(
                    _contour_event(EventKind.INTERSECTION_CONTOUR_DEATH, contour_id, time, tracked.observation)
                )

        for index, observation in enumerate(current):
            if index in matched_current:
                continue
            contour_id = f"contour-{self._next_id:06d}"
            self._next_id += 1
            next_active[contour_id] = _TrackedContour(contour_id, observation)
            events.append(_contour_event(EventKind.INTERSECTION_CONTOUR_BIRTH, contour_id, time, observation))

        self._active = next_active
        events.sort(key=lambda event: (event.kind.value, event.event_id))
        return EventBatch(tuple(events))

    def _match(self, current: list[IntersectionContourObservation]) -> list[tuple[str, int]]:
        candidates: list[tuple[float, str, int]] = []
        for contour_id, tracked in self._active.items():
            old = tracked.observation
            for index, new in enumerate(current):
                if tuple(sorted(old.layer_ids)) != tuple(sorted(new.layer_ids)):
                    continue
                score = _jaccard(old.face_pairs, new.face_pairs)
                if score >= self.match_threshold:
                    candidates.append((score, contour_id, index))
        matches: list[tuple[str, int]] = []
        used_old: set[str] = set()
        used_new: set[int] = set()
        for _, contour_id, index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if contour_id in used_old or index in used_new:
                continue
            matches.append((contour_id, index))
            used_old.add(contour_id)
            used_new.add(index)
        return matches


def _jaccard(left: frozenset[tuple[int, int]], right: frozenset[tuple[int, int]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union)


def _contour_event(
    kind: EventKind,
    contour_id: str,
    time: float,
    contour: IntersectionContourObservation,
) -> DynamicEvent:
    return DynamicEvent(
        kind=kind,
        event_id=contour_id,
        time=time,
        bracket=(time, time),
        localized=False,
        active=kind is not EventKind.INTERSECTION_CONTOUR_DEATH,
        payload={
            "layer_ids": contour.layer_ids,
            "face_pair_count": len(contour.face_pairs),
            "length": contour.length,
            "closed": contour.closed,
            "nested": contour.nested,
            "requires_ccd_root_refinement": kind is not EventKind.INTERSECTION_CONTOUR_UPDATE,
        },
    )


class DynamicTopologyEventHandler:
    """Facade used by the per-scene NDF/CortexODE solver loop."""

    def __init__(self, fold_threshold: float = 0.7, fold_hysteresis: float = 0.05) -> None:
        self.zero_crossings = ZeroCrossingEventDetector()
        self.fold_graph = FoldGraphTracker(fold_threshold, fold_hysteresis)
        self.intersection_contours = IntersectionContourTracker()

    def register_surface(self, surface: EventSurface) -> None:
        if surface.kind is EventSurfaceKind.FOLD_CONFIDENCE:
            raise ValueError("fold surfaces are managed by FoldGraphTracker")
        self.zero_crossings.register(surface)

    def observe_gaps(self, time: float, values: Mapping[str, float]) -> EventBatch:
        return self.zero_crossings.observe(time, values)

    def record_ccd_root(
        self,
        surface_id: str,
        *,
        time: float,
        active: bool,
        bracket: tuple[float, float],
        sample_values: tuple[float, float],
    ) -> EventBatch:
        return self.zero_crossings.record_external_root(
            surface_id,
            time=time,
            active=active,
            bracket=bracket,
            sample_values=sample_values,
            reason="ccd_root",
        )

    def observe_folds(self, time: float, observations: Iterable[FoldObservation]) -> EventBatch:
        return self.fold_graph.update(time, observations)

    def observe_intersection_contours(
        self,
        time: float,
        observations: Iterable[IntersectionContourObservation],
    ) -> EventBatch:
        return self.intersection_contours.update(time, observations)

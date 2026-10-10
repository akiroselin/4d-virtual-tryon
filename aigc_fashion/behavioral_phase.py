from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin

from torch import Tensor


@dataclass(frozen=True)
class BehavioralPhaseSample:
    """E(3)-invariant high-level state attached to person/body graph nodes."""

    personality_ocean: Tensor
    emotion_vad: Tensor
    time: float

    def __post_init__(self) -> None:
        if self.personality_ocean.shape != (5,):
            raise ValueError("OCEAN personality must have five scalar channels")
        if self.emotion_vad.shape != (3,):
            raise ValueError("VAD emotion must have three scalar channels")


@dataclass(frozen=True)
class BehavioralControlPort:
    """Bounded invariant controls for an external articulated-body port."""

    cadence_scale: float
    stride_scale: float
    sway_scale: float
    gesture_scale: float


@dataclass(frozen=True)
class BehavioralPhaseTrajectory:
    """Serializable personality prior plus a periodic emotion trajectory.

    These variables are behavioral evidence, not cloth material parameters.
    They enter the person/body hierarchy and can alter body-control ports; cloth
    receives their effect through body motion and contact.
    """

    personality_ocean: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    emotion_vad_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emotion_vad_amplitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emotion_frequency_hz: float = 0.0
    emotion_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if len(self.personality_ocean) != 5:
            raise ValueError("personality_ocean must contain O,C,E,A,N")
        if len(self.emotion_vad_base) != 3 or len(self.emotion_vad_amplitude) != 3:
            raise ValueError("emotion values must contain valence, arousal, dominance")
        if any(abs(value) > 1.0 for value in self.personality_ocean):
            raise ValueError("personality channels must lie in [-1,1]")
        if any(abs(value) > 1.0 for value in self.emotion_vad_base):
            raise ValueError("emotion baseline must lie in [-1,1]")
        if any(value < 0.0 or value > 1.0 for value in self.emotion_vad_amplitude):
            raise ValueError("emotion amplitudes must lie in [0,1]")
        if self.emotion_frequency_hz < 0.0:
            raise ValueError("emotion frequency must be non-negative")

    def sample(self, time: float, reference: Tensor) -> BehavioralPhaseSample:
        personality, emotion = self.sample_values(time)
        return BehavioralPhaseSample(
            reference.new_tensor(personality),
            reference.new_tensor(emotion),
            float(time),
        )

    def sample_values(
        self,
        time: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        phase = 2.0 * pi * self.emotion_frequency_hz * float(time) + self.emotion_phase_rad
        oscillation = sin(phase)
        emotion = tuple(
            max(-1.0, min(1.0, base + oscillation * amplitude))
            for base, amplitude in zip(self.emotion_vad_base, self.emotion_vad_amplitude)
        )
        return tuple(self.personality_ocean), emotion

    def body_control_port(self, time: float) -> BehavioralControlPort:
        """Map behavioral scalars to bounded gait controls.

        This is an explicit engineering prior for simulation control, not a
        validated psychological measurement model.
        """

        personality, emotion = self.sample_values(time)
        openness, _conscientiousness, extraversion, _agreeableness, neuroticism = personality
        valence, arousal, dominance = emotion

        def bounded(value: float, lower: float, upper: float) -> float:
            return max(lower, min(upper, value))

        return BehavioralControlPort(
            cadence_scale=bounded(
                1.0 + 0.18 * extraversion + 0.25 * arousal - 0.08 * neuroticism,
                0.65,
                1.45,
            ),
            stride_scale=bounded(
                1.0 + 0.15 * extraversion + 0.20 * dominance + 0.08 * openness,
                0.70,
                1.40,
            ),
            sway_scale=bounded(
                1.0 + 0.20 * valence + 0.15 * extraversion + 0.10 * neuroticism,
                0.70,
                1.40,
            ),
            gesture_scale=bounded(
                1.0 + 0.30 * extraversion + 0.25 * arousal + 0.10 * openness,
                0.60,
                1.60,
            ),
        )

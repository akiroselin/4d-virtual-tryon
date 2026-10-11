"""Small dependency-free reader for the BVH motion-capture format.

The project only needs a few retargeting signals from a BVH clip, but parsing
the hierarchy here keeps the source joint names and channel order explicit. It
also avoids baking a particular commercial avatar rig into the motion layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BVHJoint:
    name: str
    offset: np.ndarray
    channels: tuple[str, ...]
    children: tuple["BVHJoint", ...]


@dataclass(frozen=True)
class BVHMotion:
    root: BVHJoint
    joints: tuple[BVHJoint, ...]
    channel_slices: dict[str, slice]
    frame_values: np.ndarray
    frame_time_s: float

    @property
    def frame_count(self) -> int:
        return int(self.frame_values.shape[0])

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.frame_count - 1) * self.frame_time_s)

    @classmethod
    def from_file(cls, path: str | Path) -> "BVHMotion":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text)

    @classmethod
    def from_text(cls, text: str) -> "BVHMotion":
        tokens = text.replace("{", " { ").replace("}", " } ").split()
        if not tokens or tokens[0].upper() != "HIERARCHY":
            raise ValueError("BVH must start with HIERARCHY")
        cursor = 1
        parsed_joints: list[BVHJoint] = []
        joint_order: list[str] = []

        def take(expected: str | None = None) -> str:
            nonlocal cursor
            if cursor >= len(tokens):
                raise ValueError("unexpected end of BVH hierarchy")
            value = tokens[cursor]
            cursor += 1
            if expected is not None and value.upper() != expected.upper():
                raise ValueError(f"expected {expected!r}, found {value!r}")
            return value

        def parse_joint(kind: str) -> BVHJoint:
            name = take()
            take("{")
            take("OFFSET")
            offset = np.asarray([float(take()), float(take()), float(take())], dtype=np.float32)
            # BVH channel rows are serialized in preorder. Record that order
            # before descending into children; the immutable joint object is
            # assembled after its subtree has been parsed.
            joint_order.append(name)
            channels: tuple[str, ...] = ()
            children: list[BVHJoint] = []
            while cursor < len(tokens) and tokens[cursor] != "}":
                token = tokens[cursor].upper()
                if token == "CHANNELS":
                    take("CHANNELS")
                    count = int(take())
                    channels = tuple(take() for _ in range(count))
                elif token == "JOINT":
                    take("JOINT")
                    children.append(parse_joint("JOINT"))
                elif token == "END":
                    take("END")
                    take("SITE")
                    take("{")
                    take("OFFSET")
                    take(), take(), take()
                    take("}")
                else:
                    raise ValueError(f"unexpected BVH hierarchy token {tokens[cursor]!r}")
            take("}")
            joint = BVHJoint(name, offset, channels, tuple(children))
            parsed_joints.append(joint)
            return joint

        take("ROOT")
        root = parse_joint("ROOT")
        if cursor >= len(tokens) or tokens[cursor].upper() != "MOTION":
            raise ValueError("BVH hierarchy is not followed by MOTION")
        take("MOTION")
        take("FRAMES:")
        frame_count = int(take())
        take("FRAME")
        take("TIME:")
        frame_time_s = float(take())
        numeric = np.fromstring(" ".join(tokens[cursor:]), sep=" ", dtype=np.float32)
        ordered_joints = tuple(
            next(joint for joint in parsed_joints if joint.name == name)
            for name in joint_order
        )
        total_channels = sum(len(joint.channels) for joint in ordered_joints)
        expected = frame_count * total_channels
        if numeric.size != expected:
            raise ValueError(
                f"BVH motion has {numeric.size} values, expected {expected} "
                f"({frame_count} frames x {total_channels} channels)"
            )
        frame_values = numeric.reshape(frame_count, total_channels)
        channel_slices: dict[str, slice] = {}
        offset = 0
        for joint in ordered_joints:
            channel_slices[joint.name] = slice(offset, offset + len(joint.channels))
            offset += len(joint.channels)
        return cls(root, ordered_joints, channel_slices, frame_values, frame_time_s)

    def _joint(self, name: str) -> BVHJoint:
        wanted = name.casefold()
        for joint in self.joints:
            if joint.name.casefold() == wanted:
                return joint
        raise KeyError(f"BVH joint not found: {name}")

    def channel(self, joint_name: str, channel_name: str) -> np.ndarray:
        joint = self._joint(joint_name)
        try:
            local_index = next(
                index
                for index, channel in enumerate(joint.channels)
                if channel.casefold() == channel_name.casefold()
            )
        except StopIteration as error:
            raise KeyError(f"BVH channel not found: {joint_name}.{channel_name}") from error
        span = self.channel_slices[joint.name]
        return self.frame_values[:, span.start + local_index].copy()

    def first_available_channel(
        self, joint_names: Iterable[str], channel_names: Iterable[str]
    ) -> np.ndarray:
        for joint_name in joint_names:
            for channel_name in channel_names:
                try:
                    return self.channel(joint_name, channel_name)
                except KeyError:
                    continue
        names = ", ".join(joint_names)
        channels = ", ".join(channel_names)
        raise KeyError(f"no BVH channel among joints [{names}] and channels [{channels}]")

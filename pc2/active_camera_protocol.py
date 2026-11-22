"""Protocol parsing and 2-DoF head-pose conversion.

The wire coordinate system is the OpenXR right-handed convention:
+X right, +Y up, and -Z forward. Quaternions are ordered x, y, z, w.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Optional, Tuple


MAGIC = b"ACAM"
VERSION = 1
PACKET_STRUCT = struct.Struct("!4sBBHIQ3f4f")
PACKET_SIZE = PACKET_STRUCT.size

FLAG_ORIENTATION_VALID = 1 << 0
FLAG_POSITION_VALID = 1 << 1
FLAG_TRACKED = 1 << 2
FLAG_RECENTER = 1 << 3

Quaternion = Tuple[float, float, float, float]
Vector3 = Tuple[float, float, float]


class PacketError(ValueError):
    """Raised when a UDP datagram is not a valid active-camera packet."""


@dataclass(frozen=True)
class HeadPosePacket:
    sequence: int
    sender_time_ns: int
    position: Vector3
    orientation: Quaternion
    flags: int

    @property
    def orientation_valid(self) -> bool:
        return bool(self.flags & FLAG_ORIENTATION_VALID)

    @property
    def position_valid(self) -> bool:
        return bool(self.flags & FLAG_POSITION_VALID)

    @property
    def tracked(self) -> bool:
        return bool(self.flags & FLAG_TRACKED)

    @property
    def recenter_requested(self) -> bool:
        return bool(self.flags & FLAG_RECENTER)


@dataclass(frozen=True)
class YawPitchTarget:
    sequence: int
    sender_time_ns: int
    yaw_deg: float
    pitch_deg: float
    position: Vector3


def decode_packet(datagram: bytes) -> HeadPosePacket:
    """Decode and validate one fixed-size UDP datagram."""

    if len(datagram) != PACKET_SIZE:
        raise PacketError(f"wrong packet size: {len(datagram)} != {PACKET_SIZE}")

    (
        magic,
        version,
        flags,
        reserved,
        sequence,
        sender_time_ns,
        px,
        py,
        pz,
        qx,
        qy,
        qz,
        qw,
    ) = PACKET_STRUCT.unpack(datagram)

    if magic != MAGIC:
        raise PacketError("wrong magic")
    if version != VERSION:
        raise PacketError(f"unsupported version: {version}")
    if reserved != 0:
        raise PacketError("reserved field must be zero")

    values = (px, py, pz, qx, qy, qz, qw)
    if not all(math.isfinite(value) for value in values):
        raise PacketError("pose contains a non-finite value")

    orientation = (qx, qy, qz, qw)
    if flags & FLAG_ORIENTATION_VALID:
        norm = quaternion_norm(orientation)
        if not 0.95 <= norm <= 1.05:
            raise PacketError(f"quaternion is not unit length: {norm:.6f}")
        orientation = normalize_quaternion(orientation)

    return HeadPosePacket(
        sequence=sequence,
        sender_time_ns=sender_time_ns,
        position=(px, py, pz),
        orientation=orientation,
        flags=flags,
    )


def is_newer_sequence(sequence: int, previous: Optional[int]) -> bool:
    """Return True if an unsigned 32-bit sequence is newer than previous."""

    if previous is None:
        return True
    difference = (sequence - previous) & 0xFFFFFFFF
    return 0 < difference < 0x80000000


def quaternion_norm(q: Quaternion) -> float:
    return math.sqrt(sum(component * component for component in q))


def normalize_quaternion(q: Quaternion) -> Quaternion:
    norm = quaternion_norm(q)
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a * b for x/y/z/w ordered quaternions."""

    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def relative_orientation(neutral: Quaternion, current: Quaternion) -> Quaternion:
    """Return current orientation expressed relative to neutral orientation."""

    return normalize_quaternion(
        quaternion_multiply(quaternion_conjugate(neutral), current)
    )


def level_neutral(orientation: Quaternion) -> Quaternion:
    """Keep only the heading (yaw) of an orientation: a rotation about the vertical (+Y) axis.

    The OpenXR world is gravity-aligned, so a heading-only neutral makes pitch the true angle above/below
    the horizon and removes any roll/tilt the headset had when the neutral was captured. Without this a
    tilted first sample makes a pure nod show up as yaw as well.
    """

    x, y, z, w = normalize_quaternion(orientation)
    forward_x = -2.0 * (x * z + y * w)
    forward_z = -(1.0 - 2.0 * (x * x + y * y))
    if math.hypot(forward_x, forward_z) < 1e-6:      # looking straight up/down: heading undefined
        return (0.0, 0.0, 0.0, 1.0)
    heading = math.atan2(forward_x, -forward_z)       # same yaw convention as quaternion_to_yaw_pitch
    half = -heading / 2.0                             # rotation about +Y by -heading turns -Z toward +heading
    return (0.0, math.sin(half), 0.0, math.cos(half))


def quaternion_to_yaw_pitch(q: Quaternion) -> Tuple[float, float]:
    """Convert a relative OpenXR quaternion to right-positive yaw/up-positive pitch.

    Roll is intentionally removed by first rotating the OpenXR forward vector
    (0, 0, -1) and then measuring its azimuth and elevation.
    """

    x, y, z, w = normalize_quaternion(q)

    # forward = R(q) * (0, 0, -1)
    forward_x = -2.0 * (x * z + y * w)
    forward_y = 2.0 * (x * w - y * z)
    forward_z = -(1.0 - 2.0 * (x * x + y * y))

    yaw = math.atan2(forward_x, -forward_z)
    pitch = math.atan2(
        forward_y,
        math.sqrt(forward_x * forward_x + forward_z * forward_z),
    )
    return yaw, pitch


class TwoDofPoseMapper:
    """Stateful neutral calibration, limits, and optional target smoothing."""

    def __init__(
        self,
        yaw_limit_deg: Optional[float] = 80.0,
        pitch_down_limit_deg: Optional[float] = 35.0,
        pitch_up_limit_deg: Optional[float] = 45.0,
        smoothing: float = 1.0,
        neutral_mode: str = "first",
    ) -> None:
        # neutral_mode: "first" = the whole orientation of the first sample is the neutral (original behaviour);
        #               "level" = only its heading is, pitch/roll are taken from gravity (see level_neutral).
        if neutral_mode not in ("first", "level"):
            raise ValueError("neutral_mode must be 'first' or 'level'")
        self.neutral_mode = neutral_mode
        # None = no angle limit on that axis (only the motor's 0..4095 position range still applies).
        if yaw_limit_deg is not None and yaw_limit_deg <= 0:
            raise ValueError("yaw_limit_deg must be positive")
        if any(limit is not None and limit <= 0 for limit in (pitch_down_limit_deg, pitch_up_limit_deg)):
            raise ValueError("pitch limits must be positive")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")

        self.yaw_limit_deg = yaw_limit_deg
        self.pitch_down_limit_deg = pitch_down_limit_deg
        self.pitch_up_limit_deg = pitch_up_limit_deg
        self.smoothing = smoothing
        self.neutral: Optional[Quaternion] = None
        self._filtered_yaw: Optional[float] = None
        self._filtered_pitch: Optional[float] = None

    def recenter(self, orientation: Quaternion) -> None:
        if self.neutral_mode == "level":
            self.neutral = level_neutral(orientation)
            # The first sample is not necessarily level, so do not seed the smoothing filter with zero.
            self._filtered_yaw = None
            self._filtered_pitch = None
        else:
            self.neutral = normalize_quaternion(orientation)
            self._filtered_yaw = 0.0
            self._filtered_pitch = 0.0

    def update(self, packet: HeadPosePacket) -> Optional[YawPitchTarget]:
        if not packet.orientation_valid or not packet.tracked:
            return None

        if self.neutral is None or packet.recenter_requested:
            self.recenter(packet.orientation)

        assert self.neutral is not None
        relative = relative_orientation(self.neutral, packet.orientation)
        yaw_rad, pitch_rad = quaternion_to_yaw_pitch(relative)
        yaw = math.degrees(yaw_rad)
        pitch = math.degrees(pitch_rad)

        if self.yaw_limit_deg is not None:
            yaw = max(-self.yaw_limit_deg, min(self.yaw_limit_deg, yaw))
        if self.pitch_down_limit_deg is not None:
            pitch = max(-self.pitch_down_limit_deg, pitch)
        if self.pitch_up_limit_deg is not None:
            pitch = min(self.pitch_up_limit_deg, pitch)

        if self._filtered_yaw is None or self._filtered_pitch is None:
            self._filtered_yaw = yaw
            self._filtered_pitch = pitch
        else:
            alpha = self.smoothing
            self._filtered_yaw += alpha * (yaw - self._filtered_yaw)
            self._filtered_pitch += alpha * (pitch - self._filtered_pitch)

        return YawPitchTarget(
            sequence=packet.sequence,
            sender_time_ns=packet.sender_time_ns,
            yaw_deg=self._filtered_yaw,
            pitch_deg=self._filtered_pitch,
            position=packet.position,
        )


from __future__ import annotations

import pytest

from carrito_smart.detection_stabilizer import (
    RawDetection,
    TemporalDetectionStabilizer,
)


BOX = (10, 20, 110, 220)


def make_stabilizer() -> TemporalDetectionStabilizer:
    return TemporalDetectionStabilizer(
        detection_threshold=0.60,
        retention_threshold=0.45,
        confirmation_count=3,
        detection_hold_ms=700,
        confidence_ema_alpha=0.30,
    )


def raw(confidence: float, bbox=BOX) -> RawDetection:
    return RawDetection("person", confidence, bbox)


def test_requires_three_consecutive_detections_and_smooths_confidence():
    stabilizer = make_stabilizer()

    assert stabilizer.update([raw(0.65)], now=0.0)[0] == []
    assert stabilizer.update([raw(0.70)], now=0.1)[0] == []
    stable, decisions = stabilizer.update([raw(0.80)], now=0.2)

    assert len(stable) == 1
    assert stable[0].confidence == pytest.approx(0.7055)
    assert decisions[0].decision == "confirmed"


def test_hysteresis_maintains_active_detection_below_acceptance_threshold():
    stabilizer = make_stabilizer()
    for index in range(3):
        stabilizer.update([raw(0.70)], now=index * 0.1)

    stable, decisions = stabilizer.update([raw(0.50)], now=0.3)

    assert len(stable) == 1
    assert not stable[0].held
    assert decisions[0].decision == "maintained"


def test_new_detection_below_acceptance_threshold_is_discarded():
    stable, decisions = make_stabilizer().update([raw(0.55)], now=0.0)

    assert stable == []
    assert decisions[0].decision == "discarded"


def test_last_box_is_held_then_expires_after_tolerance():
    stabilizer = make_stabilizer()
    for index in range(3):
        stabilizer.update([raw(0.70)], now=index * 0.1)
    last_box = (30, 40, 130, 240)
    stabilizer.update([raw(0.50, last_box)], now=0.3)

    held, decisions = stabilizer.update([], now=0.99)
    expired, expiration_decisions = stabilizer.update([], now=1.01)

    assert held[0].bbox == last_box
    assert held[0].held
    assert decisions[0].decision == "held"
    assert expired == []
    assert expiration_decisions[0].decision == "expired"


def test_interrupted_candidate_must_restart_confirmation():
    stabilizer = make_stabilizer()
    stabilizer.update([raw(0.80)], now=0.0)

    stable, decisions = stabilizer.update([], now=0.1)

    assert stable == []
    assert decisions[0].decision == "discarded"

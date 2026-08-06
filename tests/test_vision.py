from __future__ import annotations

from carrito_smart.vision import extract_detections


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeScalarList:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBox:
    cls = [FakeScalar(1)]
    conf = [FakeScalar(0.875)]
    xyxy = [FakeScalarList([10.2, 20.4, 110.6, 220.8])]


class FakeResult:
    boxes = [FakeBox()]
    names = {1: "bicycle"}


def test_extract_detections_returns_simple_values():
    detections = extract_detections(FakeResult())

    assert len(detections) == 1
    assert detections[0].class_name == "bicycle"
    assert detections[0].confidence == 0.875
    assert detections[0].bbox == (10, 20, 111, 221)

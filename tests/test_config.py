from __future__ import annotations

import pytest

from carrito_smart.config import AppConfig


def test_vision_defaults_are_centralized(monkeypatch, tmp_path):
    variables = (
        "CARRITO_SMART_CAMERA_FPS",
        "CARRITO_SMART_INFERENCE_FPS",
        "CARRITO_SMART_UI_UPDATE_FPS",
        "CARRITO_SMART_DETECTION_THRESHOLD",
        "CARRITO_SMART_RETENTION_THRESHOLD",
        "CARRITO_SMART_CONFIRMATION_COUNT",
        "CARRITO_SMART_DETECTION_HOLD_MS",
        "CARRITO_SMART_CONFIDENCE_EMA_ALPHA",
        "CARRITO_SMART_CONFIDENCE",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("CARRITO_SMART_DATA_DIR", str(tmp_path))

    config = AppConfig.from_env()

    assert config.camera_fps == 30
    assert config.inference_fps == 8
    assert config.ui_update_fps == 4
    assert config.detection_threshold == 0.60
    assert config.retention_threshold == 0.45
    assert config.confirmation_count == 3
    assert config.detection_hold_ms == 700
    assert config.confidence_ema_alpha == 0.30
    assert config.yolo_model == "yolo26n.pt"
    assert config.fallback_yolo_model == "yolo11n.pt"


def test_invalid_hysteresis_is_rejected(tmp_path):
    config = AppConfig.from_env()
    values = {
        field: getattr(config, field)
        for field in config.__dataclass_fields__
    }
    values["data_dir"] = tmp_path
    values["database_path"] = tmp_path / "test.db"
    values["retention_threshold"] = 0.70
    values["detection_threshold"] = 0.60

    with pytest.raises(ValueError, match="retention"):
        AppConfig(**values)

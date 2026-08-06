"""Configuración central de la aplicación."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class AppConfig:
    data_dir: Path
    log_dir: Path
    database_path: Path
    camera_index: int
    yolo_model: str
    fallback_yolo_model: str
    camera_width: int
    camera_height: int
    camera_fps: float
    inference_fps: float
    ui_update_fps: float
    detection_threshold: float
    retention_threshold: float
    confirmation_count: int
    detection_hold_ms: int
    confidence_ema_alpha: float
    inference_image_size: int

    def __post_init__(self) -> None:
        if min(self.camera_fps, self.inference_fps, self.ui_update_fps) <= 0:
            raise ValueError("Las frecuencias deben ser mayores que cero")
        if not 0 <= self.retention_threshold <= self.detection_threshold <= 1:
            raise ValueError(
                "Los umbrales deben cumplir 0 <= retention <= detection <= 1"
            )
        if self.confirmation_count < 1:
            raise ValueError("confirmation_count debe ser al menos 1")
        if self.detection_hold_ms < 0:
            raise ValueError("detection_hold_ms no puede ser negativo")
        if not 0 < self.confidence_ema_alpha <= 1:
            raise ValueError("confidence_ema_alpha debe estar entre 0 y 1")

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = Path(os.getenv("CARRITO_SMART_DATA_DIR", PROJECT_ROOT / "data"))
        log_dir = Path(os.getenv("CARRITO_SMART_LOG_DIR", PROJECT_ROOT / "logs"))
        return cls(
            data_dir=data_dir,
            log_dir=log_dir,
            database_path=data_dir / "carrito_smart.db",
            camera_index=int(os.getenv("CARRITO_SMART_CAMERA", "0")),
            yolo_model=os.getenv("CARRITO_SMART_YOLO_MODEL", "yolo26n.pt"),
            fallback_yolo_model=os.getenv(
                "CARRITO_SMART_FALLBACK_MODEL", "yolo11n.pt"
            ),
            camera_width=int(os.getenv("CARRITO_SMART_CAMERA_WIDTH", "1280")),
            camera_height=int(os.getenv("CARRITO_SMART_CAMERA_HEIGHT", "720")),
            camera_fps=float(os.getenv("CARRITO_SMART_CAMERA_FPS", "30")),
            inference_fps=float(os.getenv("CARRITO_SMART_INFERENCE_FPS", "8")),
            ui_update_fps=float(os.getenv("CARRITO_SMART_UI_UPDATE_FPS", "4")),
            detection_threshold=float(
                os.getenv(
                    "CARRITO_SMART_DETECTION_THRESHOLD",
                    os.getenv("CARRITO_SMART_CONFIDENCE", "0.60"),
                )
            ),
            retention_threshold=float(
                os.getenv("CARRITO_SMART_RETENTION_THRESHOLD", "0.45")
            ),
            confirmation_count=int(
                os.getenv("CARRITO_SMART_CONFIRMATION_COUNT", "3")
            ),
            detection_hold_ms=int(
                os.getenv("CARRITO_SMART_DETECTION_HOLD_MS", "700")
            ),
            confidence_ema_alpha=float(
                os.getenv("CARRITO_SMART_CONFIDENCE_EMA_ALPHA", "0.30")
            ),
            inference_image_size=int(
                os.getenv("CARRITO_SMART_INFERENCE_IMAGE_SIZE", "640")
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from carrito_smart.config import AppConfig
from carrito_smart.database import Database
from carrito_smart.main_window import MainWindow


def test_main_window_builds_without_starting_camera(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "ui.db")
    database.initialize()
    config = AppConfig(
        data_dir=tmp_path,
        log_dir=tmp_path,
        database_path=database.path,
        camera_index=0,
        yolo_model="yolo11n.pt",
        fallback_yolo_model="yolo11n.pt",
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
        inference_fps=8,
        ui_update_fps=4,
        detection_threshold=0.60,
        retention_threshold=0.45,
        confirmation_count=3,
        detection_hold_ms=700,
        confidence_ema_alpha=0.30,
        inference_image_size=640,
    )
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)

    window = MainWindow(database, config)

    assert window.product_combo.count() == 6
    assert window.cart_table.rowCount() == 0
    assert not window.pay_button.isEnabled()
    window.close()
    application.processEvents()

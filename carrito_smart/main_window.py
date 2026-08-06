"""Ventana principal de Carrito Smart."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from carrito_smart.cart import CartError, CartService
from carrito_smart.config import AppConfig
from carrito_smart.database import Database, InventoryError, SaleValidationError
from carrito_smart.models import format_money
from carrito_smart.vision import (
    CameraWorker,
    InferenceWorker,
    LatestFrameBuffer,
    StableDetectionBuffer,
)


LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, database: Database, config: AppConfig) -> None:
        super().__init__()
        self.database = database
        self.config = config
        self.cart = CartService(database)
        self._last_frame: QImage | None = None
        self._camera_thread: QThread | None = None
        self._camera_worker: CameraWorker | None = None
        self._inference_thread: QThread | None = None
        self._inference_worker: InferenceWorker | None = None
        self._vision_metrics: dict[str, object] = {}

        self.setWindowTitle("Carrito Smart · Prototipo")
        self.resize(1280, 760)
        self.setMinimumSize(1050, 650)
        self._build_ui()
        self._apply_styles()
        self._load_products()
        self._refresh_cart()
        QTimer.singleShot(150, self._start_vision)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        vision_panel = QVBoxLayout()
        title = QLabel("Visión del carrito")
        title.setObjectName("sectionTitle")
        vision_panel.addWidget(title)

        self.video_label = QLabel("Preparando cámara…")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(620, 420)
        vision_panel.addWidget(self.video_label, 1)

        status_row = QHBoxLayout()
        self.vision_status = QLabel("Inicializando")
        self.vision_status.setObjectName("statusPill")
        self.detection_count = QLabel("0 detecciones")
        status_row.addWidget(self.vision_status)
        status_row.addStretch()
        status_row.addWidget(self.detection_count)
        vision_panel.addLayout(status_row)

        self.performance_label = QLabel(
            f"Captura ≤ {self.config.camera_fps:.0f} FPS · "
            f"YOLO ≤ {self.config.inference_fps:.0f} FPS · "
            f"Texto ≤ {self.config.ui_update_fps:.0f} FPS"
        )
        self.performance_label.setStyleSheet("color: #52667a; font-size: 12px;")
        vision_panel.addWidget(self.performance_label)

        detections_group = QGroupBox("Clases detectadas y confianza")
        detections_layout = QVBoxLayout(detections_group)
        self.detection_list = QListWidget()
        self.detection_list.setMaximumHeight(125)
        self.detection_list.addItem("Aún no hay detecciones")
        detections_layout.addWidget(self.detection_list)
        vision_panel.addWidget(detections_group)

        cart_panel = QVBoxLayout()
        cart_title = QLabel("Compra actual")
        cart_title.setObjectName("sectionTitle")
        cart_panel.addWidget(cart_title)

        simulator = QGroupBox("Simulador RFID")
        simulator_layout = QGridLayout(simulator)
        self.product_combo = QComboBox()
        self.product_combo.setMinimumWidth(320)
        self.add_button = QPushButton("＋ Entrada")
        self.add_button.setObjectName("primaryButton")
        self.remove_button = QPushButton("− Salida")
        simulator_layout.addWidget(QLabel("Producto"), 0, 0, 1, 2)
        simulator_layout.addWidget(self.product_combo, 1, 0, 1, 2)
        simulator_layout.addWidget(self.add_button, 2, 0)
        simulator_layout.addWidget(self.remove_button, 2, 1)
        cart_panel.addWidget(simulator)

        self.cart_table = QTableWidget(0, 4)
        self.cart_table.setHorizontalHeaderLabels(
            ["Producto", "Cant.", "Precio", "Subtotal"]
        )
        self.cart_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.cart_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.cart_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cart_table.verticalHeader().setVisible(False)
        header = self.cart_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        cart_panel.addWidget(self.cart_table, 1)

        total_frame = QFrame()
        total_frame.setObjectName("totalFrame")
        total_layout = QHBoxLayout(total_frame)
        total_layout.addWidget(QLabel("TOTAL"))
        total_layout.addStretch()
        self.total_label = QLabel(format_money(0))
        self.total_label.setObjectName("totalLabel")
        total_layout.addWidget(self.total_label)
        cart_panel.addWidget(total_frame)

        actions = QHBoxLayout()
        self.clear_button = QPushButton("Vaciar")
        self.pay_button = QPushButton("Simular pago aprobado")
        self.pay_button.setObjectName("payButton")
        actions.addWidget(self.clear_button)
        actions.addWidget(self.pay_button, 1)
        cart_panel.addLayout(actions)

        left_widget = QWidget()
        left_widget.setLayout(vision_panel)
        right_widget = QWidget()
        right_widget.setLayout(cart_panel)
        root.addWidget(left_widget, 3)
        root.addWidget(right_widget, 2)
        self.setCentralWidget(central)

        self.add_button.clicked.connect(self._simulate_entry)
        self.remove_button.clicked.connect(self._simulate_exit)
        self.clear_button.clicked.connect(self._clear_cart)
        self.pay_button.clicked.connect(self._pay)
        self.cart_table.itemSelectionChanged.connect(self._sync_combo_to_selection)
        self.statusBar().showMessage("Listo")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f3f5f7; color: #17212b; font-size: 14px; }
            QLabel#sectionTitle { font-size: 23px; font-weight: 700; color: #102a43; }
            QLabel#video { background: #0c1117; color: #91a3b5; border-radius: 10px; }
            QLabel#statusPill { background: #e0f2fe; color: #075985; padding: 6px 10px; border-radius: 8px; }
            QGroupBox { background: white; border: 1px solid #d9e2ec; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QComboBox, QTableWidget, QListWidget { background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px; }
            QPushButton { background: #e2e8f0; border: none; border-radius: 6px; padding: 10px 14px; font-weight: 600; }
            QPushButton:hover { background: #cbd5e1; }
            QPushButton#primaryButton { background: #2563eb; color: white; }
            QPushButton#primaryButton:hover { background: #1d4ed8; }
            QPushButton#payButton { background: #16a34a; color: white; font-size: 15px; }
            QPushButton#payButton:hover { background: #15803d; }
            QPushButton:disabled { background: #cbd5e1; color: #64748b; }
            QFrame#totalFrame { background: #102a43; border-radius: 8px; }
            QFrame#totalFrame QLabel { background: transparent; color: white; font-weight: 700; }
            QLabel#totalLabel { font-size: 25px; }
            QHeaderView::section { background: #e8edf2; border: none; padding: 8px; font-weight: 700; }
            """
        )

    def _load_products(self) -> None:
        selected_id = self.product_combo.currentData()
        self.product_combo.clear()
        products = self.database.list_products()
        for product in products:
            self.product_combo.addItem(
                f"{product.name} · {format_money(product.price_cents)} · stock {product.stock}",
                product.id,
            )
        if selected_id is not None:
            index = self.product_combo.findData(selected_id)
            if index >= 0:
                self.product_combo.setCurrentIndex(index)

    def _refresh_cart(self) -> None:
        items = self.cart.items
        self.cart_table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = (
                item.product.name,
                str(item.quantity),
                format_money(item.product.price_cents),
                format_money(item.subtotal_cents),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item.product.id)
                if column > 0:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.cart_table.setItem(row, column, cell)
        total = self.cart.total_cents
        self.total_label.setText(format_money(total))
        self.pay_button.setEnabled(bool(items))
        self.clear_button.setEnabled(bool(items))

    @Slot()
    def _simulate_entry(self) -> None:
        product_id = self.product_combo.currentData()
        if product_id is None:
            return
        try:
            self.cart.add_product(int(product_id))
        except CartError as error:
            QMessageBox.warning(self, "Entrada rechazada", str(error))
            return
        self._refresh_cart()
        self.statusBar().showMessage("Entrada RFID simulada", 3000)

    @Slot()
    def _simulate_exit(self) -> None:
        product_id = self._selected_cart_product_id()
        if product_id is None:
            product_id = self.product_combo.currentData()
        if product_id is None:
            return
        try:
            self.cart.remove_product(int(product_id))
        except CartError as error:
            QMessageBox.information(self, "Salida simulada", str(error))
            return
        self._refresh_cart()
        self.statusBar().showMessage("Salida RFID simulada", 3000)

    @Slot()
    def _clear_cart(self) -> None:
        self.cart.clear()
        self._refresh_cart()
        self.statusBar().showMessage("Carrito vaciado", 3000)

    @Slot()
    def _pay(self) -> None:
        total = self.cart.total_cents
        answer = QMessageBox.question(
            self,
            "Confirmar pago simulado",
            f"¿Aprobar el pago por {format_money(total)}?\n\n"
            "Al confirmar se registrará la venta y se descontará el inventario.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            LOGGER.info("Pago simulado cancelado por el usuario")
            return
        try:
            receipt = self.cart.checkout()
        except (InventoryError, SaleValidationError) as error:
            QMessageBox.critical(self, "Pago rechazado", str(error))
            return
        self._refresh_cart()
        self._load_products()
        QMessageBox.information(
            self,
            "Pago aprobado",
            f"Venta #{receipt.sale_id} registrada correctamente.\n"
            f"Total: {format_money(receipt.total_cents)}",
        )
        self.statusBar().showMessage(f"Venta #{receipt.sale_id} aprobada", 5000)

    def _selected_cart_product_id(self) -> int | None:
        row = self.cart_table.currentRow()
        if row < 0:
            return None
        item = self.cart_table.item(row, 0)
        return int(item.data(Qt.ItemDataRole.UserRole)) if item else None

    @Slot()
    def _sync_combo_to_selection(self) -> None:
        product_id = self._selected_cart_product_id()
        if product_id is None:
            return
        index = self.product_combo.findData(product_id)
        if index >= 0:
            self.product_combo.setCurrentIndex(index)

    def _start_vision(self) -> None:
        frames = LatestFrameBuffer()
        stable_detections = StableDetectionBuffer()

        self._camera_thread = QThread(self)
        self._camera_worker = CameraWorker(
            self.config, frames, stable_detections
        )
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_thread.started.connect(self._camera_worker.run)
        self._camera_worker.frame_ready.connect(self._show_frame)
        self._camera_worker.status_changed.connect(self._show_vision_status)
        self._camera_worker.metrics_ready.connect(self._show_vision_metrics)
        self._camera_worker.failed.connect(self._show_camera_error)
        self._camera_worker.finished.connect(
            self._camera_thread.quit, Qt.ConnectionType.DirectConnection
        )

        self._inference_thread = QThread(self)
        self._inference_worker = InferenceWorker(
            self.config, frames, stable_detections
        )
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.detections_ready.connect(self._show_detections)
        self._inference_worker.status_changed.connect(self._show_vision_status)
        self._inference_worker.metrics_ready.connect(self._show_vision_metrics)
        self._inference_worker.failed.connect(self._show_inference_error)
        self._inference_worker.finished.connect(
            self._inference_thread.quit, Qt.ConnectionType.DirectConnection
        )

        self._inference_thread.start()
        self._camera_thread.start()

    @Slot(QImage)
    def _show_frame(self, image: QImage) -> None:
        self._last_frame = image
        self._render_last_frame()

    def _render_last_frame(self) -> None:
        if self._last_frame is None:
            return
        pixmap = QPixmap.fromImage(self._last_frame)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_last_frame()

    @Slot(list)
    def _show_detections(self, detections: list[dict]) -> None:
        self.detection_list.clear()
        if not detections:
            self.detection_list.addItem("Sin objetos sobre el umbral configurado")
        else:
            for detection in detections:
                confidence = float(detection["confidence"]) * 100
                state = " · retenida" if detection.get("held") else ""
                self.detection_list.addItem(
                    f"{detection['class_name']}  ·  {confidence:.1f}%{state}"
                )
        count = len(detections)
        noun = "detección" if count == 1 else "detecciones"
        self.detection_count.setText(f"{count} {noun}")

    @Slot(str)
    def _show_vision_status(self, message: str) -> None:
        self.vision_status.setText(message)

    @Slot(str)
    def _show_camera_error(self, message: str) -> None:
        self.vision_status.setText("Visión no disponible")
        self.video_label.setText(f"No se pudo iniciar la visión\n\n{message}")
        self.statusBar().showMessage("La compra manual sigue disponible")
        if self._inference_worker is not None:
            self._inference_worker.request_stop()

    @Slot(str)
    def _show_inference_error(self, message: str) -> None:
        self.vision_status.setText("YOLO no disponible; webcam activa")
        self.statusBar().showMessage(f"Error de inferencia: {message}")

    @Slot(dict)
    def _show_vision_metrics(self, metrics: dict[str, object]) -> None:
        self._vision_metrics.update(metrics)
        capture = self._vision_metrics.get("capture_fps")
        inference = self._vision_metrics.get("inference_fps")
        latency = self._vision_metrics.get("latency_ms")
        device = self._vision_metrics.get("device", "—")
        capture_text = f"{float(capture):.1f}" if capture is not None else "—"
        inference_text = f"{float(inference):.1f}" if inference is not None else "—"
        latency_text = f"{float(latency):.0f} ms" if latency is not None else "—"
        self.performance_label.setText(
            f"Captura {capture_text} FPS · YOLO {inference_text} FPS / "
            f"{latency_text} · {device} · Texto ≤ {self.config.ui_update_fps:.0f} FPS"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        for worker in (self._camera_worker, self._inference_worker):
            if worker is not None:
                worker.request_stop()
        for name, thread in (
            ("captura", self._camera_thread),
            ("inferencia", self._inference_thread),
        ):
            if thread is not None and thread.isRunning() and not thread.wait(7000):
                LOGGER.warning("El hilo de %s no respondió a tiempo al cierre", name)
                event.ignore()
                self.statusBar().showMessage("Esperando a que termine la inferencia…")
                QTimer.singleShot(1000, self.close)
                return
        event.accept()

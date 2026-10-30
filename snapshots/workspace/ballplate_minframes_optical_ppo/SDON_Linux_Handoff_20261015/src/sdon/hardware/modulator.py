"""Thread-safe full-screen grayscale output for a DMD or SLM."""

import hashlib
import threading
import time
from typing import Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class OpticalModulator(QtWidgets.QWidget):
    update_image_signal = QtCore.pyqtSignal(object)
    update_offset_signal = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        screen: QtGui.QScreen,
        modulated_size: Tuple[int, int] = (512, 512),
        device_name: str = "Device",
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.screen = screen
        self.device_name = device_name
        self.modulated_size = tuple(modulated_size)
        self.x_offset, self.y_offset = map(int, offset)
        self.image_applied = threading.Event()
        self.offset_applied = threading.Event()
        self._display_state_lock = threading.Lock()
        self._requested_sequence = 0
        self._applied_sequence = 0
        self._requested_hash = ""
        self._applied_hash = ""
        self._applied_at_ns = 0
        self._presented_at_ns = 0
        self.image_applied.set()
        self.offset_applied.set()

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setStyleSheet("background-color: black;")
        self.setGeometry(screen.geometry())
        self.label = QtWidgets.QLabel(self)
        self.label.setFixedSize(*self.modulated_size)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self._center_label()
        self.update_image_signal.connect(self._apply_image)
        self.update_offset_signal.connect(self._apply_offset)

    def present(self) -> None:
        """Put the output surface on its physical screen and cover the desktop."""
        # A top-level QWidget is initially associated with the primary screen on
        # some Windows/Qt combinations, even when its geometry belongs to a
        # secondary screen. Associate the native window explicitly before
        # switching it to full-screen mode.
        self.setGeometry(self.screen.geometry())
        self.show()
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(self.screen)
        self.setGeometry(self.screen.geometry())
        self.showFullScreen()
        self._center_label()
        self.label.show()
        self.label.raise_()
        self.raise_()
        self.repaint()
        with self._display_state_lock:
            self._presented_at_ns = time.time_ns()

    def _center_label(self) -> None:
        width, height = self.modulated_size
        self.label.move(
            (self.width() - width) // 2 + self.x_offset,
            (self.height() - height) // 2 + self.y_offset,
        )

    def offset_limits(self) -> Tuple[int, int]:
        width, height = self.modulated_size
        return max(0, (self.width() - width) // 2), max(0, (self.height() - height) // 2)

    @QtCore.pyqtSlot(int, int)
    def _apply_offset(self, x_offset: int, y_offset: int) -> None:
        max_x, max_y = self.offset_limits()
        self.x_offset = max(-max_x, min(max_x, int(x_offset)))
        self.y_offset = max(-max_y, min(max_y, int(y_offset)))
        self._center_label()
        self.label.raise_()
        self.label.repaint()
        self.repaint()
        self.offset_applied.set()

    @QtCore.pyqtSlot(object)
    def _apply_image(self, request: object) -> None:
        if isinstance(request, tuple) and len(request) == 2:
            sequence, image = request
        else:  # Backward compatibility for direct signal users.
            sequence, image = self._requested_sequence, request
        array = np.ascontiguousarray(image, dtype=np.uint8)
        if array.ndim != 2:
            raise ValueError("modulator image must be a 2-D grayscale array")
        height, width = array.shape
        qimage = QtGui.QImage(
            array.data, width, height, width, QtGui.QImage.Format_Grayscale8
        ).copy()
        scaled = qimage.scaled(
            self.modulated_size[0],
            self.modulated_size[1],
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        self.label.setPixmap(QtGui.QPixmap.fromImage(scaled))
        # repaint() is synchronous at the QWidget level. The manager follows
        # this with a Qt/Windows event flush when it owns the GUI thread, so an
        # "applied" event means more than merely assigning a QPixmap in memory.
        self.label.show()
        self.label.raise_()
        self.label.repaint()
        self.repaint()
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        with self._display_state_lock:
            self._applied_sequence = max(self._applied_sequence, int(sequence))
            self._applied_hash = digest
            self._applied_at_ns = time.time_ns()
        self.image_applied.set()

    def update_image(self, image: np.ndarray) -> None:
        array = np.ascontiguousarray(image, dtype=np.uint8)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        with self._display_state_lock:
            self._requested_sequence += 1
            sequence = self._requested_sequence
            self._requested_hash = digest
        self.image_applied.clear()
        self.update_image_signal.emit((sequence, array))

    def update_offset(self, x_offset: int, y_offset: int) -> None:
        self.offset_applied.clear()
        self.update_offset_signal.emit(int(x_offset), int(y_offset))

    def wait_until_applied(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._display_state_lock:
                complete = (
                    self._applied_sequence >= self._requested_sequence
                    and self._applied_hash == self._requested_hash
                )
            if complete and self.offset_applied.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            # An older queued frame may have set the event. Clear it, then
            # re-check under the state lock so a newer completion cannot be
            # mistaken for (or lost behind) that older notification.
            self.image_applied.clear()
            with self._display_state_lock:
                complete = (
                    self._applied_sequence >= self._requested_sequence
                    and self._applied_hash == self._requested_hash
                )
            if complete:
                continue
            self.image_applied.wait(min(remaining, 0.05))

    def display_status(self) -> dict:
        """Return thread-safe evidence for the most recently requested frame."""
        with self._display_state_lock:
            requested_sequence = self._requested_sequence
            applied_sequence = self._applied_sequence
            requested_hash = self._requested_hash
            applied_hash = self._applied_hash
            applied_at_ns = self._applied_at_ns
            presented_at_ns = self._presented_at_ns
        geometry = self.screen.geometry()
        return {
            "device": self.device_name,
            "screen_name": self.screen.name(),
            "screen_geometry": [
                int(geometry.x()),
                int(geometry.y()),
                int(geometry.width()),
                int(geometry.height()),
            ],
            "modulated_size": list(map(int, self.modulated_size)),
            "requested_sequence": requested_sequence,
            "applied_sequence": applied_sequence,
            "requested_sha256": requested_hash,
            "applied_sha256": applied_hash,
            "hash_matches": bool(requested_hash and requested_hash == applied_hash),
            "presented_at_ns": presented_at_ns,
            "applied_at_ns": applied_at_ns,
        }

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        event.ignore()

"""Shared Qt-binding detection for the *_gui_qt modules.

Tries PyQt6 → PyQt5 → PySide6 → PySide2 in that order and re-exports a
fixed set of widgets/enums so each Qt GUI module can write::

    from spikedetect.gui._qt_imports import QApplication, QDialog, ...

without repeating the try/except chain. ``filter_gui_qt.py`` has its own
local detection block predating this module; future cleanup could route it
through here too.
"""

from __future__ import annotations


try:
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QWidget, QSizePolicy,
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer
    from PyQt6.QtGui import QKeyEvent
    QT_BINDING = "PyQt6"
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
            QLabel, QPushButton, QWidget, QSizePolicy,
        )
        from PyQt5.QtCore import Qt, pyqtSignal, QTimer
        from PyQt5.QtGui import QKeyEvent
        QT_BINDING = "PyQt5"
    except ImportError:
        try:
            from PySide6.QtWidgets import (
                QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                QLabel, QPushButton, QWidget, QSizePolicy,
            )
            from PySide6.QtCore import Qt, Signal as pyqtSignal, QTimer
            from PySide6.QtGui import QKeyEvent
            QT_BINDING = "PySide6"
        except ImportError:
            try:
                from PySide2.QtWidgets import (
                    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                    QLabel, QPushButton, QWidget, QSizePolicy,
                )
                from PySide2.QtCore import Qt, Signal as pyqtSignal, QTimer
                from PySide2.QtGui import QKeyEvent
                QT_BINDING = "PySide2"
            except ImportError as exc:
                raise ImportError(
                    "No Qt binding found. Install one of "
                    "PyQt6, PyQt5, PySide6, PySide2."
                ) from exc


__all__ = [
    "QApplication", "QDialog", "QVBoxLayout", "QHBoxLayout", "QGridLayout",
    "QLabel", "QPushButton", "QWidget", "QSizePolicy",
    "Qt", "pyqtSignal", "QTimer", "QKeyEvent",
    "QT_BINDING",
]

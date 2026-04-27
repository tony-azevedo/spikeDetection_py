"""Interactive Matplotlib GUI components for spike detection
parameter tuning."""

from spikedetect.gui.filter_gui import FilterGUI
from spikedetect.gui.template_gui import TemplateSelectionGUI
from spikedetect.gui.threshold_gui import ThresholdGUI
from spikedetect.gui.spotcheck_gui import SpotCheckGUI
from spikedetect.gui.workflow import InteractiveWorkflow

__all__ = [
    "FilterGUI",
    "FilterGUIQt",
    "InteractiveWorkflow",
    "TemplateSelectionGUI",
    "TemplateSelectionGUIQt",
    "ThresholdGUI",
    "ThresholdGUIQt",
    "SpotCheckGUI",
    "SpotCheckGUIQt",
]


_QT_GUIS = {
    "FilterGUIQt": "spikedetect.gui.filter_gui_qt",
    "TemplateSelectionGUIQt": "spikedetect.gui.template_gui_qt",
    "ThresholdGUIQt": "spikedetect.gui.threshold_gui_qt",
    "SpotCheckGUIQt": "spikedetect.gui.spotcheck_gui_qt",
}


def __getattr__(name):
    """Lazy import: Qt GUIs only pull in PyQt/PySide on first access.

    Lets `import spikedetect.gui` succeed on systems without a Qt binding
    while still exposing `from spikedetect.gui import FilterGUIQt` etc.
    """
    module_path = _QT_GUIS.get(name)
    if module_path is not None:
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

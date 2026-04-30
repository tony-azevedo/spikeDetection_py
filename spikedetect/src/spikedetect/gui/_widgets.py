"""Shared helper functions for Matplotlib-based GUI components."""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class UserCancelled(Exception):
    """Raised by a GUI's ``run()`` when the user cancels with Esc.

    Pipelines can wrap ``gui.run()`` in ``try: ... except UserCancelled:``
    to bail out cleanly. Closing the window is *not* a cancel — only Esc
    raises this. In the Qt GUIs the Cancel button and the X close also
    don't raise; they fall through with the (deep-copied) initial params,
    matching the matplotlib close-as-accept behavior.
    """


def raster_ticks(
    ax: plt.Axes,
    locations: np.ndarray,
    y_pos: float,
    tick_height: float | None = None,
    picker: float | None = None,
    **kwargs,
) -> list:
    """Draw raster tick marks at given locations on an axis.

    Args:
        ax: The axes to draw on.
        locations: 1-D array of x-positions for tick marks.
        y_pos: The y-coordinate for the base of the ticks.
        tick_height: Height of each tick mark. If None,
            defaults to 2% of the current y-axis range.
        picker: If not None, set the picker tolerance on the
            returned LineCollection to enable pick events.
            Default None (not pickable).
        **kwargs: Additional keyword arguments passed to
            ``ax.vlines``.

    Returns:
        List of LineCollection objects created.
    """
    locations = np.asarray(locations).ravel()
    if len(locations) == 0:
        return []

    if tick_height is None:
        ylim = ax.get_ylim()
        tick_height = 0.02 * abs(ylim[1] - ylim[0])
        if tick_height == 0:
            tick_height = 0.1

    defaults = {"colors": "k", "linewidths": 0.5}
    defaults.update(kwargs)

    lines = ax.vlines(locations, y_pos, y_pos + tick_height, **defaults)
    if picker is not None:
        lines.set_picker(picker)
    return [lines]


def install_finish_handlers(
    fig: Figure,
    on_key: Callable[[str], None],
    on_close: Callable[[], None],
) -> Callable[[], None]:
    """Connect key-press and close handlers without starting an event loop.

    Companion to :func:`blocking_wait` for non-blocking embedding.
    The caller is responsible for driving the event loop (either by
    calling :func:`blocking_wait` afterwards for standalone use, or
    by letting a host Qt/Tk app drive it).

    Args:
        fig: The figure to attach handlers to.
        on_key: Called with ``event.key`` on every key press.
        on_close: Called with no arguments when the figure is closed.

    Returns:
        A zero-argument function that disconnects both handlers.
    """
    def _on_key(event):
        on_key(event.key)

    def _on_close(_event):
        on_close()

    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)
    cid_close = fig.canvas.mpl_connect("close_event", _on_close)

    def disconnect() -> None:
        try:
            fig.canvas.mpl_disconnect(cid_key)
            fig.canvas.mpl_disconnect(cid_close)
        except Exception:
            pass

    return disconnect


def blocking_wait(fig: Figure) -> str | None:
    """Block until user presses a key or closes the figure.

    Works in both interactive (Qt/Tk) and notebook (ipympl)
    modes.

    Args:
        fig: The figure to wait on.

    Returns:
        The key pressed, or None if the figure was closed.
    """
    result = {"key": None}

    def _on_key(event):
        result["key"] = event.key
        fig.canvas.stop_event_loop()

    def _on_close(event):
        result["key"] = None
        fig.canvas.stop_event_loop()

    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)
    cid_close = fig.canvas.mpl_connect("close_event", _on_close)

    try:
        fig.show()
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(timeout=0)
    except Exception:
        # Fallback: use plt.waitforbuttonpress
        try:
            pressed = plt.waitforbuttonpress()
            if pressed:
                result["key"] = "enter"
        except Exception:
            pass
    finally:
        fig.canvas.mpl_disconnect(cid_key)
        fig.canvas.mpl_disconnect(cid_close)

    return result["key"]

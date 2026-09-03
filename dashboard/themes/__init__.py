"""Interchangeable style definitions for the dashboard.

A Theme is a plain bag of color/typography tokens with no plotting or UI
imports, so the same theme feeds plotly templates, CSS variables, or any
future front-end. Add a new style by appending to THEMES (or constructing
a Theme at runtime); nothing else changes.

Palette discipline follows the repo's data-viz method: categorical hues
are a fixed 8-slot order (identity: jump types), ordered series
(temperatures, ladder-update history) use a single-hue ordinal ramp, and
status colors are reserved for run state, never for data series.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Color and typography tokens for one dashboard style."""

    name: str
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    gridline: str
    baseline: str
    border: str
    # fixed-order categorical slots (identity encoding, never cycled)
    categorical: tuple[str, ...]
    # single-hue ramp for continuous magnitude (heatmaps), low -> high,
    # ordered so the low end recedes toward the surface
    sequential_ramp: tuple[str, ...]
    # clipped ramp for discrete ordered series (every step readable)
    ordinal_ramp: tuple[str, ...]
    # highlight for the 'current' / live entity (categorical slot, reserved)
    highlight: str
    # status colors (run state chips only, never data series)
    status_good: str
    status_warning: str
    status_serious: str
    status_critical: str
    font_family: str = 'system-ui, -apple-system, "Segoe UI", sans-serif'


LIGHT = Theme(
    name='light',
    surface='#fcfcfb',
    page='#f9f9f7',
    ink_primary='#0b0b0b',
    ink_secondary='#52514e',
    ink_muted='#898781',
    gridline='#e1e0d9',
    baseline='#c3c2b7',
    border='rgba(11,11,11,0.10)',
    categorical=('#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834'),
    sequential_ramp=(
        '#cde2fb',
        '#b7d3f6',
        '#9ec5f4',
        '#86b6ef',
        '#6da7ec',
        '#5598e7',
        '#3987e5',
        '#2a78d6',
        '#256abf',
        '#1c5cab',
        '#184f95',
        '#104281',
        '#0d366b',
    ),
    ordinal_ramp=(
        '#86b6ef',
        '#6da7ec',
        '#5598e7',
        '#3987e5',
        '#2a78d6',
        '#256abf',
        '#1c5cab',
        '#184f95',
        '#104281',
        '#0d366b',
    ),
    highlight='#e34948',
    status_good='#0ca30c',
    status_warning='#fab219',
    status_serious='#ec835a',
    status_critical='#d03b3b',
)

DARK = Theme(
    name='dark',
    surface='#1a1a19',
    page='#0d0d0d',
    ink_primary='#ffffff',
    ink_secondary='#c3c2b7',
    ink_muted='#898781',
    gridline='#2c2c2a',
    baseline='#383835',
    border='rgba(255,255,255,0.10)',
    categorical=('#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926'),
    sequential_ramp=(
        '#0d366b',
        '#104281',
        '#184f95',
        '#1c5cab',
        '#256abf',
        '#2a78d6',
        '#3987e5',
        '#5598e7',
        '#6da7ec',
        '#86b6ef',
        '#9ec5f4',
        '#b7d3f6',
        '#cde2fb',
    ),
    ordinal_ramp=(
        '#184f95',
        '#1c5cab',
        '#256abf',
        '#2a78d6',
        '#3987e5',
        '#5598e7',
        '#6da7ec',
        '#86b6ef',
        '#9ec5f4',
        '#b7d3f6',
        '#cde2fb',
    ),
    highlight='#e66767',
    status_good='#0ca30c',
    status_warning='#fab219',
    status_serious='#ec835a',
    status_critical='#d03b3b',
)

THEMES: dict[str, Theme] = {'light': LIGHT, 'dark': DARK}


def get_theme(name: str) -> Theme:
    """Look up a theme by name, defaulting to light for unknown names."""
    return THEMES.get(name, LIGHT)

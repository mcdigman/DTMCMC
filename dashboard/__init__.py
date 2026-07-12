"""Asynchronous telemetry dashboard for DTMCMC run artifacts.

The dashboard is a separate process from the sampler: it communicates only
through the HDF5 run artifacts the harness already flushes atomically at
every checkpoint, so a dashboard crash or lag cannot affect a live run.

Layering (see dashboard/README.md for the full design):

- ``dashboard.core``: framework-agnostic artifact reading and diagnostics.
  Pure numpy in, plot-ready series out. No UI or plotting imports.
- ``dashboard.figures``: thin plotly figure factories over core outputs,
  themed via ``dashboard.themes``.
- ``dashboard.app``: the Dash front-end (layout, callbacks, live polling).
  Interchangeable: a Panel/bokeh front-end can reuse core and figures.
"""

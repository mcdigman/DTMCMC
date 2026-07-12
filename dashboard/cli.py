"""Command-line entry point: ``python -m dashboard <artifact-or-dir>``.

The server binds localhost by default. For a run on a remote machine,
start the dashboard there and tunnel the port to your laptop, e.g.::

    ssh -N -L 8050:localhost:8050 user@server

then open http://localhost:8050 — only plotly JSON crosses the tunnel,
never the artifact files themselves.
"""

import argparse
from pathlib import Path

from dashboard.app.dash_app import DashboardConfig, run_dashboard


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog='dashboard', description='DTMCMC telemetry dashboard (live monitoring or post-hoc analysis of run artifacts)')
    parser.add_argument('artifact', type=Path, help='run artifact .h5 file, or a directory to browse')
    parser.add_argument('--host', default='127.0.0.1', help='bind address (default localhost; use an ssh tunnel for remote viewing)')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--poll', type=float, default=5., metavar='SECONDS', help='artifact poll interval (default 5)')
    parser.add_argument('--theme', default='light', help='initial theme name (light or dark)')
    parser.add_argument('--layout', default='default', help='tab layout name from figures.registry.LAYOUTS')
    parser.add_argument('--stale-after', type=float, default=600., metavar='SECONDS', help='flag a non-finalized artifact as stale after this flush age')
    parser.add_argument('--no-store', action='store_true', help='skip loading the sample store (lighter polling; posterior tab empty)')
    parser.add_argument('--debug', action='store_true', help='run the Dash dev server with hot reload')
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and serve the dashboard."""
    args = build_parser().parse_args(argv)
    run_dashboard(DashboardConfig(
        artifact=args.artifact,
        host=args.host,
        port=args.port,
        poll_seconds=args.poll,
        theme=args.theme,
        layout=args.layout,
        stale_after_seconds=args.stale_after,
        load_store=not args.no_store,
        debug=args.debug,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

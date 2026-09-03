"""Phase 4 pilot-calibration batteries (methods-paper plan §4, exploratory).

Each pilot module is a CLI that generates run specs, executes them one
process per run through the harness CLI (plan D1: one seeding per
process), analyzes the resulting artifacts post-hoc with Generator-only
code (plan D5), and emits a JSON summary consumed by
.plans/pilot-report.md. Artifacts and summaries land under
artifacts/pilots/ (gitignored); only the report and the code are
committed.
"""

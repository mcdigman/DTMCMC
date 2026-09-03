"""Shared plumbing for the adaptive convergence batteries (not a test module).

The spec builder, post-freeze loading, and structural checks all live in
experiments/ (pilots.common.make_adaptive_spec and harness.postfreeze) so
tests, pilots, and production analyses read runs and build adaptive specs
identically; this module only re-exports them under the battery names and
adds the assert wrapper.
"""

from typing import Any

from experiments.harness.postfreeze import load_post_freeze, readout_structure_violations
from experiments.pilots.common import ADAPTIVE_DE_WINDOW_BLOCKS, make_adaptive_spec

__all__ = ['DE_WINDOW_BLOCKS', 'adaptive_spec_data', 'assert_readout_structure', 'load_post_freeze']

DE_WINDOW_BLOCKS = ADAPTIVE_DE_WINDOW_BLOCKS

adaptive_spec_data = make_adaptive_spec


def assert_readout_structure(run: dict[str, Any], *, require_below_readout: bool = True) -> None:
    """Assert the shared structural conditions on a post-freeze run dict."""
    violations = readout_structure_violations(run, require_below_readout=require_below_readout)
    assert not violations, violations

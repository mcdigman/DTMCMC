"""View options shared by every figure factory.

One flat dataclass so front-ends can map UI controls onto figure inputs
without knowing which figure consumes which field.
"""

from dataclasses import dataclass, field


@dataclass
class ViewOptions:
    """User-adjustable view state consumed by the figure builders."""

    # blocks trimmed from the start of block-history plots (burn-in)
    burnin_blocks: int = 1
    # counting window for tracker-rate plots: 'segment' (since the last
    # applied ladder update), 'total', 'latest', or an archive index
    window: str | int = 'segment'
    # plot every nth ladder-history curve (declutters long adaptive runs)
    segment_stride: int = 1
    # ESD normalization: per accepted jump instead of per proposal
    accepted_only: bool = False
    # recorded-chain selections for posterior/correlation plots
    chain: int = 0
    chains: list[int] = field(default_factory=lambda: [0])
    # parameter dimensions for corner/trace/ACF plots (subset: corner plots
    # cannot show every dimension of a high-dimensional likelihood)
    dims: list[int] = field(default_factory=lambda: [0, 1, 2])
    # burn-in trim for stored-sample plots, in stored rows
    burnin_rows: int = 0
    max_lag: int = 256
    corner_max_points: int = 20000
    # trailing window (blocks) for flow-fraction averaging; 0 = all blocks
    trailing_blocks: int = 0
    # recorded-chain pair for the cross-correlation plot
    cross_a: int = 0
    cross_b: int = 0

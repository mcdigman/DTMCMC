"""Deterministic likelihood-evaluation accounting, separated by source.

Likelihood objects are stateless, so evaluation counts are assembled by the
sampler from deterministic declarations rather than threaded mutable
counters: the orchestration loop counts proposal-target evaluations exactly
(the one runtime-variable phase, since dispatch or bounds validation can
short-circuit), jumps declare a fixed per-dispatch internal cost via the
``declared_internal_evals`` attribute, and managers report scheduled
maintenance costs from ``declared_construction_evals`` and the return value
of ``post_block_update``. A component that cannot declare its cost yields an
incomplete total — never a silent zero.

``LoglikeCallSpy`` is the conformance helper for verifying declarations:
it independently counts actual ``get_loglike`` calls on one likelihood
instance, so tests (including third-party extension tests) can assert that
declared costs equal observed calls.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from DTMCMC.likelihood import AbstractLikelihood


@dataclass
class EvalAccounting[LikelihoodType: AbstractLikelihood = AbstractLikelihood]:
    """Likelihood-evaluation totals by source phase.

    ``complete`` is False when any component could not declare its cost
    (an unknown is never silently treated as zero); consumers must not
    present an incomplete total as an exact evaluation count.
    """

    initialization: int = 0
    proposal_targets: int = 0
    proposal_internal: int = 0
    post_block: int = 0
    complete: bool = True

    @property
    def total(self) -> int:
        """Total likelihood evaluations across all phases."""
        return self.initialization + self.proposal_targets + self.proposal_internal + self.post_block


class LoglikeCallSpy[LikelihoodType: AbstractLikelihood = AbstractLikelihood]:
    """Independently count actual ``get_loglike`` calls on one likelihood.

    Context manager that wraps the instance's ``get_loglike`` with a
    counting shim (restored on exit). Use it to verify that declared
    evaluation costs match observed behavior::

        with LoglikeCallSpy(like_obj) as spy:
            manager.post_block_update(itrn, block_size, samples, logLs)
        assert spy.n_calls == declared_cost
    """

    def __init__(self, like_obj: LikelihoodType) -> None:
        self.like_obj: LikelihoodType = like_obj
        self.n_calls: int = 0

    def __enter__(self) -> LoglikeCallSpy[LikelihoodType]:
        original: Callable[[NDArray[np.floating]], float] = self.like_obj.get_loglike

        def counting_loglike(params_in: NDArray[np.floating]) -> float:
            self.n_calls += 1
            return original(params_in)

        # instance attribute shadows the class method until __exit__
        self.like_obj.get_loglike = counting_loglike  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc_info: object) -> None:
        del self.like_obj.get_loglike

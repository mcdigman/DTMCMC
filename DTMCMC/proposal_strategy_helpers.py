"""C 2023 Matthew C. Digman
hold some helpers to help determine the proposal strategy
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configparser import ConfigParser


class ProposalStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        self.config: ConfigParser = config

    def copy(self) -> ProposalStrategyParameters:
        """Copy the object"""
        # TODO make global config captureable
        return ProposalStrategyParameters(self.config)

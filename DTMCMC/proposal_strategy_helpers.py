"""C 2023 Matthew C. Digman
hold some helpers to help determine the proposal strategy
"""

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configparser import ConfigParser


@dataclass(init=False)
class ProposalStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        del config

    def copy(self) -> ProposalStrategyParameters:
        """Copy the object"""
        return copy(self)

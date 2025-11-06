"""C 2023 Matthew C. Digman
hold some helpers to help determine the proposal strategy
"""


class ProposalStrategyParameters():
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config):
        """Initialize the object with the prescribed parameters"""
        self.config = config

    def copy(self):
        """Copy the object"""
        # TODO make global config captureable
        return ProposalStrategyParameters(self.config)

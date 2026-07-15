"""Tests for ConfigParser-backed proposal strategy dataclasses."""

import configparser
from dataclasses import is_dataclass

import pytest

from DTMCMC.auxilliary_manager import AuxilliaryStrategyParameters
from DTMCMC.de_manager import DEStrategyParameters
from DTMCMC.fisher_manager import FisherStrategyParameters
from DTMCMC.history_jump_manager import HistoryStrategyParameters
from DTMCMC.prior_manager import PriorStrategyParameters
from DTMCMC.proposal_strategy_helpers import ProposalStrategyParameters


def _config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read('default_config.ini')
    return config


@pytest.mark.parametrize(
    'strategy_type',
    [
        DEStrategyParameters,
        FisherStrategyParameters,
        PriorStrategyParameters,
        AuxilliaryStrategyParameters,
        HistoryStrategyParameters,
        ProposalStrategyParameters,
    ],
)
def test_strategy_parameters_are_independent_copyable_dataclasses(strategy_type) -> None:
    """Strategy snapshots no longer retain a live ConfigParser as hidden state."""
    strategy = strategy_type(_config())
    copied = strategy.copy()

    assert is_dataclass(strategy)
    assert copied is not strategy
    assert vars(copied) == vars(strategy)
    assert 'config' not in vars(strategy)


def test_strategy_copy_preserves_runtime_overrides() -> None:
    """Copying preserves values changed after ConfigParser construction."""
    strategy = DEStrategyParameters(_config())
    strategy.de_thin = 17
    copied = strategy.copy()

    assert copied.de_thin == 17
    copied.de_thin = 3
    assert strategy.de_thin == 17


def test_strategy_record_config_uses_dataclass_values() -> None:
    """Existing configuration serialization reads the explicit fields."""
    config = _config()
    strategy = PriorStrategyParameters(config)
    strategy.cold_prior_weight = 0.125
    strategy.hot_prior_target_weight = 0.875
    strategy.record_config(config)

    assert config['PriorManager'].getfloat('cold_prior_weight') == 0.125
    assert config['PriorManager'].getfloat('hot_prior_target_weight') == 0.875

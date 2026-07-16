"""RunSpec: the serializable description of a single sampler run.

A spec is TOML on disk and a frozen dataclass in memory. It captures
everything the runner needs to reconstruct a run: likelihood name + params,
ladder spec, proposal mixture overrides (mapped onto the existing
ConfigParser sections), exchange strategy, run geometry (steps, blocks,
storage/thinning), and the run seed. The fully resolved spec text is
embedded verbatim in the run artifact (plan D2).

TOML reading uses stdlib tomllib; writing uses a minimal emitter for the
restricted spec schema (scalars, lists of scalars, nested tables), verified
by round-trip against tomllib in the tests.
"""

import configparser
import io
import json
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import DTMCMC.exchange_manager as em

from .paths import default_config_path

TomlScalar = str | int | float | bool
TomlValue = TomlScalar | list[str] | list[int] | list[float] | list[bool]

# maps spec strings onto the integer strategy codes in DTMCMC.exchange_manager
EXCHANGE_STRATEGY_CODES: dict[str, int] = {
    'random': em.RANDOM_TARGETS,
    'sequential': em.SEQUENTIAL_TARGETS,
    'adjacent': em.ADJACENT_TARGETS,
    'null': em.NULL_TARGETS,
    'reverse_sequential': em.REVERSE_SEQUENTIAL_TARGETS,
    'alternate_sequential': em.ALTERNATE_SEQUENTIAL_TARGETS,
}

LIKELIHOOD_NAMES: frozenset[str] = frozenset(
    {
        'gaussian',
        'cake',
        'constant_rectangular',
        'eggbox',
        'hawaii',
        'ar1',
        'banana',
        'gaussian_mixture',
        'gaussian_shell',
        'hyperpyramid',
        'random_wheel',
        'rosenbrock',
        'spoke_wheel',
        'uniform_gaussian_prior',
    }
)

LADDER_KINDS: frozenset[str] = frozenset({'geometric', 'entropy_file', 'length_file', 'acceptance_file', 'explicit'})

# the [run] table's full key set: like [adaptive], an unknown key must fail
# loudly rather than be silently dropped. This closes the gap where a legacy
# run.n_record spec parsed successfully and silently recorded nothing extra
# (its recording intent lost) after n_record was replaced by arg_record.
RUN_KEYS: frozenset[str] = frozenset(
    {
        'n_steps',
        'n_steps_per_major_report',
        'block_size',
        'store_thin',
        'arg_record',
        'checkpoint_every_blocks',
        'zero_loglike',
    }
)

# the ConfigParser sections the proposal mixture maps onto
PROPOSAL_SECTIONS: frozenset[str] = frozenset(
    {
        'FisherJumpManager',
        'DEJumpManager',
        'PriorManager',
        'ProposalManager',
        'AuxilliaryJumpManager',
        'LadderHistoryJumpManager',
    }
)

# adaptive controller modes; a test pins this to experiments.adaptive's
# ADAPTIVE_MODES (spec stays a pure-data layer, so no runtime import)
ADAPTIVE_MODES: frozenset[str] = frozenset({'entropy', 'length', 'acceptance'})

# the [adaptive] table's full key set: these knobs feed the paper, so a
# typo must fail loudly rather than silently run with a default (see
# build_adaptive_controller for semantics and defaults)
ADAPTIVE_KEYS: frozenset[str] = frozenset(
    {
        'mode',
        'update_every_blocks',
        'forgetting',
        'freeze_dlog',
        'freeze_consecutive',
        'remap_rule',
        'T_min_factor',
        'budget_blocks',
        'var_estimator',
        'n_prior_draws',
        'min_updates_at_target',
        'window_extension_factor',
        'ds_link_cap',
        'cold_cap_links',
        'cap_ratio_min',
        'cap_ratio_max',
        'var_history_length',
        'pool_dlog_tol',
        'discard_blocks_after_update',
    }
)

_BARE_KEY_RE = re.compile(r'^[A-Za-z0-9_-]+$')


class SpecError(ValueError):
    """Raised when a run spec fails validation."""

    def __init__(self, detail: str) -> None:
        super().__init__(f'invalid run spec: {detail}')


def config_to_text(config: configparser.ConfigParser) -> str:
    """Serialize a ConfigParser instance as INI text."""
    buffer = io.StringIO()
    config.write(buffer)
    return buffer.getvalue()


def _require_table(data: dict[str, object], key: str) -> dict[str, object]:
    """Fetch a required sub-table from parsed TOML data."""
    value = data.get(key)
    if not isinstance(value, dict):
        msg = f'missing or non-table [{key}] section'
        raise SpecError(msg)
    return dict(value)


def _require_str(table: dict[str, object], key: str, ctx: str) -> str:
    """Fetch a required string entry from parsed TOML data."""
    value = table.get(key)
    if not isinstance(value, str):
        msg = f'{ctx}.{key} must be a string'
        raise SpecError(msg)
    return value


def _require_int(table: dict[str, object], key: str, ctx: str) -> int:
    """Fetch a required integer entry from parsed TOML data."""
    value = table.get(key)
    # bool is an int subclass; reject it explicitly
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f'{ctx}.{key} must be an integer'
        raise SpecError(msg)
    return value


def _opt_int(table: dict[str, object], key: str, ctx: str, default: int) -> int:
    """Fetch an optional integer entry from parsed TOML data."""
    if key not in table:
        return default
    return _require_int(table, key, ctx)


def _opt_int_list(table: dict[str, object], key: str, ctx: str) -> list[int]:
    """Fetch an optional list-of-integers entry from parsed TOML data (default empty)."""
    if key not in table:
        return []
    value = table.get(key)
    if not isinstance(value, list):
        msg = f'{ctx}.{key} must be a list of integers'
        raise SpecError(msg)
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            msg = f'{ctx}.{key} entries must be integers'
            raise SpecError(msg)
    return list(value)


def _opt_bool(table: dict[str, object], key: str, ctx: str, default: bool) -> bool:
    """Fetch an optional boolean entry from parsed TOML data."""
    value = table.get(key, default)
    if not isinstance(value, bool):
        msg = f'{ctx}.{key} must be a boolean'
        raise SpecError(msg)
    return value


def _check_toml_value(value: object, ctx: str) -> TomlValue:
    """Check that a parsed value is a spec-schema scalar or list of scalars."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, bool | int | float | str):
                msg = f'{ctx} list entries must be scalars'
                raise SpecError(msg)
        return value
    msg = f'{ctx} must be a scalar or a list of scalars'
    raise SpecError(msg)


def _format_toml_key(key: str) -> str:
    """Format a table key, quoting it if it is not a bare key."""
    if _BARE_KEY_RE.match(key):
        return key
    return json.dumps(key)


def _format_toml_scalar(value: TomlScalar) -> str:
    """Format one scalar as TOML source text."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'nan'
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
        text = repr(value)
        # TOML floats need a decimal point or exponent
        if 'e' not in text and '.' not in text:
            text += '.0'
        return text
    # JSON string escaping is a subset of TOML basic-string escaping
    return json.dumps(value)


def _format_toml_value(value: TomlValue) -> str:
    """Format one scalar or list value as TOML source text."""
    if isinstance(value, list):
        return '[' + ', '.join(_format_toml_scalar(item) for item in value) + ']'
    return _format_toml_scalar(value)


def _emit_toml_table(lines: list[str], table: dict[str, object], prefix: str) -> None:
    """Emit one table (scalars first, then sub-tables) into lines."""
    subtables: list[tuple[str, dict[str, object]]] = []
    for key, value in table.items():
        if isinstance(value, dict):
            subtables.append((key, value))
        else:
            lines.append(f'{_format_toml_key(key)} = {_format_toml_value(_check_toml_value(value, key))}')

    for key, subtable in subtables:
        full_key = f'{prefix}.{_format_toml_key(key)}' if prefix else _format_toml_key(key)
        lines.append('')
        lines.append(f'[{full_key}]')
        _emit_toml_table(lines, subtable, full_key)


def dumps_toml(data: dict[str, object]) -> str:
    """Serialize a nested dict of scalars/lists/tables as TOML text.

    Minimal emitter for the spec schema only (no dates, no arrays of
    tables); round-trip against tomllib is covered by the tests.
    """
    lines: list[str] = []
    _emit_toml_table(lines, data, '')
    return '\n'.join(lines).lstrip('\n') + '\n'


@dataclass(frozen=True)
class RunSpec:
    """Full description of a single sampler run.

    Parameters
    ----------
    name: str
        Short identifier; used in artifact file names
    seed: int
        Run seed; derives both child stream seeds (plan D1)
    likelihood_name: str
        One of LIKELIHOOD_NAMES; maps to a DTMCMC.likelihoods class
    likelihood_params: dict[str, TomlValue]
        Constructor kwargs for the likelihood (e.g. n_par, cutoff)
    ladder: dict[str, TomlValue]
        Ladder spec: 'kind' (LADDER_KINDS), 'n_chain', 'n_cold', plus
        kind-specific constructor parameters
    n_steps: int
        Total iterations (each advances all chains once); multiple of block_size
    n_steps_per_major_report: int
        Iterations between "major report" boundaries at which the sampler
        emits a final-style tracker summary and marks the artifact
        finalized; a positive multiple of block_size. Defaults to n_steps
        (one major report at the end). This is an interval, not a total:
        the sampler consumes it so it can flag major reports periodically
        without knowing how many iterations it will ultimately be run for
        (parent-sampler design principle — runs are indefinite).
    block_size: int
        Iterations per block
    store_thin: int
        Thinning applied to the stored cold-chain samples
    arg_record: list[int]
        Indices of additional chains recorded in storage beyond the
        ladder's n_cold readout chains (default none). The readout chains
        always occupy the first n_cold store columns and their indices
        are recomputed at every ladder update; duplicates are kept
    checkpoint_every_blocks: int
        Artifact flush cadence in blocks
    zero_loglike: bool
        Whether sampler target log-likelihood values are forced to zero.
        Proposal-internal calculations continue to use the configured
        likelihood. Embedded in artifacts as run provenance.
    exchange_strategy: str
        One of EXCHANGE_STRATEGY_CODES
    track_full_exchanges: bool
        Whether the tracker keeps the full exchange matrix
    proposal_overrides: dict[str, dict[str, TomlValue]]
        Per-section overrides applied on top of default_config.ini
    """

    name: str
    seed: int
    likelihood_name: str
    likelihood_params: dict[str, TomlValue] = field(default_factory=dict)
    ladder: dict[str, TomlValue] = field(default_factory=dict)
    n_steps: int = 0
    n_steps_per_major_report: int = 0
    block_size: int = 0
    store_thin: int = 1
    arg_record: list[int] = field(default_factory=list)
    checkpoint_every_blocks: int = 8
    zero_loglike: bool = False
    exchange_strategy: str = 'sequential'
    track_full_exchanges: bool = False
    proposal_overrides: dict[str, dict[str, TomlValue]] = field(default_factory=dict)
    adaptive: dict[str, TomlValue] | None = None

    def __post_init__(self) -> None:
        """Validate cross-field constraints."""
        if self.likelihood_name not in LIKELIHOOD_NAMES:
            msg = f'unknown likelihood {self.likelihood_name!r}; known: {sorted(LIKELIHOOD_NAMES)}'
            raise SpecError(msg)
        if 'name' in self.likelihood_params:
            msg = "likelihood params may not include 'name'"
            raise SpecError(msg)

        kind = self.ladder.get('kind')
        if kind not in LADDER_KINDS:
            msg = f'unknown ladder kind {kind!r}; known: {sorted(LADDER_KINDS)}'
            raise SpecError(msg)
        n_chain = self.n_chain
        n_cold = self.n_cold
        if n_cold < 1:
            msg = 'ladder.n_cold must be >= 1'
            raise SpecError(msg)
        if n_chain < n_cold:
            msg = 'ladder.n_chain must be >= ladder.n_cold'
            raise SpecError(msg)
        if kind == 'explicit':
            # the actual ladder is built from Ts, so a mismatch with the
            # declared n_chain would silently run the wrong chain count
            # and embed contradictory provenance (PR #9 review)
            Ts_raw = self.ladder.get('Ts')
            if (
                not isinstance(Ts_raw, list)
                or not Ts_raw
                or not all(isinstance(T_loc, int | float) and not isinstance(T_loc, bool) for T_loc in Ts_raw)
            ):
                msg = 'explicit ladder requires a non-empty numeric ladder.Ts list'
                raise SpecError(msg)
            if len(Ts_raw) != n_chain:
                msg = f'explicit ladder.Ts has {len(Ts_raw)} entries but ladder.n_chain is {n_chain}'
                raise SpecError(msg)

        if self.block_size < 2:
            msg = 'run.block_size must be >= 2 (blocks alternate regular and exchange steps)'
            raise SpecError(msg)
        if self.n_steps < 1 or self.n_steps % self.block_size != 0:
            msg = 'run.n_steps must be a positive multiple of run.block_size'
            raise SpecError(msg)
        # an unset (0) report interval means "one major report at the end";
        # resolve it to n_steps so the effective value is always stored (the
        # dataclass is frozen, so post-init normalization goes through
        # object.__setattr__)
        if self.n_steps_per_major_report == 0:
            object.__setattr__(self, 'n_steps_per_major_report', self.n_steps)
        if self.n_steps_per_major_report < 1 or self.n_steps_per_major_report % self.block_size != 0:
            msg = 'run.n_steps_per_major_report must be a positive multiple of run.block_size'
            raise SpecError(msg)
        if self.store_thin < 1:
            msg = 'run.store_thin must be >= 1'
            raise SpecError(msg)
        for record_idx in self.arg_record:
            if isinstance(record_idx, bool) or not isinstance(record_idx, int) or not 0 <= record_idx < n_chain:
                msg = 'run.arg_record entries must be integers in [0, n_chain)'
                raise SpecError(msg)
        if self.checkpoint_every_blocks < 1:
            msg = 'run.checkpoint_every_blocks must be >= 1'
            raise SpecError(msg)

        if self.exchange_strategy not in EXCHANGE_STRATEGY_CODES:
            msg = f'unknown exchange strategy {self.exchange_strategy!r}; known: {sorted(EXCHANGE_STRATEGY_CODES)}'
            raise SpecError(msg)

        for section, entries in self.proposal_overrides.items():
            if section not in PROPOSAL_SECTIONS:
                msg = f'unknown proposal section {section!r}; known: {sorted(PROPOSAL_SECTIONS)}'
                raise SpecError(msg)
            for key, value in entries.items():
                _check_toml_value(value, f'proposals.{section}.{key}')

        if self.adaptive is not None:
            adaptive_mode = self.adaptive.get('mode')
            if adaptive_mode not in ADAPTIVE_MODES:
                msg = f'unknown adaptive mode {adaptive_mode!r}; known: {sorted(ADAPTIVE_MODES)}'
                raise SpecError(msg)
            unknown_keys = set(self.adaptive) - ADAPTIVE_KEYS
            if unknown_keys:
                msg = f'unknown [adaptive] keys {sorted(unknown_keys)}; known: {sorted(ADAPTIVE_KEYS)}'
                raise SpecError(msg)
            if 'budget_blocks' not in self.adaptive:
                msg = '[adaptive] requires budget_blocks (hard adaptation cap in blocks, plan Phase 5)'
                raise SpecError(msg)
            t_min_factor = self.adaptive.get('T_min_factor', 1)
            if (
                isinstance(t_min_factor, bool)
                or not isinstance(t_min_factor, int | float)
                or not 0.0 < float(t_min_factor) <= 1.0
            ):
                msg = (
                    'adaptive.T_min_factor must be in (0, 1]: the cold-edge target is a multiple of the '
                    'T=1 readout temperature (sub-unit rungs are supported now that storage is index-based)'
                )
                raise SpecError(msg)
            for key, value in self.adaptive.items():
                _check_toml_value(value, f'adaptive.{key}')

    @property
    def n_chain(self) -> int:
        """Total number of chains, from the ladder table."""
        value = self.ladder.get('n_chain')
        if not isinstance(value, int) or isinstance(value, bool):
            msg = 'ladder.n_chain must be an integer'
            raise SpecError(msg)
        return value

    @property
    def n_cold(self) -> int:
        """Number of cold chains, from the ladder table."""
        value = self.ladder.get('n_cold')
        if not isinstance(value, int) or isinstance(value, bool):
            msg = 'ladder.n_cold must be an integer'
            raise SpecError(msg)
        return value

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the run."""
        return self.n_steps // self.block_size

    @property
    def store_size(self) -> int:
        """Stored-sample rows so the ring buffer is exactly filled, never wrapped."""
        return -(-self.n_steps // self.store_thin)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RunSpec:
        """Build and validate a RunSpec from parsed TOML data."""
        likelihood = _require_table(data, 'likelihood')
        likelihood_name = _require_str(likelihood, 'name', 'likelihood')
        likelihood_params = {
            key: _check_toml_value(value, f'likelihood.{key}') for key, value in likelihood.items() if key != 'name'
        }

        ladder_raw = _require_table(data, 'ladder')
        ladder = {key: _check_toml_value(value, f'ladder.{key}') for key, value in ladder_raw.items()}

        run = _require_table(data, 'run')
        if 'n_record' in run:
            msg = (
                'run.n_record was replaced by run.arg_record (a list of extra chain indices to '
                'record beyond the readout chains); update the spec instead of relying on n_record'
            )
            raise SpecError(msg)
        unknown_run_keys = set(run) - RUN_KEYS
        if unknown_run_keys:
            msg = f'unknown [run] keys {sorted(unknown_run_keys)}; known: {sorted(RUN_KEYS)}'
            raise SpecError(msg)

        exchange_raw = data.get('exchange', {})
        if not isinstance(exchange_raw, dict):
            msg = '[exchange] must be a table'
            raise SpecError(msg)
        exchange = dict(exchange_raw)

        proposals_raw = data.get('proposals', {})
        if not isinstance(proposals_raw, dict):
            msg = '[proposals] must be a table of section tables'
            raise SpecError(msg)
        proposal_overrides: dict[str, dict[str, TomlValue]] = {}
        for section, entries in proposals_raw.items():
            if not isinstance(entries, dict):
                msg = f'proposals.{section} must be a table'
                raise SpecError(msg)
            proposal_overrides[section] = {
                key: _check_toml_value(value, f'proposals.{section}.{key}') for key, value in entries.items()
            }

        adaptive_raw = data.get('adaptive')
        adaptive: dict[str, TomlValue] | None = None
        if adaptive_raw is not None:
            if not isinstance(adaptive_raw, dict):
                msg = '[adaptive] must be a table'
                raise SpecError(msg)
            adaptive = {key: _check_toml_value(value, f'adaptive.{key}') for key, value in adaptive_raw.items()}

        return cls(
            name=_require_str(data, 'name', 'spec'),
            seed=_require_int(data, 'seed', 'spec'),
            likelihood_name=likelihood_name,
            likelihood_params=likelihood_params,
            ladder=ladder,
            n_steps=_require_int(run, 'n_steps', 'run'),
            n_steps_per_major_report=_opt_int(run, 'n_steps_per_major_report', 'run', 0),
            block_size=_require_int(run, 'block_size', 'run'),
            store_thin=_opt_int(run, 'store_thin', 'run', 1),
            arg_record=_opt_int_list(run, 'arg_record', 'run'),
            checkpoint_every_blocks=_opt_int(run, 'checkpoint_every_blocks', 'run', 8),
            zero_loglike=_opt_bool(run, 'zero_loglike', 'run', False),
            exchange_strategy=_require_str(exchange, 'strategy', 'exchange')
            if 'strategy' in exchange
            else 'sequential',
            track_full_exchanges=_opt_bool(exchange, 'track_full_exchanges', 'exchange', False),
            proposal_overrides=proposal_overrides,
            adaptive=adaptive,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> RunSpec:
        """Read and validate a RunSpec from a TOML file."""
        with Path(path).open('rb') as spec_file:
            data = tomllib.load(spec_file)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, object]:
        """Get the nested-dict (TOML-shaped) form of this spec."""
        data: dict[str, object] = {
            'name': self.name,
            'seed': self.seed,
            'likelihood': {'name': self.likelihood_name, **self.likelihood_params},
            'ladder': dict(self.ladder),
            'run': {
                'n_steps': self.n_steps,
                'n_steps_per_major_report': self.n_steps_per_major_report,
                'block_size': self.block_size,
                'store_thin': self.store_thin,
                'arg_record': list(self.arg_record),
                'checkpoint_every_blocks': self.checkpoint_every_blocks,
                'zero_loglike': self.zero_loglike,
            },
            'exchange': {
                'strategy': self.exchange_strategy,
                'track_full_exchanges': self.track_full_exchanges,
            },
            'proposals': {section: dict(entries) for section, entries in self.proposal_overrides.items()},
        }
        if self.adaptive is not None:
            data['adaptive'] = dict(self.adaptive)
        return data

    def to_toml_text(self) -> str:
        """Serialize the fully resolved spec as TOML text (artifact embedding)."""
        return dumps_toml(self.to_dict())

    def with_seed(self, seed: int) -> RunSpec:
        """Get a copy of this spec with a different run seed."""
        data = self.to_dict()
        data['seed'] = seed
        return RunSpec.from_dict(data)

    def with_zero_loglike(self, enabled: bool) -> RunSpec:
        """Get a copy of this spec with zero-log-likelihood mode set explicitly."""
        return replace(self, zero_loglike=enabled)

    def build_proposal_config(self) -> configparser.ConfigParser:
        """Build the proposal-manager ConfigParser: defaults plus spec overrides.

        The defaults are read from the resolved repo-root default_config.ini
        (never the CWD); the result is passed explicitly to the proposal
        manager, so engine config reading never depends on the CWD.
        """
        config = configparser.ConfigParser()
        with default_config_path().open() as config_file:
            config.read_file(config_file)
        for section, entries in self.proposal_overrides.items():
            if not config.has_section(section):
                config.add_section(section)
            for key, value in entries.items():
                config[section][key] = str(value)
        return config

    def resolved_config_text(self) -> str:
        """Get the fully resolved proposal config as INI text (artifact embedding).

        Builds a fresh ConfigParser; the runner instead serializes the one
        instance it hands to the sampler, via config_to_text, so the
        artifact records exactly the config the run used.
        """
        return config_to_text(self.build_proposal_config())

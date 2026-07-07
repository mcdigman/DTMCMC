"""CLI entry point for a single run: python -m experiments.harness.run spec.toml --seed N --out dir/."""

import argparse
import sys

from .artifact import validate
from .paths import resolve
from .runner import run_from_spec
from .spec import RunSpec


def main(argv: list[str] | None = None) -> int:
    """Run a single spec end to end and validate the resulting artifact."""
    parser = argparse.ArgumentParser(description='Execute a single DTMCMC run from a TOML spec')
    parser.add_argument('spec', help='path to the run spec TOML (repo-root relative or absolute)')
    parser.add_argument('--seed', type=int, default=None, help='override the run seed in the spec')
    parser.add_argument('--out', default='artifacts', help='output directory for the artifact (default: artifacts/)')
    parser.add_argument(
        '--sampler-verbosity', type=int, default=0, choices=(0, 1, 2),
        help='tracker-summary printing: 0 = silent (default), 1 = at each major-report boundary only, 2 = every checkpoint',
    )
    args = parser.parse_args(argv)

    spec = RunSpec.from_toml(resolve(args.spec))
    if args.seed is not None:
        spec = spec.with_seed(args.seed)

    artifact_path = run_from_spec(spec, args.out, sampler_verbosity=args.sampler_verbosity)

    problems = validate(artifact_path, mode='complete')
    if problems:
        for problem in problems:
            print(f'artifact validation problem: {problem}', file=sys.stderr)
        return 1

    print(f'wrote {artifact_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

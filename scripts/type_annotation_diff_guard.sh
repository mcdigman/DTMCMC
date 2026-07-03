#!/usr/bin/env bash
#
# type_annotation_diff_guard.sh
# -----------------------------
# Guard for a "add type annotations to every function" refactor.
#
# Thin wrapper around type_annotation_diff_guard.py, which does a tokenize-based
# comparison of the OLD (base ref) and NEW (working tree) version of every
# changed .py file. The diff PASSES only if every change is one of:
#
#     1. A type annotation added to a parameter, return, or variable
#     2. A newly created `if TYPE_CHECKING:` block
#     3. A newly added import
#
# Any other change -- renamed identifier, edited comment/docstring, mutated
# string, new statement, suppression comment, reordered/deleted imports,
# deleted line -- fails the guard. Annotation *content* is also policed:
# no Any/object/unknown/Unknown/ndarray, no capitalized generics
# (List/Dict/Tuple/...), generics must be subscripted, banned imports rejected.
#
# See the module docstring in type_annotation_diff_guard.py for the exact model
# and its one deliberate blind spot (token-preserving whitespace reflow).
#
# Interface (designed for repeated calls by a refactoring agent):
#
#     scripts/type_annotation_diff_guard.sh [BASE_REF]
#
#     exit 0  -> diff is clean, keep going
#     exit 1  -> one or more violations (printed as `path:line: [CODE] msg`)
#     exit 2  -> usage / environment error
#
# BASE_REF resolution: first positional arg, else $GUARD_BASE, else the
# merge-base with `master`, else HEAD. The diff always includes uncommitted
# (staged + unstaged) changes so the agent can be checked mid-edit.
#
set -uo pipefail

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
case "${1:-}" in -h|--help) usage 0 ;; esac

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not inside a git repository" >&2
    exit 2
fi

BASE="${1:-${GUARD_BASE:-}}"
if [ -z "$BASE" ]; then
    BASE=$(git merge-base master HEAD 2>/dev/null) || BASE="HEAD"
fi
if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null 2>&1; then
    echo "error: base ref '$BASE' is not a valid commit" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/type_annotation_diff_guard.py"

# Use the project interpreter (>=3.14 floor) so tokenizing matches file syntax.
# Fall back to a bare python3 if the env is unavailable.
if command -v mamba >/dev/null 2>&1; then
    mamba run -n DTMCMC-dev python "$PY" "$BASE"
else
    python3 "$PY" "$BASE"
fi

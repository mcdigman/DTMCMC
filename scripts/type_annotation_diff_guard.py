#!/usr/bin/env python3
"""Tokenize + AST guard for an "add type annotations" refactor.

Enforcement engine behind ``type_annotation_diff_guard.sh``. It compares the
OLD (base ref) and NEW (working tree) version of every changed ``.py`` file and
fails unless every difference is an allowed addition:

    1. A type annotation added to a parameter, return, or variable
       (``x`` -> ``x: T``,  ``)`` -> ``) -> T``,  ``v =`` -> ``v: T =``)
    2. A newly created ``if TYPE_CHECKING:`` block whose body is imports only
    3. A newly added ``import`` / ``from ... import`` statement

Everything else -- a renamed identifier, an edited comment/docstring/string, a
new statement, a suppression comment, reordered/deleted imports, a deleted
line, or *re-indenting existing code under a new block* -- is a violation.

Design
------
* SYNTAX GATE: the new source must ``ast.parse`` cleanly. Tokenizing alone is
  lexical, so it accepts ``total = x: int + y`` or a bare ``if TYPE_CHECKING:``
  header; parsing rejects those.
* STRUCTURE: adding annotations/imports only ever *inserts* tokens, so the old
  token stream must be an ordered subsequence of the new one.
  ``difflib.SequenceMatcher`` then yields only ``equal``/``insert`` opcodes for
  a valid refactor; any ``delete``/``replace`` proves existing code changed.
  INDENT/DEDENT are kept (normalized to type only) so re-indenting existing
  code under a new block shows up as an orphan inserted INDENT and is rejected.
* CONTENT: each inserted annotation's type expression is ``ast.parse``d and
  walked, so the banned-name / generic-subscript policy is applied
  structurally -- including inside string forward references
  (``x: "Any"``) -- while ``Literal["object"]`` string *values* are left alone.

Deliberate non-goals
--------------------
Non-token whitespace is ignored (blank lines, ``a,b`` -> ``a, b``, wrapping a
bracketed expression). Pure formatting drift is out of scope here; enforce it
with a separate ``ruff format --check`` gate.

Exit codes: 0 clean, 1 violations, 2 usage/environment error.
"""

import ast
import io
import subprocess
import sys
import token as tokmod
import tokenize
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# --- annotation / import content policy --------------------------------------

# Names that may never appear in an annotation. `unknown` is matched
# case-insensitively (UNKNOWN/Unknown/unknown); the rest are exact.
BANNED = {'Any', 'object', 'ndarray'}

# Deprecated capitalized generics (PEP 585 / 604 replace them).
DEPRECATED = {'List', 'Dict', 'Tuple', 'Set', 'FrozenSet', 'Optional', 'Union', 'Type'}

# Generics that must carry a subscript, e.g. ``list[int]`` not bare ``list``.
NEED_SUBSCRIPT = {
    'list', 'tuple', 'dict', 'set', 'frozenset', 'type',
    'Callable', 'Iterator', 'Iterable', 'Generator', 'AsyncIterator',
    'AsyncGenerator', 'Coroutine', 'Awaitable', 'Sequence', 'Mapping',
    'MutableMapping', 'MutableSequence', 'NDArray',
}

# Symbols that must not be imported (importing them only enables a banned use).
IMPORT_BANNED = BANNED | DEPRECATED | {'cast', 'Unknown', 'unknown'}

# tokens dropped before comparison (pure layout: blank lines, encoding markers)
_LAYOUT = frozenset({tokmod.NL, tokmod.ENCODING, tokmod.ENDMARKER})

# tokens allowed to delimit a type expression at the token level
_TYPE_OPS = frozenset({'.', '[', ']', '(', ')', ',', '|', '*', '...'})
_OPEN = {'(', '[', '{'}
_CLOSE = {')', ']', '}'}


@dataclass
class Violation:
    path: str
    line: int
    code: str
    msg: str
    src: str = ''


def _keys_and_toks(src: str):
    """Return (comparison keys, TokenInfo list).

    NL / ENCODING / ENDMARKER are dropped so pure whitespace is ignored.
    NEWLINE and INDENT/DEDENT are kept but normalized to type only, so
    indentation *width* changes are ignored while indentation *structure*
    changes (an extra nesting level) remain visible.
    """
    keys: list[tuple[int, str]] = []
    toks: list[tokenize.TokenInfo] = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in _LAYOUT:
            continue
        s = t.string
        if t.type in (tokmod.NEWLINE, tokmod.INDENT, tokmod.DEDENT):
            s = ''
        keys.append((t.type, s))
        toks.append(t)
    return keys, toks


def _bracket_context(new_toks: list[tokenize.TokenInfo], upto: int) -> str | None:
    """Innermost enclosing bracket char just before index ``upto`` (or None)."""
    stack: list[str] = []
    for t in new_toks[:upto]:
        if t.type == tokmod.OP:
            if t.string in _OPEN:
                stack.append(t.string)
            elif t.string in _CLOSE and stack:
                stack.pop()
    return stack[-1] if stack else None


def _consume_type(run, start):
    """Consume a type expression starting at ``run[start]``; return end index.

    Stops at a bracket-depth-0 comma or colon: a single annotation never has a
    top-level comma (that would be a smuggled extra parameter) or colon.
    """
    j = start
    depth = 0
    while j < len(run):
        t = run[j]
        if t.type in (tokmod.NAME, tokmod.STRING, tokmod.NUMBER):
            j += 1
        elif t.type == tokmod.OP and t.string in _TYPE_OPS:
            if t.string in ('(', '['):
                depth += 1
            elif t.string in (')', ']'):
                if depth == 0:
                    break
                depth -= 1
            elif t.string == ',' and depth == 0:
                break
            j += 1
        else:
            break
    return j


def _slice_src(lines, start, end) -> str:
    """Exact source text between (row, col) ``start`` and ``end`` (1-based rows)."""
    (sr, sc), (er, ec) = start, end
    if sr == er:
        return lines[sr - 1][sc:ec]
    parts = [lines[sr - 1][sc:]]
    parts.extend(lines[r - 1] for r in range(sr + 1, er))
    parts.append(lines[er - 1][:ec])
    return ''.join(parts)


def _dotted_final(node) -> str | None:
    """Final identifier of a Name or dotted Attribute (``np.ndarray`` -> ndarray)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _check_name(name: str, *, subscripted: bool, path, line, src, out) -> None:
    if name.lower() == 'unknown' or name in BANNED:
        out.append(Violation(path, line, 'BANNED-TYPE',
                             f'banned type `{name}` in annotation', src))
    elif name in DEPRECATED:
        out.append(Violation(path, line, 'DEPRECATED-GENERIC',
                             f'deprecated capitalized generic `{name}` '
                             f'(use lowercase / X | None)', src))
    elif name in NEED_SUBSCRIPT and not subscripted:
        out.append(Violation(path, line, 'GENERIC-NO-ARGS',
                             f'unparametrized generic `{name}` (needs [...])', src))


def _ann_policy(node, path, line, src, out) -> None:
    """Walk a parsed annotation expression and apply the type policy."""
    if isinstance(node, ast.Name):
        _check_name(node.id, subscripted=False, path=path, line=line, src=src, out=out)
    elif isinstance(node, ast.Attribute):
        final = _dotted_final(node)
        if final:
            _check_name(final, subscripted=False, path=path, line=line, src=src, out=out)
    elif isinstance(node, ast.Subscript):
        base = _dotted_final(node.value)
        if base == 'Literal':
            return  # Literal["object"] etc. -- the slice holds values, not types
        if base:
            _check_name(base, subscripted=True, path=path, line=line, src=src, out=out)
        _ann_policy(node.slice, path, line, src, out)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        _ann_policy(node.left, path, line, src, out)
        _ann_policy(node.right, path, line, src, out)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _ann_policy(elt, path, line, src, out)
    elif isinstance(node, ast.Starred):
        _ann_policy(node.value, path, line, src, out)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        # string forward reference -- parse and police its contents
        try:
            inner = ast.parse(node.value, mode='eval').body
        except SyntaxError:
            return
        _ann_policy(inner, path, line, src, out)


def _policy_on_type_text(text: str, path, line, src, out) -> None:
    try:
        tree = ast.parse(text.strip(), mode='eval')
    except SyntaxError:
        return  # the whole-file ast.parse gate already rejected broken syntax
    _ann_policy(tree.body, path, line, src, out)


def _classify_insert(run, run_abs_start, new_toks, lines, path, out) -> None:
    """Classify one inserted token run; append a Violation per illegal piece."""
    i = 0
    n = len(run)
    while i < n:
        t = run[i]
        line = t.start[0]
        src = t.line.rstrip('\n')

        if t.type in (tokmod.NEWLINE, tokmod.DEDENT):
            i += 1
            continue

        # inserted INDENT: legitimate only when its matching DEDENT is also in
        # this run (a fully-inserted block body). An orphan INDENT means
        # existing code was re-indented (e.g. moved under TYPE_CHECKING).
        if t.type == tokmod.INDENT:
            depth = 1
            k = i + 1
            while k < n and depth:
                if run[k].type == tokmod.INDENT:
                    depth += 1
                elif run[k].type == tokmod.DEDENT:
                    depth -= 1
                k += 1
            if depth:
                out.append(Violation(path, line, 'REINDENT',
                                     'existing code re-indented under a new '
                                     'block (scope change)', src))
            i += 1
            continue

        # inline annotation:  : T   or   -> T
        if t.type == tokmod.OP and t.string in (':', '->'):
            if t.string == ':':
                ctx = _bracket_context(new_toks, run_abs_start + i)
                if ctx in ('[', '{'):
                    out.append(Violation(path, line, 'NON-ANNOTATION',
                                         'inserted `:` is a slice/dict colon, '
                                         'not an annotation', src))
                    i += 1
                    continue
            end = _consume_type(run, i + 1)
            type_toks = run[i + 1:end]
            if not type_toks:
                out.append(Violation(path, line, 'NON-ANNOTATION',
                                     f'`{t.string}` inserted without a type', src))
            else:
                text = _slice_src(lines, type_toks[0].start, type_toks[-1].end)
                _policy_on_type_text(text, path, line, src, out)
            i = end
            continue

        # new import statement
        if t.type == tokmod.NAME and t.string in ('import', 'from'):
            j = i
            names_after_import = False
            while j < n and run[j].type != tokmod.NEWLINE:
                tk = run[j]
                if tk.type == tokmod.COMMENT:
                    out.append(Violation(path, line, 'COMMENT',
                                         'comment added on a new import line', src))
                elif tk.type == tokmod.OP and tk.string == '*':
                    out.append(Violation(path, tk.start[0], 'WILDCARD-IMPORT',
                                         'wildcard import can introduce banned '
                                         'names', src))
                elif tk.type == tokmod.NAME and tk.string == 'import':
                    names_after_import = True
                elif (names_after_import and tk.type == tokmod.NAME
                        and tk.string in IMPORT_BANNED):
                    out.append(Violation(path, tk.start[0], 'BANNED-IMPORT',
                                         f'import of banned symbol `{tk.string}`',
                                         src))
                j += 1
            i = j
            continue

        # if TYPE_CHECKING: header (its body is validated as separate inserts)
        if (t.type == tokmod.NAME and t.string == 'if'
                and i + 2 < n
                and run[i + 1].type == tokmod.NAME
                and run[i + 1].string == 'TYPE_CHECKING'
                and run[i + 2].type == tokmod.OP and run[i + 2].string == ':'):
            i += 3
            continue

        # anything else is disallowed new content
        if t.type == tokmod.COMMENT:
            out.append(Violation(path, line, 'COMMENT',
                                 f'comment added: `{t.string}`', src))
        else:
            out.append(Violation(path, line, 'NEW-CODE',
                                 f'non-annotation token inserted: `{t.string}`',
                                 src))
        i += 1


def analyze(path: str, old_src: str, new_src: str) -> list[Violation]:
    out: list[Violation] = []

    # SYNTAX GATE -- reject anything that does not parse (lexically-valid but
    # ungrammatical annotation insertions, bare block headers, etc.)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        return [Violation(path, e.lineno or 0, 'SYNTAX',
                          f'new version does not parse: {e.msg}')]

    try:
        old_keys, _ = _keys_and_toks(old_src)
        new_keys, new_toks = _keys_and_toks(new_src)
    except (tokenize.TokenError, IndentationError) as e:
        return [Violation(path, 0, 'TOKENIZE', f'could not tokenize: {e}')]

    lines = new_src.splitlines(keepends=True)
    sm = SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag in ('delete', 'replace'):
            # a valid annotation-only edit is always a pure insert; any
            # delete/replace means an existing token was removed or changed
            tk = new_toks[j1] if j1 < len(new_toks) else None
            out.append(Violation(
                path, tk.start[0] if tk else 0, 'MUTATED',
                'existing code removed or changed (rename / comment / '
                'docstring / string / deletion)',
                tk.line.rstrip('\n') if tk else ''))
        elif tag == 'insert':
            _classify_insert(new_toks[j1:j2], j1, new_toks, lines, path, out)
    return out


def _git(args: list[str]) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True,  # noqa: S603, S607
                          check=True).stdout


def _old_source(base: str, path: str) -> str | None:
    r = subprocess.run(['git', 'show', f'{base}:{path}'],  # noqa: S603, S607
                       capture_output=True, text=True, check=False)
    return r.stdout if r.returncode == 0 else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('usage: type_annotation_diff_guard.py BASE_REF', file=sys.stderr)
        return 2
    base = argv[1]

    violations: list[Violation] = []
    violations.extend(
        Violation(f, 0, 'NEW-FILE', 'new/untracked .py file')
        for f in _git(['ls-files', '--others', '--exclude-standard',
                       '--', '*.py']).splitlines() if f
    )

    status = _git(['diff', '--name-status', base, '--', '*.py'])
    for row in status.splitlines():
        parts = row.split('\t')
        code = parts[0]
        if code.startswith('A'):
            violations.append(Violation(parts[1], 0, 'NEW-FILE',
                                        'file added; refactor annotates '
                                        'existing code only'))
        elif code.startswith('D'):
            violations.append(Violation(parts[1], 0, 'DELETED', 'file deleted'))
        elif code.startswith(('R', 'C')):
            violations.append(Violation(parts[-1], 0, 'RENAMED',
                                        f'file renamed/copied from {parts[1]}'))
        elif code.startswith('M'):
            path = parts[1]
            old = _old_source(base, path)
            try:
                new = Path(path).read_text(encoding='utf-8')
            except OSError as e:
                violations.append(Violation(path, 0, 'IO', str(e)))
                continue
            if old is None:
                violations.append(Violation(path, 0, 'NEW-FILE',
                                            'no base version found'))
                continue
            violations.extend(analyze(path, old, new))

    violations.sort(key=lambda v: (v.path, v.line))
    for v in violations:
        print(f'{v.path}:{v.line}: [{v.code}] {v.msg}')
        if v.src:
            print(f'    | {v.src}')

    if violations:
        print(f'FAIL: type-annotation diff guard found {len(violations)} '
              f'violation(s) (base={base})', file=sys.stderr)
        return 1
    print(f'PASS: diff contains only type annotations, TYPE_CHECKING blocks, '
          f'and imports (base={base})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

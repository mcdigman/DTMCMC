#!/usr/bin/env python3
"""Tokenize-based guard for an "add type annotations" refactor.

This is the enforcement engine behind ``type_annotation_diff_guard.sh``. It
compares the OLD (base ref) and NEW (working tree) version of every changed
``.py`` file at the Python *token* level and fails unless every difference is
an allowed addition:

    1. A type annotation added to a parameter, return, or variable
       (``x`` -> ``x: T``,  ``)`` -> ``) -> T``,  ``v =`` -> ``v: T =``)
    2. A newly created ``if TYPE_CHECKING:`` block header
    3. A newly added ``import`` / ``from ... import`` statement

Everything else -- a renamed identifier, an edited comment or docstring, a
mutated string literal, a new statement, a suppression comment, reordered
imports, a deleted line -- is a violation.

Why tokenize instead of regex-over-word-diff
---------------------------------------------
Adding annotations/imports only ever *inserts* tokens; it never removes or
rewrites an existing token. Therefore the OLD token stream must be an ordered
subsequence of the NEW token stream. Running ``difflib.SequenceMatcher`` over
the two streams then yields ONLY ``equal`` and ``insert`` opcodes for a valid
refactor: any ``delete`` or ``replace`` opcode is proof that existing code was
touched. Each ``insert`` run is classified as annotation / import /
TYPE_CHECKING-header or rejected.

Because comments and string literals are their own tokens, comment/docstring
mutations are caught structurally (no regex), and because the banned-word and
generic-subscript policy runs over NAME tokens (not raw text), a string like
``Literal["object"]`` is not mistaken for the ``object`` type.

Deliberate blind spot
---------------------
Token comparison ignores non-token whitespace, so a purely cosmetic reflow
that changes no token (e.g. ``a,b`` -> ``a, b``, or wrapping a bracketed
expression across lines) is NOT flagged. Any reflow that changes a real token
(a trailing comma, added parentheses, a split string) IS flagged. This is the
only non-annotation change the guard can overlook.

Exit codes: 0 clean, 1 violations, 2 usage/environment error.
"""


import io
import subprocess
import sys
import token as tokmod
import tokenize
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# --- annotation / import content policy --------------------------------------

# Names that may never appear in an annotation or be imported.
BANNED = {'Any', 'object', 'unknown', 'Unknown', 'ndarray'}

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
IMPORT_BANNED = BANNED | DEPRECATED | {'cast'}

# tokens that carry no code/comment/string content -- dropped before comparing
_LAYOUT = frozenset({
    tokmod.NL, tokmod.INDENT, tokmod.DEDENT,
    tokmod.ENCODING, tokmod.ENDMARKER,
})

# tokens allowed to make up a type expression (NAME/STRING/NUMBER handled separately)
_TYPE_OPS = frozenset({'.', '[', ']', '(', ')', ',', '|', '*', '...'})
_OPEN = {'(': ')', '[': ']', '{': '}'}
_CLOSE = {')', ']', '}'}


@dataclass
class Violation:
    path: str
    line: int
    code: str
    msg: str
    src: str = ''


def _keys_and_toks(src: str):
    """Return (comparison keys, TokenInfo list) with layout tokens filtered.

    NEWLINE is kept (it delimits statements) but normalized so line-ending
    differences do not register as changes.
    """
    keys: list[tuple[int, str]] = []
    toks: list[tokenize.TokenInfo] = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in _LAYOUT:
            continue
        s = '' if t.type == tokmod.NEWLINE else t.string
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


def _check_type_tokens(type_toks, path, line, src, out: list[Violation]) -> None:
    """Apply banned/deprecated/subscript policy to an annotation's tokens."""
    for i, t in enumerate(type_toks):
        if t.type != tokmod.NAME:
            continue
        name = t.string
        if name in BANNED:
            out.append(Violation(path, line, 'BANNED-TYPE',
                                 f'banned type `{name}` in annotation', src))
        elif name in DEPRECATED:
            out.append(Violation(path, line, 'DEPRECATED-GENERIC',
                                 f'deprecated capitalized generic `{name}` '
                                 f'(use lowercase / X | None)', src))
        elif name in NEED_SUBSCRIPT:
            nxt = type_toks[i + 1] if i + 1 < len(type_toks) else None
            if not (nxt and nxt.type == tokmod.OP and nxt.string == '['):
                out.append(Violation(path, line, 'GENERIC-NO-ARGS',
                                     f'unparametrized generic `{name}` '
                                     f'(needs [...])', src))


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
                    break  # closing bracket belongs to the enclosing context
                depth -= 1
            elif t.string == ',' and depth == 0:
                break
            j += 1
        else:
            break
    return j


def _classify_insert(run, run_abs_start, new_toks, path, out: list[Violation]) -> None:
    """Classify one inserted token run; append a Violation per illegal piece."""
    i = 0
    n = len(run)
    while i < n:
        t = run[i]
        line = t.start[0]
        src = t.line.rstrip('\n')

        # statement separators between stacked insertions
        if t.type == tokmod.NEWLINE:
            i += 1
            continue

        # ---- inline annotation:  : T   or   -> T --------------------------
        if t.type == tokmod.OP and t.string in (':', '->'):
            if t.string == ':':
                ctx = _bracket_context(new_toks, run_abs_start + i)
                if ctx in ('[', '{'):
                    # a colon inside [] or {} is a slice/dict, i.e. real code
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
                _check_type_tokens(type_toks, path, line, src, out)
            i = end
            continue

        # ---- new import statement ----------------------------------------
        if t.type == tokmod.NAME and t.string in ('import', 'from'):
            j = i
            names_after_import = False
            bad_comment = False
            while j < n and run[j].type != tokmod.NEWLINE:
                tk = run[j]
                if tk.type == tokmod.COMMENT:
                    bad_comment = True
                if tk.type == tokmod.NAME and tk.string == 'import':
                    names_after_import = True
                elif (names_after_import and tk.type == tokmod.NAME
                        and tk.string in IMPORT_BANNED):
                    out.append(Violation(path, tk.start[0], 'BANNED-IMPORT',
                                         f'import of banned symbol '
                                         f'`{tk.string}`', src))
                j += 1
            if bad_comment:
                out.append(Violation(path, line, 'COMMENT',
                                     'comment added on a new import line', src))
            i = j
            continue

        # ---- if TYPE_CHECKING: header ------------------------------------
        if (t.type == tokmod.NAME and t.string == 'if'
                and i + 2 < n
                and run[i + 1].type == tokmod.NAME
                and run[i + 1].string == 'TYPE_CHECKING'
                and run[i + 2].type == tokmod.OP and run[i + 2].string == ':'):
            i += 3
            continue

        # ---- anything else is disallowed new content ---------------------
        if t.type == tokmod.COMMENT:
            out.append(Violation(path, line, 'COMMENT',
                                 f'comment added: `{t.string}`', src))
        else:
            out.append(Violation(path, line, 'NEW-CODE',
                                 f'non-annotation token inserted: '
                                 f'`{t.string}`', src))
        i += 1


def analyze(path: str, old_src: str, new_src: str) -> list[Violation]:
    out: list[Violation] = []
    try:
        old_keys, _ = _keys_and_toks(old_src)
    except (tokenize.TokenError, SyntaxError, IndentationError) as e:
        return [Violation(path, 0, 'TOKENIZE',
                          f'base version does not tokenize: {e}')]
    try:
        new_keys, new_toks = _keys_and_toks(new_src)
    except (tokenize.TokenError, SyntaxError, IndentationError) as e:
        return [Violation(path, 0, 'TOKENIZE',
                          f'new version does not tokenize (syntax broken?): {e}')]

    sm = SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag in ('delete', 'replace'):
            # A valid annotation-only edit is always a pure `insert`; any
            # delete/replace means an existing token was removed or changed.
            tk = new_toks[j1] if j1 < len(new_toks) else None
            line = tk.start[0] if tk else 0
            src = tk.line.rstrip('\n') if tk else ''
            out.append(Violation(
                path, line, 'MUTATED',
                'existing code removed or changed (rename / comment / '
                'docstring / string / deletion)', src))
        elif tag == 'insert':
            _classify_insert(new_toks[j1:j2], j1, new_toks, path, out)
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

    # brand-new untracked .py files are not part of an annotation refactor
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
            violations.append(Violation(parts[1], 0, 'DELETED',
                                        'file deleted'))
        elif code.startswith(('R', 'C')):
            violations.append(Violation(parts[-1], 0, 'RENAMED',
                                        f'file renamed/copied from {parts[1]}'))
        elif code.startswith('M'):
            path = parts[1]
            old = _old_source(base, path)
            try:
                with Path(path).open(encoding='utf-8') as fh:
                    new = fh.read()
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

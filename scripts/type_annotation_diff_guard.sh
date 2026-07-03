#!/usr/bin/env bash
#
# type_annotation_diff_guard.sh
# -----------------------------
# Guard for a "add type annotations to every function" refactor.
#
# It inspects the current `git diff` (working tree vs a base ref) and FAILS
# (non-zero exit) unless every change is one of:
#
#     1. A type annotation added to a function argument   (`x`  -> `x: T`)
#     2. A type annotation added to a function return     (`)`  -> `) -> T`)
#     3. A newly created `if TYPE_CHECKING:` block
#     4. A newly added import (top-level or inside a TYPE_CHECKING block)
#
# Anything else -- renamed identifiers, reworded comments or docstrings,
# reflowed/re-wrapped lines, deleted lines, new statements, moved code -- is a
# violation. In addition it enforces annotation *content* policy:
#
#     * rejects explicit `Any`, `object`, `unknown`
#     * rejects `np.ndarray` / `numpy.ndarray` / bare `ndarray`
#     * rejects deprecated capitalized generics: List/Dict/Tuple/Set/
#       FrozenSet/Optional/Union
#     * requires type arguments on generics: list/tuple/dict/set/frozenset/
#       type/Callable/Iterator/Iterable/Generator/Sequence/Mapping/NDArray/...
#     * rejects `# type: ignore` and `cast(...)`
#
# STRICT mode: a changed line may differ from the original ONLY by inserted
# annotation tokens. Any formatter-driven re-wrapping counts as a violation.
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

usage() {
    sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

case "${1:-}" in
    -h|--help) usage 0 ;;
esac

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not inside a git repository" >&2
    exit 2
fi

# ----- resolve base ref ------------------------------------------------------
BASE="${1:-${GUARD_BASE:-}}"
if [ -z "$BASE" ]; then
    if BASE=$(git merge-base master HEAD 2>/dev/null); then
        :
    else
        BASE="HEAD"
    fi
fi
if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null 2>&1; then
    echo "error: base ref '$BASE' is not a valid commit" >&2
    exit 2
fi

fail=0

# ----- brand-new / untracked .py files are not part of the refactor ----------
while IFS= read -r f; do
    [ -n "$f" ] || continue
    echo "$f:0: [NEW-FILE] new/untracked .py file is not an annotation change"
    fail=1
done < <(git ls-files --others --exclude-standard -- '*.py')

# ----- inspect the diff ------------------------------------------------------
# Word-diff isolates insertions ({+..+}) from removals ([-..-]) so a
# regex-based line classifier can tell "only annotations were inserted" from
# "existing text was changed".
diff_out=$(git -c color.ui=never diff --word-diff=plain \
    --word-diff-regex='[[:alnum:]_]+|[^[:space:]]' "$BASE" -- '*.py')

if [ -n "$diff_out" ]; then
    if ! printf '%s\n' "$diff_out" | perl -e '
        use strict;
        use warnings;

        # --- annotation content policy -------------------------------------
        my @banned = (
            [ qr/\bAny\b/,               "explicit Any" ],
            [ qr/\bobject\b/,            "explicit object" ],
            [ qr/\bunknown\b/,           "explicit unknown" ],
            [ qr/\bnp\.ndarray\b/,       "np.ndarray (use NDArray[...])" ],
            [ qr/\bnumpy\.ndarray\b/,    "numpy.ndarray (use NDArray[...])" ],
            [ qr/\bndarray\b/,           "bare ndarray (use NDArray[...])" ],
            [ qr/\b(?:List|Dict|Tuple|Set|FrozenSet|Optional|Union)\b/,
                                         "deprecated capitalized generic (use lowercase / X | None)" ],
            [ qr/#\s*type:\s*ignore/,    "# type: ignore" ],
            [ qr/\bcast\s*\(/,           "cast(...)" ],
        );
        # generics that MUST carry a subscript
        my $need_args = qr/\b(list|tuple|dict|set|frozenset|type|Callable|
                            Iterator|Iterable|Generator|Sequence|Mapping|
                            MutableMapping|MutableSequence|NDArray)\b/x;

        sub policy {
            # returns list of human messages for a type/insertion string
            my ($s) = @_;
            my @msgs;
            for my $b (@banned) {
                push @msgs, $b->[1] if $s =~ $b->[0];
            }
            while ($s =~ /$need_args/g) {
                my $g = $1;
                my $rest = substr($s, pos($s));
                push @msgs, "unparametrized generic `$g` (needs [...])"
                    unless $rest =~ /\A\s*\[/;
            }
            return @msgs;
        }

        # allowed shapes for a fully-new inserted line
        sub new_line_ok {
            my ($l) = @_;
            return 1 if $l =~ /^\s*$/;                                  # blank
            return 1 if $l =~ /^\s*if\s+TYPE_CHECKING\s*:\s*$/;         # guard
            return 1 if $l =~ /^\s*import\s+\S/;                        # import
            return 1 if $l =~ /^\s*from\s+[.\w]+\s+import\b/;           # import
            return 1 if $l =~ /^\s*[\w.]+(?:\s+as\s+\w+)?\s*,?\s*$/;    # import continuation
            return 1 if $l =~ /^\s*[()]\s*,?\s*$/;                      # paren of import block
            return 0;
        }

        my $curfile = "";
        my $skip    = 1;      # skip until a .py file header is seen
        my $newline = 0;      # best-effort line number in the new file
        my ($pending_new, $pending_del, $diffpath) = (0, 0, "");
        my $violations = 0;

        sub report {
            my ($code, $msg, $src) = @_;
            print "$curfile:$newline: [$code] $msg\n";
            print "    | $src\n" if defined $src && $src ne "";
            $violations++;
        }

        while (my $line = <STDIN>) {
            chomp $line;

            if ($line =~ m{^diff --git a/(\S+) b/(\S+)}) {
                $diffpath = $2;
                ($pending_new, $pending_del) = (0, 0);
                $skip = 1;
                next;
            }
            if ($line =~ /^new file mode/)     { $pending_new = 1; next; }
            if ($line =~ /^deleted file mode/) { $pending_del = 1; next; }

            if ($line =~ m{^\+\+\+ (.*)$}) {
                my $p = $1;
                $p =~ s{^b/}{};
                $curfile = ($p eq "/dev/null") ? $diffpath : $p;
                $skip = ($curfile =~ /\.py$/) ? 0 : 1;
                if (!$skip && $pending_new) {
                    $newline = 0;
                    report("NEW-FILE",
                        "new file added; refactor only annotates existing code");
                }
                if (!$skip && $pending_del) {
                    $newline = 0;
                    report("DELETED",
                        "file deleted; refactor only annotates existing code");
                }
                next;
            }

            next if $skip;

            # metadata we ignore
            next if $line =~ /^(index |--- |old mode|new mode|similarity |
                                rename |copy |Binary |\\ No newline)/x;

            if ($line =~ /^@@ -\d+(?:,\d+)? \+(\d+)/) {
                $newline = $1;
                next;
            }

            # ---- a diff content line ------------------------------------
            my $has_del = ($line =~ /\[-/);
            my $has_ins = ($line =~ /\{\+/);

            # does this line exist in the NEW file? (pure deletions do not)
            my $stripped = $line;
            $stripped =~ s/\[-.*?-\]//g;   # drop removed content
            $stripped =~ s/\{\+.*?\+\}//g; # drop added content
            my $is_pure_deletion = ($has_del && !$has_ins && $stripped =~ /^\s*$/);

            if ($has_del) {
                report("MUTATED",
                    "existing code removed or changed (rename / comment / "
                    . "docstring / reflow / deletion)", $line);
                $newline++ unless $is_pure_deletion;
                next;
            }

            if (!$has_ins) {
                # unchanged context line
                $newline++;
                next;
            }

            # reconstruct the new-file line text (markers removed)
            my $newtext = $line;
            $newtext =~ s/\{\+(.*?)\+\}/$1/g;

            # collect insertion spans
            my @spans = ($line =~ /\{\+(.*?)\+\}/g);

            # remainder after removing insertions: whitespace-only => whole
            # line is brand new
            my $remainder = $line;
            $remainder =~ s/\{\+.*?\+\}//g;

            if ($remainder =~ /^\s*$/) {
                # fully-new inserted line: must be import / TYPE_CHECKING / blank
                if (!new_line_ok($newtext)) {
                    report("NEW-CODE",
                        "added line is not an import or TYPE_CHECKING block",
                        $newtext);
                } elsif ($newtext =~ /\b(Any|cast)\b/) {
                    report("BANNED-IMPORT",
                        "import of a banned symbol (Any/cast)", $newtext);
                }
                $newline++;
                next;
            }

            # inline insertions into an existing line: each must be an annotation
            for my $s (@spans) {
                (my $t = $s) =~ s/^\s+//;
                $t =~ s/\s+$//;

                my $type;
                if    ($t =~ /^:\s*(.+)$/s)  { $type = $1; }   # arg / var annotation
                elsif ($t =~ /^->\s*(.+)$/s) { $type = $1; }   # return annotation
                else {
                    report("NON-ANNOTATION",
                        "inserted text is not a type annotation: `$s`", $newtext);
                    next;
                }

                # a comma at bracket-depth 0 means the insertion runs past a
                # single annotation -- e.g. a brand-new parameter was added.
                my $depth = 0;
                for my $c (split //, $type) {
                    $depth++ if $c =~ /[\[({]/;
                    $depth-- if $c =~ /[\])}]/;
                    if ($c eq "," && $depth <= 0) {
                        report("NEW-PARAM",
                            "insertion spans past one annotation "
                            . "(new parameter or reflow?): `$s`", $newtext);
                        last;
                    }
                }

                for my $m (policy($type)) {
                    report("POLICY", "$m in `$s`", $newtext);
                }
            }
            $newline++;
        }

        exit($violations > 0 ? 1 : 0);
    '; then
        fail=1
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "FAIL: type-annotation diff guard found violations (base=$BASE)" >&2
    exit 1
fi

echo "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=$BASE)"
exit 0

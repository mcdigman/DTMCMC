# Latest Type Annotation Diff Guard Adversarial Report

```json
{
  "schema": "dtmcmc.diff_guard_latest_adversarial_report.v1",
  "generated_at": "2026-07-02",
  "target_files": [
    "/Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.sh",
    "/Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.py"
  ],
  "scope_adjustment": "Variable and class annotations outside function arguments were treated as acceptable when they are strictly annotations and comply with annotation policy.",
  "verification_method": {
    "summary": "Created scratch Git repositories under /private/tmp, committed baseline sample.py files, applied one candidate diff at a time, and invoked both the shell wrapper and the Python engine with base ref HEAD.",
    "shell_invocation": "XDG_CACHE_HOME=/private/tmp/codex-xdg-cache bash /Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.sh HEAD",
    "engine_invocation": "python3 /Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.py HEAD",
    "syntax_check_for_selected_cases": "python3 -m py_compile sample.py",
    "note": "Without XDG_CACHE_HOME, the shell wrapper attempted mamba run and failed before invoking the validator in this sandbox."
  },
  "old_holes_rechecked": [
    {
      "case": "comment mutation by appending annotation-shaped text",
      "diff_lines": ["-    # marker", "+    # marker: int"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "docstring mutation by appending annotation-shaped text",
      "diff_lines": ["-    \"\"\"value x\"\"\"", "+    \"\"\"value x: int\"\"\""],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "runtime string mutation by appending annotation-shaped text",
      "diff_lines": ["-    label = \"cake\"", "+    label = \"cake: int\""],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "split-token # type: ignore",
      "diff_lines": ["-    # type", "+    # type: ignore"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "new return statement",
      "diff_lines": ["+    return"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "new pass statement",
      "diff_lines": ["+    pass"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "TYPE_CHECKING block with pass body",
      "diff_lines": ["+if TYPE_CHECKING:", "+    pass"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "deprecated generic import",
      "diff_lines": ["+from typing import List"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "ndarray import",
      "diff_lines": ["+from numpy import ndarray"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "object import",
      "diff_lines": ["+from builtins import object"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "capitalized Unknown annotation",
      "diff_lines": ["-def f(x, y):", "+def f(x: Unknown, y):"],
      "latest_result": "closed",
      "observed_exit": 1
    },
    {
      "case": "Literal string false positives from prior regex implementation",
      "diff_lines": ["+def f(x: Literal[\"object\"], y):", "+def f(x: Literal[\"]\", \"x\"], y):"],
      "latest_result": "closed as false positives; both now pass",
      "observed_exit": 0
    },
    {
      "case": "variable and class annotations",
      "diff_lines": ["+    total: int = x + y", "+    value: int = 1"],
      "latest_result": "accepted under updated review scope",
      "observed_exit": 0
    }
  ],
  "findings": [
    {
      "id": "L-FN001",
      "type": "false_negative",
      "severity": "critical",
      "title": "Existing function can be moved under TYPE_CHECKING by indentation only",
      "violated_rule": "Existing code must not be mutated; moving a runtime function under TYPE_CHECKING changes semantics.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "+from typing import TYPE_CHECKING",
        "+if TYPE_CHECKING:",
        "-def f(x, y):",
        "-    total = x + y",
        "-    return total",
        "+    def f(x, y):",
        "+        total = x + y",
        "+        return total"
      ],
      "root_cause": "INDENT and DEDENT are dropped from the comparison keys, so semantic indentation changes are invisible while the inserted TYPE_CHECKING header is allowed.",
      "script_evidence_lines": [73, 77, 102, 108, 229, 236]
    },
    {
      "id": "L-FN002",
      "type": "false_negative",
      "severity": "critical",
      "title": "Invalid bare TYPE_CHECKING header before existing code passes",
      "violated_rule": "The guard should not pass syntactically invalid diffs.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 1,
      "py_compile_stderr_excerpt": "IndentationError: expected an indented block after 'if' statement",
      "diff_lines": [
        "+if TYPE_CHECKING:",
        " def f(x, y):"
      ],
      "root_cause": "The validator tokenizes but does not parse or compile the new source, and it accepts any inserted if TYPE_CHECKING: token sequence without checking suite structure.",
      "script_evidence_lines": [249, 260, 229, 236]
    },
    {
      "id": "L-FN003",
      "type": "false_negative",
      "severity": "critical",
      "title": "Invalid colon insertion inside expression passes",
      "violated_rule": "Inserted annotation-shaped tokens must be actual annotations, not syntax-breaking tokens in expressions.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 1,
      "py_compile_stderr_excerpt": "SyntaxError: invalid syntax",
      "diff_lines": [
        "-    total = x + y",
        "+    total = x: int + y"
      ],
      "root_cause": "Any inserted ':' outside '[' or '{' is treated as an annotation prefix, with no AST context check.",
      "script_evidence_lines": [185, 203]
    },
    {
      "id": "L-FN004",
      "type": "false_negative",
      "severity": "critical",
      "title": "Invalid arrow insertion inside assignment passes",
      "violated_rule": "Inserted return-annotation-shaped tokens must be actual return annotations.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 1,
      "py_compile_stderr_excerpt": "SyntaxError: invalid syntax",
      "diff_lines": [
        "-    total = x + y",
        "+    total -> int = x + y"
      ],
      "root_cause": "Any inserted '->' token starts an accepted annotation run, regardless of whether it occurs in a function signature.",
      "script_evidence_lines": [185, 203]
    },
    {
      "id": "L-FN005",
      "type": "false_negative",
      "severity": "critical",
      "title": "Invalid annotation after return keyword passes",
      "violated_rule": "Inserted annotation-shaped tokens must preserve valid Python syntax.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 1,
      "py_compile_stderr_excerpt": "SyntaxError: invalid syntax",
      "diff_lines": [
        "-    return total",
        "+    return: int total"
      ],
      "root_cause": "The classifier accepts ':' plus type tokens without verifying the surrounding AST node is an annotation site.",
      "script_evidence_lines": [185, 203]
    },
    {
      "id": "L-FN006",
      "type": "false_negative",
      "severity": "high",
      "title": "Whitespace-only operator reflow passes",
      "violated_rule": "Formatter reflows were explicitly ruled out of scope.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-    z = x + y",
        "+    z = x+y"
      ],
      "root_cause": "Token comparison intentionally ignores non-token whitespace.",
      "script_evidence_lines": [33, 39, 94, 108]
    },
    {
      "id": "L-FN007",
      "type": "false_negative",
      "severity": "high",
      "title": "Blank lines can be added anywhere",
      "violated_rule": "Only annotations, TYPE_CHECKING blocks, and imports may be added.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "+",
        "+"
      ],
      "root_cause": "NL layout tokens are dropped before comparison.",
      "script_evidence_lines": [73, 77, 102, 108]
    },
    {
      "id": "L-FN008",
      "type": "false_negative",
      "severity": "high",
      "title": "Bracketed expression line wrapping passes",
      "violated_rule": "Formatter reflows were explicitly ruled out of scope.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-    vals = [1, 2, 3]",
        "+    vals = [",
        "+        1, 2, 3",
        "+    ]"
      ],
      "root_cause": "NL, INDENT, and DEDENT are ignored, so token-preserving wrapping is invisible.",
      "script_evidence_lines": [33, 39, 73, 77, 102, 108]
    },
    {
      "id": "L-FN009",
      "type": "false_negative",
      "severity": "high",
      "title": "Wildcard typing import passes",
      "violated_rule": "Banned imports should be rejected; wildcard typing imports can introduce Any, cast, deprecated generics, and other banned names.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "+from typing import *"
      ],
      "root_cause": "Import policy only checks NAME tokens after import; '*' is not rejected.",
      "script_evidence_lines": [206, 227]
    },
    {
      "id": "L-FN010",
      "type": "false_negative",
      "severity": "high",
      "title": "Wildcard numpy import passes",
      "violated_rule": "Banned ndarray exposure should be rejected; wildcard numpy import can introduce ndarray.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "+from numpy import *"
      ],
      "root_cause": "Import policy only checks NAME tokens after import; '*' is not rejected.",
      "script_evidence_lines": [206, 227]
    },
    {
      "id": "L-FN011",
      "type": "false_negative",
      "severity": "high",
      "title": "String forward-reference Any annotation passes",
      "violated_rule": "No explicit Any may be added.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: \"Any\", y):"
      ],
      "root_cause": "Annotation policy ignores STRING tokens, even when the string is itself a forward-reference annotation.",
      "script_evidence_lines": [123, 141, 153, 155]
    },
    {
      "id": "L-FN012",
      "type": "false_negative",
      "severity": "high",
      "title": "String forward-reference np.ndarray annotation passes",
      "violated_rule": "np.ndarray and bare ndarray must be rejected.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: \"np.ndarray\", y):"
      ],
      "root_cause": "Annotation policy ignores STRING tokens, even when the string is itself a forward-reference annotation.",
      "script_evidence_lines": [123, 141, 153, 155]
    },
    {
      "id": "L-FN013",
      "type": "false_negative",
      "severity": "high",
      "title": "String forward-reference bare list annotation passes",
      "violated_rule": "Generic types like list must have type arguments.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: \"list\", y):"
      ],
      "root_cause": "STRING annotations are not parsed for banned names or unparametrized generics.",
      "script_evidence_lines": [123, 141, 153, 155]
    },
    {
      "id": "L-FN014",
      "type": "false_negative",
      "severity": "medium",
      "title": "All-caps UNKNOWN annotation passes",
      "violated_rule": "Explicit unknown should be rejected.",
      "observed_shell_exit": 0,
      "observed_engine_exit": 0,
      "py_compile_exit": 0,
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: UNKNOWN, y):"
      ],
      "root_cause": "The banned set includes unknown and Unknown but not other case variants.",
      "script_evidence_lines": [56, 57, 123, 141]
    },
    {
      "id": "L-FP001",
      "type": "false_positive",
      "severity": "medium",
      "title": "Shell wrapper can fail before validation when mamba exists but cannot lock its cache",
      "violated_expectation": "The guard interface should be suitable for repeated calling and should distinguish environment errors from diff failures.",
      "observed_shell_exit_without_cache_workaround": 1,
      "stderr_excerpt": "libmamba Could not open lockfile '/Users/mcdigman/.cache/mamba/proc/proc.lock'",
      "root_cause": "The wrapper uses mamba whenever command -v mamba succeeds and does not fall back to python3 if mamba run fails.",
      "script_evidence_file": "/Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.sh"
    }
  ],
  "recommendations": [
    {
      "id": "L-R001",
      "text": "Parse or compile the new source with ast.parse before token-diff classification; tokenization alone does not reject many SyntaxError and IndentationError cases."
    },
    {
      "id": "L-R002",
      "text": "Do not drop INDENT and DEDENT from semantic comparison. They are syntax tokens and can move existing runtime code under TYPE_CHECKING."
    },
    {
      "id": "L-R003",
      "text": "Verify inserted ':' and '->' occur at AST-recognized annotation sites, not merely outside [] or {}."
    },
    {
      "id": "L-R004",
      "text": "Reject token-preserving formatter reflows if the contract still forbids formatter changes. A line-level or exact-whitespace diff check is needed in addition to token comparison."
    },
    {
      "id": "L-R005",
      "text": "Reject wildcard imports in this guard, especially from typing, numpy, numpy.typing, builtins, and typing_extensions."
    },
    {
      "id": "L-R006",
      "text": "Parse string-literal forward annotations and apply the same banned-type and generic-subscript policy to their contents, while continuing to allow Literal string values."
    },
    {
      "id": "L-R007",
      "text": "Make the wrapper either use python3 directly or fall back when mamba run fails; if it exits for an environment problem, use exit code 2 rather than a diff-failure shaped exit 1."
    }
  ]
}
```

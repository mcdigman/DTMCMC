# Type Annotation Diff Guard Adversarial Report

```json
{
  "schema": "dtmcmc.diff_guard_adversarial_report.v1",
  "generated_at": "2026-07-02",
  "target_script": "/Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.sh",
  "target_script_trust": "untrusted",
  "verification_method": {
    "summary": "Created scratch Git repositories under /private/tmp, committed a baseline sample.py, applied one candidate diff at a time, then invoked the guard with base ref HEAD.",
    "guard_invocation": "bash /Users/mcdigman/Claude/Projects/DTMCMC/scripts/type_annotation_diff_guard.sh HEAD",
    "repo_mutation": "No project files were modified during verification except this report file.",
    "main_repo_status_before_report": "## dev...origin/dev\n?? scripts/"
  },
  "priority": {
    "false_negatives": "higher",
    "false_positives": "lower"
  },
  "recommendations": [
    {
      "id": "R001",
      "text": "Reject any added or reconstructed new line containing the suppression tokens ignore or noqa. Verified bypasses include appended pyrefly: ignore, mypy: ignore, pyright: ignore, and noqa."
    },
    {
      "id": "R002",
      "text": "Reject any new # token and any modification to a line whose reconstructed new text is a comment, because comments are outside the allowed scope."
    },
    {
      "id": "R003",
      "text": "Do not classify inserted text as allowed merely because it starts with : or ->. Confirm the insertion is inside a function signature parameter or return annotation."
    },
    {
      "id": "R004",
      "text": "Do not allow generic standalone word lines as import continuations unless parser state proves they are inside a newly added import statement."
    },
    {
      "id": "R005",
      "text": "Run banned-type and deprecated-generic policy on all added import lines, all reconstructed annotation text, and full reconstructed changed lines where split-token bypasses are possible."
    }
  ],
  "findings": [
    {
      "id": "FN001",
      "type": "false_negative",
      "severity": "high",
      "title": "Local variable annotation accepted",
      "violated_rule": "Only function argument and function return annotations may be added.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The inline insertion classifier accepts any inserted span beginning with :, including local variable annotations.",
      "diff_lines": [
        "-    total = x + y",
        "+    total: int = x + y"
      ],
      "script_evidence_lines": [248, 249]
    },
    {
      "id": "FN002",
      "type": "false_negative",
      "severity": "high",
      "title": "Class attribute annotation accepted",
      "violated_rule": "Only function argument and function return annotations may be added.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The guard does not distinguish class attributes from function signatures.",
      "diff_lines": [
        "-    value = 1",
        "+    value: int = 1"
      ],
      "script_evidence_lines": [248, 249]
    },
    {
      "id": "FN003",
      "type": "false_negative",
      "severity": "high",
      "title": "Comment mutation accepted",
      "violated_rule": "Comments must not be mutated.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Appending annotation-shaped text to an existing comment is treated as an allowed annotation insertion.",
      "diff_lines": [
        "-    # marker",
        "+    # marker: int"
      ],
      "script_evidence_lines": [248, 249]
    },
    {
      "id": "FN004",
      "type": "false_negative",
      "severity": "high",
      "title": "Docstring edit accepted",
      "violated_rule": "Docstring edits are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Appending annotation-shaped text inside a string is not distinguished from a real annotation.",
      "diff_lines": [
        "-    \"\"\"value x\"\"\"",
        "+    \"\"\"value x: int\"\"\""
      ],
      "script_evidence_lines": [248, 249]
    },
    {
      "id": "FN005",
      "type": "false_negative",
      "severity": "high",
      "title": "Runtime string literal mutation accepted",
      "violated_rule": "Existing code must not be mutated.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The guard applies annotation-shape regexes inside runtime string literals.",
      "diff_lines": [
        "-    label = \"cake\"",
        "+    label = \"cake: int\""
      ],
      "script_evidence_lines": [248, 249]
    },
    {
      "id": "FN006",
      "type": "false_negative",
      "severity": "high",
      "title": "Split-token type ignore accepted",
      "violated_rule": "# type: ignore is prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Only the inserted span is checked for # type: ignore, so changing an existing '# type' comment to '# type: ignore' inserts only ': ignore'.",
      "diff_lines": [
        "-    # type",
        "+    # type: ignore"
      ],
      "script_evidence_lines": [94, 103, 271]
    },
    {
      "id": "FN007",
      "type": "false_negative",
      "severity": "high",
      "title": "Pyrefly ignore accepted when appended to existing comment",
      "violated_rule": "Suppression comments are out of scope and comment mutations are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "verification_note": "A fully new '# pyrefly: ignore' comment line was rejected, but appending ': ignore' to an existing '# pyrefly' comment passed.",
      "exploit": "The inserted span ': ignore' is annotation-shaped and the full reconstructed comment line is not checked for suppression text.",
      "diff_lines": [
        "-    # pyrefly",
        "+    # pyrefly: ignore"
      ],
      "script_evidence_lines": [248, 249, 271]
    },
    {
      "id": "FN008",
      "type": "false_negative",
      "severity": "high",
      "title": "Mypy ignore accepted when appended to existing comment",
      "violated_rule": "Suppression comments are out of scope and comment mutations are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "verification_note": "A fully new '# mypy: ignore' comment line was rejected, but appending ': ignore' to an existing '# mypy' comment passed.",
      "exploit": "The inserted span ': ignore' is annotation-shaped and the full reconstructed comment line is not checked for suppression text.",
      "diff_lines": [
        "-    # mypy",
        "+    # mypy: ignore"
      ],
      "script_evidence_lines": [248, 249, 271]
    },
    {
      "id": "FN009",
      "type": "false_negative",
      "severity": "high",
      "title": "Pyright ignore accepted when appended to existing comment",
      "violated_rule": "Suppression comments are out of scope and comment mutations are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "verification_note": "A fully new '# pyright: ignore' comment line was rejected, but appending ': ignore' to an existing '# pyright' comment passed.",
      "exploit": "The inserted span ': ignore' is annotation-shaped and the full reconstructed comment line is not checked for suppression text.",
      "diff_lines": [
        "-    # pyright",
        "+    # pyright: ignore"
      ],
      "script_evidence_lines": [248, 249, 271]
    },
    {
      "id": "FN010",
      "type": "false_negative",
      "severity": "high",
      "title": "Noqa accepted when appended to existing comment",
      "violated_rule": "Lint suppression comments are out of scope and comment mutations are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "verification_note": "A fully new '# noqa' comment line was rejected, but appending ': noqa' to an existing comment passed.",
      "exploit": "The inserted span ': noqa' is annotation-shaped and no policy rejects noqa.",
      "diff_lines": [
        "-    # lint",
        "+    # lint: noqa"
      ],
      "script_evidence_lines": [94, 105, 248, 249]
    },
    {
      "id": "FN011",
      "type": "false_negative",
      "severity": "high",
      "title": "New return statement accepted",
      "violated_rule": "No new statements or extant-code mutation is allowed.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The fully-new-line classifier treats any standalone word-like line as an import continuation, regardless of parser state.",
      "diff_lines": [
        "     if x:",
        "         y += 1",
        "+    return",
        "     return total"
      ],
      "script_evidence_lines": [127, 134, 229]
    },
    {
      "id": "FN012",
      "type": "false_negative",
      "severity": "high",
      "title": "New pass statement accepted",
      "violated_rule": "No new statements or extant-code mutation is allowed.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The fully-new-line classifier treats any standalone word-like line as an import continuation, regardless of parser state.",
      "diff_lines": [
        "     if x:",
        "         y += 1",
        "+    pass",
        "     return total"
      ],
      "script_evidence_lines": [127, 134, 229]
    },
    {
      "id": "FN013",
      "type": "false_negative",
      "severity": "high",
      "title": "TYPE_CHECKING block with non-import body accepted",
      "violated_rule": "Only creation of an if TYPE_CHECKING block and new imports are allowed; arbitrary body statements are not.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The guard does not track whether lines after if TYPE_CHECKING are imports only; pass is accepted by the standalone-word continuation rule.",
      "diff_lines": [
        "+from typing import TYPE_CHECKING",
        "+if TYPE_CHECKING:",
        "+    pass",
        "+",
        " def f(x, y):"
      ],
      "script_evidence_lines": [127, 134, 229]
    },
    {
      "id": "FN014",
      "type": "false_negative",
      "severity": "medium",
      "title": "Deprecated generic import accepted",
      "violated_rule": "Deprecated capitalized generics are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Banned import checking on fully new lines only rejects Any and cast, not List, Dict, Tuple, Set, FrozenSet, Optional, or Union.",
      "diff_lines": [
        "+from typing import List"
      ],
      "script_evidence_lines": [94, 105, 235]
    },
    {
      "id": "FN015",
      "type": "false_negative",
      "severity": "medium",
      "title": "Banned ndarray import accepted",
      "violated_rule": "Bare ndarray and np.ndarray are prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Banned import checking on fully new lines only rejects Any and cast, not ndarray.",
      "diff_lines": [
        "+from numpy import ndarray"
      ],
      "script_evidence_lines": [94, 105, 235]
    },
    {
      "id": "FN016",
      "type": "false_negative",
      "severity": "medium",
      "title": "Explicit object import accepted",
      "violated_rule": "Explicit object is prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "Banned import checking on fully new lines only rejects Any and cast, not object.",
      "diff_lines": [
        "+from builtins import object"
      ],
      "script_evidence_lines": [94, 105, 235]
    },
    {
      "id": "FN017",
      "type": "false_negative",
      "severity": "medium",
      "title": "Capitalized Unknown accepted",
      "violated_rule": "Explicit unknown is prohibited.",
      "observed_exit": 0,
      "observed_stdout": "PASS: diff contains only type annotations, TYPE_CHECKING blocks, and imports (base=HEAD)",
      "exploit": "The unknown policy check is lowercase-only.",
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: Unknown, y):"
      ],
      "script_evidence_lines": [94, 97]
    },
    {
      "id": "FP001",
      "type": "false_positive",
      "severity": "low",
      "title": "Literal string containing closing bracket and comma rejected",
      "violated_expectation": "A valid function argument annotation should be accepted if it otherwise follows policy.",
      "observed_exit": 1,
      "observed_stdout": "sample.py:1: [NEW-PARAM] insertion spans past one annotation (new parameter or reflow?): `: Literal[\\\"]\\\", \\\"x\\\"]`",
      "exploit": "The bracket-depth scanner counts bracket characters inside string literals.",
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: Literal[\\\"]\\\", \\\"x\\\"], y):"
      ],
      "script_evidence_lines": [257, 269]
    },
    {
      "id": "FP002",
      "type": "false_positive",
      "severity": "low",
      "title": "Literal string value object rejected",
      "violated_expectation": "A valid Literal annotation containing the string value object should not be treated as the object type.",
      "observed_exit": 1,
      "observed_stdout": "sample.py:1: [POLICY] explicit object in `: Literal[\\\"object\\\"]`",
      "exploit": "The banned-word policy runs over raw annotation text and does not parse Python syntax.",
      "diff_lines": [
        "-def f(x, y):",
        "+def f(x: Literal[\\\"object\\\"], y):"
      ],
      "script_evidence_lines": [94, 97, 271]
    }
  ],
  "confirmed_rejections": [
    {
      "id": "CR001",
      "case": "fully_new_pyrefly_ignore_comment",
      "observed_exit": 1,
      "diff_lines": ["+    # pyrefly: ignore"]
    },
    {
      "id": "CR002",
      "case": "fully_new_mypy_ignore_comment",
      "observed_exit": 1,
      "diff_lines": ["+    # mypy: ignore"]
    },
    {
      "id": "CR003",
      "case": "fully_new_pyright_ignore_comment",
      "observed_exit": 1,
      "diff_lines": ["+    # pyright: ignore"]
    },
    {
      "id": "CR004",
      "case": "fully_new_noqa_comment",
      "observed_exit": 1,
      "diff_lines": ["+    # noqa"]
    },
    {
      "id": "CR005",
      "case": "append_new_hash_to_code",
      "observed_exit": 1,
      "diff_lines": [
        "-    total = x + y",
        "+    total = x + y  # note"
      ]
    }
  ]
}
```

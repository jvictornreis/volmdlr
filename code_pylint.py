import os
import random
import sys
from datetime import date
import math

from pylint import __version__
from pylint.lint import Run

MIN_NOTE = 8.20

UNWATCHED_ERRORS = ['fixme', 'trailing-whitespace', 'import-error', 'missing-final-newline']

EFFECTIVE_DATE = date(2023, 1, 31)
WEEKLY_DECREASE = 0.03

MAX_ERROR_BY_TYPE = {
                     "wrong-spelling-in-comment": 425,
                     "wrong-spelling-in-docstring": 371,
                     'invalid-name': 743,
                     'no-else-return': 49,
                     'consider-using-f-string': 77,
                     'no-member': 9,
                     'inconsistent-return-statements': 6,
                     'unused-variable': 32,
                     'arguments-differ': 58,
                     'too-many-locals': 73,
                     'unused-argument': 43,
                     'too-many-arguments': 62,
                     'line-too-long': 19,
                     'consider-using-enumerate': 22,
                     'too-many-branches': 27,
                     'too-many-statements': 18,
                     'super-init-not-called': 13,
                     'no-name-in-module': 5,
                     'abstract-method': 34,
                     'duplicate-code': 9,
                     'arguments-renamed': 8,
                     'too-many-ancestors': 29,
                     'too-few-public-methods': 2,
                     # 'expression-not-assigned': 1,
                     'non-parent-init-called': 6,
                     'too-many-public-methods': 11,
                     'use-implicit-booleaness-not-comparison': 8,
                     'too-many-instance-attributes': 10,
                     'protected-access': 4,
                     'undefined-loop-variable': 5,
                     'unspecified-encoding': 5,
                     'too-many-function-args': 7,
                     'too-many-nested-blocks': 7,
                     'attribute-defined-outside-init': 6,
                     'too-many-return-statements': 1,
                     'cyclic-import': 4,
                     'raise-missing-from': 2,
                     'no-else-raise': 3,
                     'no-else-continue': 4,
                     'undefined-variable': 6,  # 2 when gmsh is fixed
                     'no-else-break': 4,
                     'broad-except': 1,
                     "broad-exception-caught": 1,
                     'too-many-boolean-expressions': 3,
                     'too-many-lines': 3,
                     'redundant-keyword-arg': 3,
                     'c-extension-no-member': 2,
                     # 'access-member-before-definition': 1,
                     'modified-iterating-list': 2,
                     'consider-using-with': 1,
                     'unnecessary-dunder-call': 2,
                     'unnecessary-lambda': 2,
                     'chained-comparison': 2,
                     'missing-module-docstring': 2,
                     'consider-using-generator': 1,
                     'cell-var-from-loop': 1,
                     'import-outside-toplevel': 1,
                     'unsubscriptable-object': 1,
                     }

ERRORS_WITHOUT_TIME_DECREASE = []

print("pylint version: ", __version__)

time_decrease_coeff = 1 - (date.today() - EFFECTIVE_DATE).days / 7.0 * WEEKLY_DECREASE

f = open(os.devnull, "w")

old_stdout = sys.stdout
sys.stdout = f

results = Run(["volmdlr", "--output-format=json", "--reports=no"], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
sys.stdout = old_stdout

PYLINT_OBJECTS = True
if hasattr(results.linter.stats, "global_note"):
    pylint_note = results.linter.stats.global_note
    PYLINT_OBJECT_STATS = True
else:
    pylint_note = results.linter.stats["global_note"]
    PYLINT_OBJECT_STATS = False


def extract_messages_by_type(type_):
    return [m for m in results.linter.reporter.messages if m.symbol == type_]


error_detected = False
error_over_ratchet_limit = False

if PYLINT_OBJECT_STATS:
    stats_by_msg = results.linter.stats.by_msg
else:
    stats_by_msg = results.linter.stats["by_msg"]

for error_type, number_errors in stats_by_msg.items():
    if error_type not in UNWATCHED_ERRORS:
        base_errors = MAX_ERROR_BY_TYPE.get(error_type, 0)

        if error_type in ERRORS_WITHOUT_TIME_DECREASE:
            max_errors = base_errors
        else:
            max_errors = math.ceil(base_errors * time_decrease_coeff)

        time_decrease_effect = base_errors - max_errors
        # print('time_decrease_effect', time_decrease_effect)

        if number_errors > max_errors:
            error_detected = True
            print(
                f"\nFix some {error_type} errors: {number_errors}/{max_errors} "
                f"(time effect: {time_decrease_effect} errors)")

            messages = extract_messages_by_type(error_type)
            messages_to_show = sorted(random.sample(messages, min(30, len(messages))), key=lambda m: (m.path, m.line))
            for message in messages_to_show:
                print(f"{message.path} line {message.line}: {message.msg}")
        elif number_errors < max_errors:
            print(f"\nYou can lower number of {error_type} to {number_errors+time_decrease_effect}"
                  f" (actual {base_errors})")

for error_type in MAX_ERROR_BY_TYPE:
    if error_type not in stats_by_msg:
        print(f"You can delete {error_type} entry from MAX_ERROR_BY_TYPE dict")

if error_detected:
    raise RuntimeError("Too many errors\nRun pylint volmdlr to get the errors")

if error_over_ratchet_limit:
    raise RuntimeError("Please lower the error limits in code_pylint.py MAX_ERROR_BY_TYPE according to warnings above")

print("Pylint note: ", pylint_note)
if pylint_note < MIN_NOTE:
    raise ValueError(f"Pylint not is too low: {pylint_note}, expected {MIN_NOTE}")

print("You can increase MIN_NOTE in pylint to {} (actual: {})".format(pylint_note, MIN_NOTE))

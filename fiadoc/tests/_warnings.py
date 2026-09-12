"""Strict warning assertion helper for parser tests

Use `assert_warnings` to declare, per test case, exactly which warnings are expected (required),
which are tolerated/optional (allowed), and to fail the test if any other warning slips through.
"""
import re
import warnings
from contextlib import contextmanager
from typing import Iterable, Optional

import pytest


@contextmanager
def assert_warnings(
        required: Optional[Iterable[str]] = None,
        allowed: Optional[Iterable[str]] = None,
        category: type = Warning
):
    """Strict warning assertion

    Any warning of `category` matching neither `required` nor `allowed` fails the test, as does any
    `required` pattern that never matched

    :param required: regex patterns. Each must match at least one emitted warning
    :param allowed: regex patterns. If find warnings not in `required` but match `allowed`, it's OK
    :param category: only warnings of this category (or a subclass) are policed. Other categories
                     are passed through untouched
    """
    required_patterns = [re.compile(p) for p in (required or [])]
    allowed_patterns = [re.compile(p) for p in (allowed or [])]
    satisfied = [False] * len(required_patterns)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        yield recorded

    unexpected = []
    for w in recorded:
        if not issubclass(w.category, category):  # Skip warnings not in `category`
            continue
        msg = str(w.message)
        matched = False
        for i, pat in enumerate(required_patterns):
            if pat.search(msg):
                satisfied[i] = True
                matched = True
                break
        if matched:                                           # Should either match `required`
            continue
        if any(pat.search(msg) for pat in allowed_patterns):  # Or match `allowed`
            continue
        unexpected.append(w)                                  # Otherwise, an unexpected warning

    # All `required` should be there
    missing = [required_patterns[i].pattern for i, ok in enumerate(satisfied) if not ok]

    if not missing and not unexpected:
        return

    lines = []
    if missing:
        lines.append('Required but not emitted warning patterns:')
        for pat in missing:
            lines.append(f'    - {pat!r}')
    if unexpected:
        lines.append('Unexpected warnings:')
        for w in unexpected:
            lines.append(f'    - {w.category.__name__}: {w.message} at {w.filename}:{w.lineno}')

    pytest.fail('\n'.join(lines))

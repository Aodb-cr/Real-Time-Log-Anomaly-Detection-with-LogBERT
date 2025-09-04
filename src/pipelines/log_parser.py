from __future__ import annotations

"""Simple regex-based log parser to derive a normalized log key.

parse_raw_log(line: str) -> str performs lightweight normalization:
- removes timestamps in the form YYYY-MM-DD HH:MM:SS (or 'T' separator)
- replaces IPv4 addresses with '<IP>'
- replaces integers with >=5 digits with '<NUM>'
- collapses multiple whitespace to a single space and strips ends

This is a pure function suitable for unit testing.
"""

import re

# Precompiled patterns for efficiency
_TS_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")
_IPV4_PATTERN = re.compile(r"\b(?:(?:\d{1,3})\.){3}(?:\d{1,3})\b")
_LONG_INT_PATTERN = re.compile(r"\b\d{5,}\b")
_WS_PATTERN = re.compile(r"\s+")


def parse_raw_log(line: str) -> str:
    """Convert a raw log line to a normalized log key (template).

    Parameters
    - line: raw log string

    Returns
    - normalized template string
    """
    if not isinstance(line, str):
        line = str(line)

    s = line
    # 1) strip timestamps
    s = _TS_PATTERN.sub(" ", s)
    # 2) anonymize IPv4 addresses
    s = _IPV4_PATTERN.sub("<IP>", s)
    # 3) anonymize long integers (>=5 digits)
    s = _LONG_INT_PATTERN.sub("<NUM>", s)
    # 4) collapse spaces
    s = _WS_PATTERN.sub(" ", s).strip()
    return s

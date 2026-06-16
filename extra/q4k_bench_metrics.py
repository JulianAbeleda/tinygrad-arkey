#!/usr/bin/env python3
"""Single source for the regexes that parse extra/q4_k_bench.py text output.

q4_k_bench.py (run with --format text --primitive) emits the device-bandwidth
and primitive-correctness lines these patterns match. Centralized so the several
generation/audit drivers that subprocess q4_k_bench share one definition of its
output grammar.
"""
from __future__ import annotations

import re

DEVICE_RE = re.compile(r"q4k_primitive_gemv:.*device_q4_eff=(?P<dev>[0-9.]+) GB/s")
CORRECT_RE = re.compile(r"primitive_gemv_correctness: (?P<status>PASS|FAIL)")

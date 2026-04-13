#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Shared utilities for TileLang ST test cases.

Provides:
  - Case helpers:  get_valid_shape()
  - Data helpers:  setup_case_rng(), save_case_data()
  - Compare:       compare_bin(), run_compare()  (full compare entry point)
  - Styling:       supports_color(), style_pass(), style_fail()
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Case helpers
# ---------------------------------------------------------------------------

REQUIRED_CASE_KEYS = {"name", "dtype", "shape", "valid_shape", "eps"}


def validate_cases(cases):
    """Check that every case has all required keys."""
    for i, case in enumerate(cases):
        missing = REQUIRED_CASE_KEYS - case.keys()
        if missing:
            raise ValueError(f"cases[{i}] ({case.get('name', '?')}) missing keys: {missing}")


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def setup_case_rng(case):
    """Set a per-case deterministic random seed.

    Using hash(name) ensures that adding/reordering cases does not change
    the random data of existing cases.
    """
    np.random.seed(hash(case["name"]) & 0xFFFFFFFF)


def save_case_data(case_name, data_dict):
    """Create case directory and write {name}.bin for each entry in data_dict.

    Args:
        case_name: subdirectory name (e.g. "f32_16x64").
        data_dict: mapping from file stem to numpy array,
                   e.g. {"input1": arr1, "input2": arr2, "golden": golden}.
    """
    os.makedirs(case_name, exist_ok=True)
    for name, arr in data_dict.items():
        arr.tofile(os.path.join(case_name, f"{name}.bin"))


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

ANSI_RESET = "\033[0m"
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_BOLD_RED = "\033[1;31m"


def supports_color():
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "", "dumb")


def style_pass(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_GREEN}{text}{ANSI_RESET}"


def style_fail(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_RED}{text}{ANSI_RESET}"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_bin(golden_path, output_path, dtype, eps, shape, valid_shape):
    """Compare golden and output binary files within the valid region.

    Returns True on pass, False on mismatch.
    """
    golden = np.fromfile(golden_path, dtype=dtype).reshape(shape)
    output = np.fromfile(output_path, dtype=dtype).reshape(shape)

    vr, vc = valid_shape
    g = golden[:vr, :vc].astype(np.float64, copy=False)
    o = output[:vr, :vc].astype(np.float64, copy=False)

    if g.shape != o.shape:
        print(style_fail(f"[ERROR] Shape mismatch: golden {g.shape} vs output {o.shape}"))
        return False
    if not np.allclose(g, o, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(g - o)
        idx = int(np.argmax(abs_diff))
        print(style_fail(f"[ERROR] Mismatch: max diff={float(abs_diff.flat[idx])} "
                         f"at flat idx={idx} "
                         f"(golden={g.flat[idx]}, output={o.flat[idx]})"))
        return False
    return True


def run_compare(cases):
    """Main entry point for per-testcase compare.py scripts.

    Reads an optional case filter from sys.argv[1], iterates over *cases*,
    and exits with code 2 if any comparison fails.
    """
    validate_cases(cases)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in cases:
        if case_filter is not None and case["name"] != case_filter:
            continue
        case_dir = case["name"]
        golden_path = os.path.join(case_dir, "golden.bin")
        output_path = os.path.join(case_dir, "output.bin")
        ok = compare_bin(golden_path, output_path, case["dtype"], case["eps"],
                         case["shape"], case["valid_shape"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))

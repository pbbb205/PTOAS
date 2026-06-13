#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Compare output.bin vs golden.bin for tstore_fp ST cases."""

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass


def get_dtype_for_bin(case):
    """Return the numpy dtype used for binary file storage of golden and output."""
    dst_dtype_raw = case.get("dst_dtype_raw", None)
    if dst_dtype_raw == "bf16":
        return np.uint16  # bfloat16 is stored as uint16
    if case["dst_dtype"] is not None:
        return case["dst_dtype"]
    return np.uint16  # fallback


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        M = case["M"]
        N = case["N"]
        name = case["name"]
        eps = case["eps"]

        dtype = get_dtype_for_bin(case)

        golden = np.fromfile(os.path.join(name, "golden.bin"), dtype=dtype).reshape(M, N)
        output = np.fromfile(os.path.join(name, "output.bin"), dtype=dtype).reshape(M, N)

        # For bf16 stored as uint16, convert back to f32 for comparison
        dst_dtype_raw = case.get("dst_dtype_raw", None)
        if dst_dtype_raw == "bf16":
            golden_f32 = (golden.astype(np.uint32) << 16).view(np.float32)
            output_f32 = (output.astype(np.uint32) << 16).view(np.float32)
            ok = result_cmp(golden_f32, output_f32, eps)
        else:
            ok = result_cmp(golden, output, eps)

        if ok:
            print(style_pass(f"[INFO] {name}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {name}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()

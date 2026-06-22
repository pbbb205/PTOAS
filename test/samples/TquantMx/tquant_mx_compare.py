#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CI/remote-validation compare for the TquantMx sample.

The tquant.mx kernel has four outputs with different dtypes (fp8 dst, e8m0 exp,
f32 max, f32 scaling), so we cannot use the single-dtype compare_outputs helper.
Instead we compare each output with its own dtype and tolerance.
"""

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import compare_file, finalize_compare, load_case_meta


def main():
    meta = load_case_meta()
    output_names = meta.outputs

    # Outputs are ordered by tstore appearance: dst, exp, max, scaling.
    # Map by position; fall back to name heuristics if fewer than 4 detected.
    dst_name = output_names[0] if len(output_names) > 0 else "v2"
    exp_name = output_names[1] if len(output_names) > 1 else "v3"
    max_name = output_names[2] if len(output_names) > 2 else "v4"
    scaling_name = output_names[3] if len(output_names) > 3 else "v5"

    ok = True
    # dst: fp8 e4m3fn packed as int8 — exact byte match.
    ok = compare_file(f"golden_{dst_name}.bin", f"{dst_name}.bin", np.int8, atol=0.0) and ok
    # exp: e8m0 as uint8 — exact match.
    ok = compare_file(f"golden_{exp_name}.bin", f"{exp_name}.bin", np.uint8, atol=0.0) and ok
    # max: f32 per-group absmax.
    ok = compare_file(f"golden_{max_name}.bin", f"{max_name}.bin", np.float32, atol=1e-5) and ok
    # scaling: f32 per-group reciprocal scale.
    ok = compare_file(f"golden_{scaling_name}.bin", f"{scaling_name}.bin", np.float32, atol=1e-5) and ok

    finalize_compare(ok)


if __name__ == "__main__":
    main()

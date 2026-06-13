#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for tstore_fp ST cases.

For each case:
  - x1_gm[M, K]: left input matrix (loaded via TLOAD.MAT)
  - x2_gm[K, N]: right input matrix (loaded via TLOAD.MAT)
  - quant_vector[1, N]: per-column quantization vector (for TSTORE_FP)
  - golden[M, N]: expected result = matmul(x1, x2) * quant_vector cast to dst_dtype

Ref: pto-isa tstore_acc2gm vector quant gen_data.py patterns.
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data


def bf16_to_uint16(arr):
    """Convert float32 array to bfloat16 stored as uint16."""
    # bfloat16 is the upper 16 bits of float32
    f32_view = arr.view(np.uint32)
    bf16_uint16 = (f32_view >> 16).astype(np.uint16)
    return bf16_uint16


for case in CASES:
    setup_case_rng(case)

    M = case["M"]
    N = case["N"]
    K = case["K"]
    quant_mode = case["quant_mode"]
    name = case["name"]

    src_dtype_raw = case.get("src_dtype_raw", None)
    dst_dtype_raw = case.get("dst_dtype_raw", None)
    scaling_dtype_raw = case.get("scaling_dtype_raw", None)

    # Generate input matrices as float32 for computation
    x1_f32 = np.random.uniform(-1, 1, size=(M, K)).astype(np.float32)
    x2_f32 = np.random.uniform(-1, 1, size=(K, N)).astype(np.float32)

    # Compute golden in float32
    golden_f32 = np.matmul(x1_f32, x2_f32)

    # Prepare input data in source dtype
    if src_dtype_raw == "bf16":
        x1_bin = bf16_to_uint16(x1_f32)
        x2_bin = bf16_to_uint16(x2_f32)
    elif case["src_dtype"] == np.float16:
        x1_bin = x1_f32.astype(np.float16)
        x2_bin = x2_f32.astype(np.float16)
    else:
        x1_bin = x1_f32
        x2_bin = x2_f32

    # Generate quantization vector (per-column scaling)
    quant_vector_f32 = np.ones((1, N), dtype=np.float32)
    if scaling_dtype_raw == "bf16":
        quant_bin = bf16_to_uint16(quant_vector_f32)
    elif case.get("scaling_dtype") == np.float16:
        quant_bin = quant_vector_f32.astype(np.float16)
    else:
        quant_bin = bf16_to_uint16(quant_vector_f32)

    # Golden with quantization: result * quant_vector
    golden_quant = golden_f32 * quant_vector_f32
    if dst_dtype_raw == "bf16":
        golden_bin = bf16_to_uint16(golden_quant)
    elif case["dst_dtype"] == np.float16:
        golden_bin = golden_quant.astype(np.float16)
    else:
        golden_bin = golden_quant

    data_dict = {
        "x1_gm": x1_bin,
        "x2_gm": x2_bin,
        "quant_vector": quant_bin,
        "golden": golden_bin,
    }

    save_case_data(name, data_dict)
    print(f"[INFO] gen_data: {name} M={M} N={N} K={K} quant_mode={quant_mode}")

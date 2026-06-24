# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


def float32_to_bfloat16_and_back(f32_arr):
    """Simulate bfloat16 truncation (round-to-nearest-even truncated to 16-bit)."""
    as_uint32 = f32_arr.view(np.uint32)
    rounded = ((as_uint32 + np.uint32(0x7FFF)) & np.uint32(0xFFFF0000))
    return rounded.view(np.float32)


for case in CASES:
    setup_case_rng(case)
    m, k, n = case["m"], case["k"], case["n"]
    dtype = case["dtype"]
    dtype_out = case["dtype_out"]
    id_dtype = case.get("id_dtype")

    A = np.random.uniform(-1.0, 1.0, size=(m, k)).astype(dtype)
    B = np.random.uniform(-1.0, 1.0, size=(k, n)).astype(dtype)
    matmul_f32 = np.matmul(A.astype(np.float32), B.astype(np.float32))

    data = {"input1": A, "input2": B}

    if id_dtype is not None:
        id_mat = np.eye(m, dtype=dtype_out)

        if case["name"].startswith("acc2mat_bf16"):
            id_as_f32 = float32_to_bfloat16_and_back(id_mat)
            id_as_bf16_bits = (id_as_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            data["input3"] = id_as_bf16_bits
            quantized = float32_to_bfloat16_and_back(matmul_f32)
        elif case["name"].startswith("acc2mat_f16"):
            data["input3"] = id_mat.astype(id_dtype)
            quantized = matmul_f32.astype(np.float16).astype(np.float32)
        else:
            data["input3"] = id_mat.astype(id_dtype)
            quantized = matmul_f32

        golden = np.matmul(quantized, id_mat.astype(np.float32)).astype(dtype_out)
    else:
        golden = matmul_f32.astype(dtype_out)

    data["golden"] = golden

    save_case_data(case["name"], data)
    print(
        f"[INFO] gen_data: {case['name']} A=({m},{k}) B=({k},{n}) "
        f"dtype={dtype.__name__} dtype_out={dtype_out.__name__}"
        + (f" id_dtype={id_dtype.__name__}" if id_dtype else "")
    )

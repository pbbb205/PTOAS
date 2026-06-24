# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Test cases for pto.tinsert ST (Acc->Mat NZ only)."""

import numpy as np


CASES = [
    {
        "name": "acc2mat_f16_16x16",
        "kernel": "TINSERT_acc2mat_f16_16x16",
        "m": 16, "k": 16, "n": 16,
        "dtype": np.float16,
        "dtype_out": np.float32,
        "id_dtype": np.float16,
        "path": "acc2mat_nz",
        "has_output": True,
        "eps": 1e-2,
    },
    {
        "name": "acc2mat_bf16_16x16",
        "kernel": "TINSERT_acc2mat_bf16_16x16",
        "m": 16, "k": 16, "n": 16,
        "dtype": np.float16,
        "dtype_out": np.float32,
        "id_dtype": np.uint16,
        "path": "acc2mat_nz",
        "has_output": True,
        "eps": 1e-2,
    },
]

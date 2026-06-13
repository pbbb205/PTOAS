#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tstore_fp ST test cases.

End-to-end cube pipeline: TLOAD.MAT -> TMATMUL -> TSTORE_FP (ACC + FP -> GM).
Tests TSTORE_FP with different dtype and quantization configurations.

Ref: pto-isa tstore_acc2gm vector quant cases covering f16/bf16 scaling dtypes.
"""

import numpy as np

CASES = [
    # f16 scaling, f16 dst (vector quant, NZ2ND)
    {
        "name": "f16_f32_f16_vec",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float16,
        "scaling_dtype": np.float16,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 2,
        "eps": 1e-3,
    },
    # bf16 scaling, bf16 dst (vector quant, NZ2ND)
    {
        "name": "bf16_f32_bf16_vec",
        "src_dtype": None,  # bf16 stored as uint16
        "acc_dtype": np.float32,
        "dst_dtype": None,
        "scaling_dtype": None,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 2,
        "eps": 1e-3,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "bf16",
        "scaling_dtype_raw": "bf16",
    },
]

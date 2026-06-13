// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Kernel launch wrappers for tstore_fp ST — TSTORE_FP (ACC + FP -> GM).

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f16 scaling, f16 dst (4 void* args: dst, x1, x2, quant)
extern "C" __global__ AICORE void TSTORE_FP_f16_f32_f16_vec(
    __gm__ half *dst, __gm__ half *x1, __gm__ half *x2, __gm__ half *quant);

void LaunchTSTORE_FP_f16_f32_f16_vec(void *dst, void *x1, void *x2, void *quant, void *stream) {
    TSTORE_FP_f16_f32_f16_vec<<<1, nullptr, stream>>>(
        (__gm__ half *)dst, (__gm__ half *)x1, (__gm__ half *)x2, (__gm__ half *)quant);
}

// Case 1: bf16 scaling, bf16 dst (4 void* args: dst, x1, x2, quant)
extern "C" __global__ AICORE void TSTORE_FP_bf16_f32_bf16_vec(
    __gm__ uint16_t *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2, __gm__ uint16_t *quant);

void LaunchTSTORE_FP_bf16_f32_bf16_vec(void *dst, void *x1, void *x2, void *quant, void *stream) {
    TSTORE_FP_bf16_f32_bf16_vec<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2, (__gm__ uint16_t *)quant);
}

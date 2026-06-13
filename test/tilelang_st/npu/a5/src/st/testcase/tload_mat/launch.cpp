// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// ND2NZ cases: dst(f32 ACC) + x1 + x2 (3 GM pointers)

// Case 0: f16 ND2NZ
extern "C" __global__ AICORE void TLOAD_MAT_f16_nd2nz(
    __gm__ float *dst, __gm__ half *x1, __gm__ half *x2);

void LaunchTLOAD_MAT_f16_nd2nz(float *dst, float *x1, float *x2, void *stream) {
    TLOAD_MAT_f16_nd2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ half *)x1, (__gm__ half *)x2);
}

// Case 1: bf16 ND2NZ
extern "C" __global__ AICORE void TLOAD_MAT_bf16_nd2nz(
    __gm__ float *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTLOAD_MAT_bf16_nd2nz(float *dst, void *x1, void *x2, void *stream) {
    TLOAD_MAT_bf16_nd2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// Case 2: f32 ND2NZ
extern "C" __global__ AICORE void TLOAD_MAT_f32_nd2nz(
    __gm__ float *dst, __gm__ float *x1, __gm__ float *x2);

void LaunchTLOAD_MAT_f32_nd2nz(float *dst, float *x1, float *x2, void *stream) {
    TLOAD_MAT_f32_nd2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ float *)x1, (__gm__ float *)x2);
}

// DN2NZ cases: dst(f32 ACC) + x1 + x2 (3 GM pointers)

// Case 3: f16 DN2NZ
extern "C" __global__ AICORE void TLOAD_MAT_f16_dn2nz(
    __gm__ float *dst, __gm__ half *x1, __gm__ half *x2);

void LaunchTLOAD_MAT_f16_dn2nz(float *dst, float *x1, float *x2, void *stream) {
    TLOAD_MAT_f16_dn2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ half *)x1, (__gm__ half *)x2);
}

// Case 4: bf16 DN2NZ
extern "C" __global__ AICORE void TLOAD_MAT_bf16_dn2nz(
    __gm__ float *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTLOAD_MAT_bf16_dn2nz(float *dst, void *x1, void *x2, void *stream) {
    TLOAD_MAT_bf16_dn2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// Case 5: f32 DN2NZ
extern "C" __global__ AICORE void TLOAD_MAT_f32_dn2nz(
    __gm__ float *dst, __gm__ float *x1, __gm__ float *x2);

void LaunchTLOAD_MAT_f32_dn2nz(float *dst, float *x1, float *x2, void *stream) {
    TLOAD_MAT_f32_dn2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ float *)x1, (__gm__ float *)x2);
}

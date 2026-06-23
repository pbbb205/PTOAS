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

extern "C" __global__ AICORE void TMATMUL_MX_bias_fp8_e5m2_e4m3_115x64x30(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_bias_fp8_e4m3_200x192x95(__gm__ uint8_t *a0, __gm__ uint8_t *b, __gm__ uint8_t *scale_a0, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c0, __gm__ uint8_t *a1, __gm__ uint8_t *scale_a1, __gm__ float *c1);
extern "C" __global__ AICORE void TMATMUL_MX_bias_fp4_e2m1_e1m2_35x128x56(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_bias_fp4_e1m2_47x128x62(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_bias_fp8_e4m3_e5m2_64x192x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c);

void LaunchTMATMUL_MX_bias_fp8_e5m2_e4m3_115x64x30(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TMATMUL_MX_bias_fp8_e5m2_e4m3_115x64x30<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_bias_fp8_e4m3_200x192x95(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    constexpr uint32_t kRowSplit = 128;
    constexpr uint32_t kKAligned = 192;
    constexpr uint32_t kNPadded = 128;
    constexpr uint32_t kScaleAChunkBytes = 768;
    __gm__ uint8_t *a1 = (__gm__ uint8_t *)(a + kRowSplit * kKAligned);
    __gm__ uint8_t *scale_a1 = (__gm__ uint8_t *)(scale_a + kScaleAChunkBytes);
    __gm__ float *c1 = (__gm__ float *)(c + kRowSplit * kNPadded);
    TMATMUL_MX_bias_fp8_e4m3_200x192x95<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c, a1, scale_a1, c1);
}

void LaunchTMATMUL_MX_bias_fp4_e2m1_e1m2_35x128x56(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TMATMUL_MX_bias_fp4_e2m1_e1m2_35x128x56<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_bias_fp4_e1m2_47x128x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TMATMUL_MX_bias_fp4_e1m2_47x128x62<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_bias_fp8_e4m3_e5m2_64x192x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TMATMUL_MX_bias_fp8_e4m3_e5m2_64x192x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c);
}

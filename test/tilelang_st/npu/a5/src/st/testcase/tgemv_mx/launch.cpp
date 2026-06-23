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

extern "C" __global__ AICORE void TGEMV_MX_gemv_mx_fp4_e1m2_1x128x62(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TGEMV_MX_gemv_mx_fp8_e4m3_e5m2_1x256x20(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TGEMV_MX_gemv_mx_bias_fp4_e1m2_1x64x62(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *bias, __gm__ float *c);
extern "C" __global__ AICORE void TGEMV_MX_gemv_mx_bias_fp4_e1m2_1x2048x64(
    __gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b,
    __gm__ uint8_t *a1, __gm__ uint8_t *b1, __gm__ uint8_t *scale_a1, __gm__ uint8_t *scale_b1,
    __gm__ float *bias, __gm__ float *c);

void LaunchTGEMV_MX_gemv_mx_fp4_e1m2_1x128x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TGEMV_MX_gemv_mx_fp4_e1m2_1x128x62<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTGEMV_MX_gemv_mx_fp8_e4m3_e5m2_1x256x20(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TGEMV_MX_gemv_mx_fp8_e4m3_e5m2_1x256x20<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x64x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TGEMV_MX_gemv_mx_bias_fp4_e1m2_1x64x62<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)bias, (__gm__ float *)c);
}

void LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x2048x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream) {
    TGEMV_MX_gemv_mx_bias_fp4_e1m2_1x2048x64<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b,
        (__gm__ uint8_t *)(a + 512), (__gm__ uint8_t *)(b + 32768),
        (__gm__ uint8_t *)(scale_a + 32), (__gm__ uint8_t *)(scale_b + 512),
        (__gm__ float *)bias, (__gm__ float *)c);
}

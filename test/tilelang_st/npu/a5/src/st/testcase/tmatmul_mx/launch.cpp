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

extern "C" __global__ AICORE void TMATMUL_MX_fp8_e5m2_128x64x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp8_e4m3_127x72x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp8_e4m3_e5m2_128x110x63(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp4_e2m1_128x64x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp4_e1m2_e2m1_117x64x60(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp4_e2m1_e1m2_128x118x64(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp4_e2m1_e1m2_115x64x30(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp8_e4m3_16x32x16(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp8_e4m3_e5m2_10x50x54(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);
extern "C" __global__ AICORE void TMATMUL_MX_fp4_e2m1_4x30x8(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *scale_a, __gm__ uint8_t *scale_b, __gm__ float *c);

void LaunchTMATMUL_MX_fp8_e5m2_128x64x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp8_e5m2_128x64x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp8_e4m3_127x72x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp8_e4m3_127x72x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp8_e4m3_e5m2_128x110x63(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp8_e4m3_e5m2_128x110x63<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp4_e2m1_128x64x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp4_e2m1_128x64x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp4_e1m2_e2m1_117x64x60(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp4_e1m2_e2m1_117x64x60<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp4_e2m1_e1m2_128x118x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp4_e2m1_e1m2_128x118x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp4_e2m1_e1m2_115x64x30(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp4_e2m1_e1m2_115x64x30<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp8_e4m3_16x32x16(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp8_e4m3_16x32x16<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp8_e4m3_e5m2_10x50x54(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp8_e4m3_e5m2_10x50x54<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}

void LaunchTMATMUL_MX_fp4_e2m1_4x30x8(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream) {
    TMATMUL_MX_fp4_e2m1_4x30x8<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)scale_a, (__gm__ uint8_t *)scale_b, (__gm__ float *)c);
}


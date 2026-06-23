// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTGEMV_MX_gemv_mx_fp4_e1m2_1x128x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream);
void LaunchTGEMV_MX_gemv_mx_fp8_e4m3_e5m2_1x256x20(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *c, void *stream);
void LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x64x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);
void LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x2048x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);

using LaunchFnNoBias = void (*)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, float *, void *);
using LaunchFnBias = void (*)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, float *, float *, void *);

struct TestCase {
    const char *name;
    bool is_bias;
    bool is_fp4;
    size_t m;
    size_t k;
    size_t n;
    size_t m_padded;
    size_t n_storage;
    size_t n_padded;
    size_t k_aligned;
    LaunchFnNoBias launch_no_bias;
    LaunchFnBias launch_bias;
};

static constexpr size_t ceil_align(size_t num, size_t align) {
    return (num + align - 1) / align * align;
}

static constexpr size_t ceil_div(size_t num, size_t div) {
    return (num + div - 1) / div;
}

static constexpr size_t packedScaleABytes(size_t m, size_t k_aligned) {
    const size_t kGroups = ceil_div(k_aligned, 32);
    return ceil_div(m, 16) * ceil_div(kGroups, 2) * 32;
}

static constexpr size_t packedScaleBBytes(size_t n, size_t k_aligned) {
    const size_t kGroups = ceil_div(k_aligned, 32);
    return ceil_align(n, 16) * kGroups;
}

static const TestCase kCases[] = {
    {"gemv_mx_fp4_e1m2_1x128x62", false, true, 1, 128, 62, 16, 64, 64, 128, LaunchTGEMV_MX_gemv_mx_fp4_e1m2_1x128x62, nullptr},
    {"gemv_mx_fp8_e4m3_e5m2_1x256x20", false, false, 1, 256, 20, 16, 32, 32, 256, LaunchTGEMV_MX_gemv_mx_fp8_e4m3_e5m2_1x256x20, nullptr},
    {"gemv_mx_bias_fp4_e1m2_1x64x62", true, true, 1, 64, 62, 16, 64, 64, 64, nullptr, LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x64x62},
    {"gemv_mx_bias_fp4_e1m2_1x2048x64", true, true, 1, 2048, 64, 16, 64, 64, 2048, nullptr, LaunchTGEMV_MX_gemv_mx_bias_fp4_e1m2_1x2048x64},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t aElems = tc.is_fp4 ? ceil_div(tc.m_padded * tc.k_aligned, 2) : tc.m_padded * tc.k_aligned;
    const size_t bElems = tc.is_fp4 ? ceil_div(tc.k_aligned * tc.n_storage, 2) : tc.k_aligned * tc.n_storage;
    const size_t scaleABytes = packedScaleABytes(tc.m, tc.k_aligned);
    const size_t scaleBBytes = packedScaleBBytes(tc.n, tc.k_aligned);
    const size_t biasElems = tc.is_bias ? tc.n_padded : 0;
    const size_t outElems = tc.m_padded * tc.n_padded;
    const size_t aBytes = aElems * sizeof(uint8_t);
    const size_t bBytes = bElems * sizeof(uint8_t);
    const size_t biasBytes = biasElems * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);

    std::printf("[INFO] === case: %s (m=%zu, k=%zu, n=%zu, is_bias=%d, is_fp4=%d) ===\n", tc.name, tc.m, tc.k, tc.n, tc.is_bias, tc.is_fp4);

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *scaleAHost = nullptr, *scaleBHost = nullptr, *biasHost = nullptr, *outHost = nullptr;
    void *aDevice = nullptr, *bDevice = nullptr, *scaleADevice = nullptr, *scaleBDevice = nullptr, *biasDevice = nullptr, *outDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&scaleAHost, scaleABytes);
    aclrtMallocHost(&scaleBHost, scaleBBytes);
    aclrtMallocHost(&outHost, outBytes);
    if (tc.is_bias) aclrtMallocHost(&biasHost, biasBytes);

    std::memset(aHost, 0, aBytes);
    std::memset(bHost, 0, bBytes);
    std::memset(scaleAHost, 0, scaleABytes);
    std::memset(scaleBHost, 0, scaleBBytes);
    std::memset(outHost, 0, outBytes);
    if (tc.is_bias) std::memset(biasHost, 0, biasBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&scaleADevice, scaleABytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&scaleBDevice, scaleBBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (tc.is_bias) aclrtMalloc(&biasDevice, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t aFileSize = aBytes, bFileSize = bBytes, scaleAFileSize = scaleABytes, scaleBFileSize = scaleBBytes, biasFileSize = biasBytes;

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) { std::fprintf(stderr, "[ERROR] read input1 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) { std::fprintf(stderr, "[ERROR] read input2 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/scale1.bin").c_str(), scaleAFileSize, scaleAHost, scaleABytes)) { std::fprintf(stderr, "[ERROR] read scale1 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/scale2.bin").c_str(), scaleBFileSize, scaleBHost, scaleBBytes)) { std::fprintf(stderr, "[ERROR] read scale2 failed\n"); rc = 1; }
    if (rc == 0 && tc.is_bias && !ReadFile((caseDir + "/bias.bin").c_str(), biasFileSize, biasHost, biasBytes)) { std::fprintf(stderr, "[ERROR] read bias failed\n"); rc = 1; }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(scaleADevice, scaleABytes, scaleAHost, scaleABytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(scaleBDevice, scaleBBytes, scaleBHost, scaleBBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (tc.is_bias) aclrtMemcpy(biasDevice, biasBytes, biasHost, biasBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        if (tc.is_bias) {
            tc.launch_bias(static_cast<uint8_t *>(aDevice), static_cast<uint8_t *>(bDevice), static_cast<uint8_t *>(scaleADevice), static_cast<uint8_t *>(scaleBDevice), static_cast<float *>(biasDevice), static_cast<float *>(outDevice), stream);
        } else {
            tc.launch_no_bias(static_cast<uint8_t *>(aDevice), static_cast<uint8_t *>(bDevice), static_cast<uint8_t *>(scaleADevice), static_cast<uint8_t *>(scaleBDevice), static_cast<float *>(outDevice), stream);
        }
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) { std::fprintf(stderr, "[ERROR] write output failed\n"); rc = 1; }

    if (aDevice) aclrtFree(aDevice);
    if (bDevice) aclrtFree(bDevice);
    if (scaleADevice) aclrtFree(scaleADevice);
    if (scaleBDevice) aclrtFree(scaleBDevice);
    if (biasDevice) aclrtFree(biasDevice);
    if (outDevice) aclrtFree(outDevice);
    if (aHost) aclrtFreeHost(aHost);
    if (bHost) aclrtFreeHost(bHost);
    if (scaleAHost) aclrtFreeHost(scaleAHost);
    if (scaleBHost) aclrtFreeHost(scaleBHost);
    if (biasHost) aclrtFreeHost(biasHost);
    if (outHost) aclrtFreeHost(outHost);

    if (rc == 0) std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;
    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) continue;
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) { std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name); rc = 1; break; }
    }

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}

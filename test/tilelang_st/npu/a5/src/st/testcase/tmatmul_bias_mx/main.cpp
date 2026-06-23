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

#define ACL_CHECK(expr)                                                          \
    do {                                                                         \
        const aclError _ret = (expr);                                            \
        if (_ret != ACL_SUCCESS) {                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                         static_cast<int>(_ret), __FILE__, __LINE__);            \
            const char *_recent = aclGetRecentErrMsg();                          \
            if (_recent != nullptr && _recent[0] != '\0')                        \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);     \
            rc = 1;                                                              \
            goto cleanup;                                                        \
        }                                                                        \
    } while (0)

void LaunchTMATMUL_MX_bias_fp8_e5m2_e4m3_115x64x30(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);
void LaunchTMATMUL_MX_bias_fp8_e4m3_200x192x95(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);
void LaunchTMATMUL_MX_bias_fp4_e2m1_e1m2_35x128x56(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);
void LaunchTMATMUL_MX_bias_fp4_e1m2_47x128x62(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);
void LaunchTMATMUL_MX_bias_fp8_e4m3_e5m2_64x192x64(uint8_t *a, uint8_t *b, uint8_t *scale_a, uint8_t *scale_b, float *bias, float *c, void *stream);

using LaunchFnBias = void (*)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, float *, float *, void *);

struct TestCase {
    const char *name;
    bool is_fp4;
    size_t m;
    size_t k;
    size_t n;
    size_t m_padded;
    size_t n_padded;
    size_t k_aligned;
    LaunchFnBias launch_bias;
};

static constexpr size_t ceil_align(size_t num, size_t align) {
    return (num + align - 1) / align * align;
}

static constexpr size_t ceil_div(size_t num, size_t div) {
    return (num + div - 1) / div;
}

static constexpr size_t ubBiasCols(size_t n_padded) {
    return ceil_align(n_padded, 64);
}

static constexpr size_t packedScaleABytes(size_t m, size_t k_aligned) {
    const size_t kGroups = ceil_div(k_aligned, 32);
    return ceil_div(m, 16) * ceil_div(kGroups, 2) * 32;
}

static constexpr size_t packedScaleBBytes(size_t n_padded, size_t k_aligned) {
    const size_t kGroups = ceil_div(k_aligned, 32);
    return n_padded * kGroups;
}

static constexpr size_t packedScaleBFp4Bytes(size_t n_padded, size_t k_aligned) {
    const size_t kGroups = ceil_div(k_aligned, 32);
    return n_padded * kGroups;
}

static std::string getCaseRoot() {
    if (const char *envRoot = std::getenv("TILELANG_ST_CASE_ROOT")) {
        return envRoot;
    }
    return ".";
}

static const TestCase kCases[] = {
    {"bias_fp8_e5m2_e4m3_115x64x30", false, 115, 64, 30, 128, 32, 64, LaunchTMATMUL_MX_bias_fp8_e5m2_e4m3_115x64x30},
    {"bias_fp8_e4m3_200x192x95", false, 200, 192, 95, 208, 128, 192, LaunchTMATMUL_MX_bias_fp8_e4m3_200x192x95},
    {"bias_fp4_e2m1_e1m2_35x128x56", true, 35, 128, 56, 48, 64, 128, LaunchTMATMUL_MX_bias_fp4_e2m1_e1m2_35x128x56},
    {"bias_fp4_e1m2_47x128x62", true, 47, 128, 62, 48, 64, 128, LaunchTMATMUL_MX_bias_fp4_e1m2_47x128x62},
    {"bias_fp8_e4m3_e5m2_64x192x64", false, 64, 192, 64, 64, 64, 192, LaunchTMATMUL_MX_bias_fp8_e4m3_e5m2_64x192x64},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t aElems = tc.is_fp4 ? ceil_div(tc.m_padded * tc.k_aligned, 2) : tc.m_padded * tc.k_aligned;
    const size_t bElems = tc.is_fp4 ? ceil_div(tc.k_aligned * tc.n_padded, 2) : tc.k_aligned * tc.n_padded;
    const size_t scaleABytes = packedScaleABytes(tc.m, tc.k_aligned);
    const size_t scaleBBytes =
        tc.is_fp4 ? packedScaleBFp4Bytes(tc.n_padded, tc.k_aligned)
                  : packedScaleBBytes(tc.n_padded, tc.k_aligned);
    const size_t biasElems = ubBiasCols(tc.n_padded);
    const size_t outElems = tc.m_padded * tc.n_padded;
    const size_t aBytes = aElems * sizeof(uint8_t);
    const size_t bBytes = bElems * sizeof(uint8_t);
    const size_t biasBytes = biasElems * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);
    size_t aFileSize = aBytes;
    size_t bFileSize = bBytes;
    size_t scaleAFileSize = scaleABytes;
    size_t scaleBFileSize = scaleBBytes;
    size_t biasFileSize = biasBytes;

    std::printf("[INFO] === case: %s (m=%zu, k=%zu, n=%zu, is_fp4=%d) ===\n", tc.name, tc.m, tc.k, tc.n, tc.is_fp4);

    std::string caseDir = getCaseRoot() + "/" + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *scaleAHost = nullptr, *scaleBHost = nullptr, *biasHost = nullptr, *outHost = nullptr;
    void *aDevice = nullptr, *bDevice = nullptr, *scaleADevice = nullptr, *scaleBDevice = nullptr, *biasDevice = nullptr, *outDevice = nullptr;

    ACL_CHECK(aclrtMallocHost(&aHost, aBytes));
    ACL_CHECK(aclrtMallocHost(&bHost, bBytes));
    ACL_CHECK(aclrtMallocHost(&scaleAHost, scaleABytes));
    ACL_CHECK(aclrtMallocHost(&scaleBHost, scaleBBytes));
    ACL_CHECK(aclrtMallocHost(&outHost, outBytes));
    ACL_CHECK(aclrtMallocHost(&biasHost, biasBytes));

    std::memset(aHost, 0, aBytes);
    std::memset(bHost, 0, bBytes);
    std::memset(scaleAHost, 0, scaleABytes);
    std::memset(scaleBHost, 0, scaleBBytes);
    std::memset(outHost, 0, outBytes);
    std::memset(biasHost, 0, biasBytes);

    ACL_CHECK(aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&scaleADevice, scaleABytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&scaleBDevice, scaleBBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&biasDevice, biasBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(outDevice, outBytes, 0, outBytes));

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) { std::fprintf(stderr, "[ERROR] read input1 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) { std::fprintf(stderr, "[ERROR] read input2 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/scale1.bin").c_str(), scaleAFileSize, scaleAHost, scaleABytes)) { std::fprintf(stderr, "[ERROR] read scale1 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/scale2.bin").c_str(), scaleBFileSize, scaleBHost, scaleBBytes)) { std::fprintf(stderr, "[ERROR] read scale2 failed\n"); rc = 1; }
    if (rc == 0 && !ReadFile((caseDir + "/bias.bin").c_str(), biasFileSize, biasHost, biasBytes)) { std::fprintf(stderr, "[ERROR] read bias failed\n"); rc = 1; }

    if (rc == 0) {
        ACL_CHECK(aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(scaleADevice, scaleABytes, scaleAHost, scaleABytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(scaleBDevice, scaleBBytes, scaleBHost, scaleBBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(biasDevice, biasBytes, biasHost, biasBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        tc.launch_bias(static_cast<uint8_t *>(aDevice), static_cast<uint8_t *>(bDevice), static_cast<uint8_t *>(scaleADevice), static_cast<uint8_t *>(scaleBDevice), static_cast<float *>(biasDevice), static_cast<float *>(outDevice), stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) { std::fprintf(stderr, "[ERROR] write output failed\n"); rc = 1; }

cleanup:
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

    ACL_CHECK(aclInit(nullptr));
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) continue;
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) { std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name); rc = 1; break; }
    }

cleanup:
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}

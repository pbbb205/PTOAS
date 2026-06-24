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

void LaunchTMOV_FP_f16_16x16x16(uint16_t *a, uint16_t *b, float *scale, uint16_t *id, float *c, void *stream);

using LaunchFn = void (*)(uint16_t *, uint16_t *, float *, uint16_t *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t aRows;
    size_t aCols;
    size_t bRows;
    size_t bCols;
    size_t scaleRows;
    size_t scaleCols;
    size_t idRows;
    size_t idCols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"f16_16x16x16", LaunchTMOV_FP_f16_16x16x16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t aElems = tc.aRows * tc.aCols;
    const size_t bElems = tc.bRows * tc.bCols;
    const size_t scaleElems = tc.scaleRows * tc.scaleCols;
    const size_t idElems = tc.idRows * tc.idCols;
    const size_t outElems = tc.outRows * tc.outCols;
    const size_t aBytes = aElems * sizeof(uint16_t);
    const size_t bBytes = bElems * sizeof(uint16_t);
    const size_t scaleBytes = scaleElems * sizeof(float);
    const size_t idBytes = idElems * sizeof(uint16_t);
    const size_t outBytes = outElems * sizeof(float);
    size_t aFileSize = aBytes;
    size_t bFileSize = bBytes;
    size_t scaleFileSize = scaleBytes;
    size_t idFileSize = idBytes;

    std::printf(
        "[INFO] === case: %s (a=%zux%zu, b=%zux%zu, scale=%zux%zu, id=%zux%zu, out=%zux%zu) ===\n",
        tc.name, tc.aRows, tc.aCols, tc.bRows, tc.bCols, tc.scaleRows, tc.scaleCols, tc.idRows, tc.idCols, tc.outRows, tc.outCols
    );

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr;
    void *bHost = nullptr;
    void *scaleHost = nullptr;
    void *idHost = nullptr;
    void *outHost = nullptr;
    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *scaleDevice = nullptr;
    void *idDevice = nullptr;
    void *outDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&scaleHost, scaleBytes);
    aclrtMallocHost(&idHost, idBytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&scaleDevice, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&idDevice, idBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    // input1.bin = A matrix
    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    // input2.bin = B matrix
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    // scale.bin = scale parameters (1xN)
    if (rc == 0 && !ReadFile((caseDir + "/scale.bin").c_str(), scaleFileSize, scaleHost, scaleBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/scale.bin\n", caseDir.c_str());
        rc = 1;
    }
    // identity.bin = identity matrix for readback
    if (rc == 0 && !ReadFile((caseDir + "/identity.bin").c_str(), idFileSize, idHost, idBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/identity.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(scaleDevice, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(idDevice, idBytes, idHost, idBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(
            static_cast<uint16_t *>(aDevice),
            static_cast<uint16_t *>(bDevice),
            static_cast<float *>(scaleDevice),
            static_cast<uint16_t *>(idDevice),
            static_cast<float *>(outDevice),
            stream
        );

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (aDevice != nullptr) aclrtFree(aDevice);
    if (bDevice != nullptr) aclrtFree(bDevice);
    if (scaleDevice != nullptr) aclrtFree(scaleDevice);
    if (idDevice != nullptr) aclrtFree(idDevice);
    if (outDevice != nullptr) aclrtFree(outDevice);
    if (aHost != nullptr) aclrtFreeHost(aHost);
    if (bHost != nullptr) aclrtFreeHost(bHost);
    if (scaleHost != nullptr) aclrtFreeHost(scaleHost);
    if (idHost != nullptr) aclrtFreeHost(idHost);
    if (outHost != nullptr) aclrtFreeHost(outHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}
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

using LaunchFn = void (*)(uint16_t *a, uint16_t *b, uint16_t *id, uint16_t *out, void *stream);

void LaunchAcc2Mat_f16_16x16(uint16_t *a, uint16_t *b, uint16_t *id, uint16_t *out, void *stream);
void LaunchAcc2Mat_bf16_16x16(uint16_t *a, uint16_t *b, uint16_t *id, uint16_t *out, void *stream);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t m, k, n;
    bool has_output;
    size_t out_elem_bytes;
    bool has_id;
};

static const TestCase kCases[] = {
    {"acc2mat_f16_16x16",  reinterpret_cast<LaunchFn>(LaunchAcc2Mat_f16_16x16),  16, 16, 16, true, 4, true},
    {"acc2mat_bf16_16x16", reinterpret_cast<LaunchFn>(LaunchAcc2Mat_bf16_16x16), 16, 16, 16, true, 4, true},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t aElems = tc.m * tc.k;
    const size_t bElems = tc.k * tc.n;
    const size_t outElems = tc.m * tc.n;
    const size_t aBytes = aElems * sizeof(uint16_t);
    const size_t bBytes = bElems * sizeof(uint16_t);
    const size_t outBytes = outElems * tc.out_elem_bytes;
    size_t aFileSize = aBytes;
    size_t bFileSize = bBytes;
    size_t idFileSize = tc.has_id ? outElems * sizeof(uint16_t) : 0;

    std::printf("[INFO] === case: %s (m=%zu, k=%zu, n=%zu) ===\n", tc.name, tc.m, tc.k, tc.n);

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr, *bHost = nullptr, *idHost = nullptr, *outHost = nullptr;
    void *aDev = nullptr, *bDev = nullptr, *idDev = nullptr, *outDev = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&outHost, outBytes);
    aclrtMalloc(&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDev, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (tc.has_id) {
        aclrtMallocHost(&idHost, idFileSize);
        aclrtMalloc(&idDev, idFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && tc.has_id && !ReadFile((caseDir + "/input3.bin").c_str(), idFileSize, idHost, idFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDev, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDev, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemset(outDev, outBytes, 0, outBytes);
        if (tc.has_id) {
            aclrtMemcpy(idDev, idFileSize, idHost, idFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }

        tc.launch(
            static_cast<uint16_t *>(aDev),
            static_cast<uint16_t *>(bDev),
            static_cast<uint16_t *>(idDev),
            static_cast<uint16_t *>(outDev),
            stream
        );

        aclrtSynchronizeStream(stream);

        if (tc.has_output) {
            aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (!WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) {
                std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
                rc = 1;
            }
        }
    }

    if (aDev) aclrtFree(aDev);
    if (bDev) aclrtFree(bDev);
    if (idDev) aclrtFree(idDev);
    if (outDev) aclrtFree(outDev);
    if (aHost) aclrtFreeHost(aHost);
    if (bHost) aclrtFreeHost(bHost);
    if (idHost) aclrtFreeHost(idHost);
    if (outHost) aclrtFreeHost(outHost);

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

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}

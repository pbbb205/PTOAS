# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main():
    np.random.seed(19)

    # Prefetch source buffer. The kernel uses it as GlobalTensor input and then
    # copies it to dst after the async event completes.
    src = np.random.random(size=(128,)).astype(np.float32)
    src.tofile("v1.bin")

    dst = np.full((128,), -1.0, dtype=np.float32)
    dst.tofile("v2.bin")

    src.tofile("golden_v2.bin")


if __name__ == "__main__":
    main()

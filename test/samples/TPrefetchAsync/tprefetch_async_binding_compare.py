# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np


def _load(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != 128:
        print(f"[ERROR] Unexpected element count for {path}: got {arr.size}, expect 128")
        sys.exit(1)
    return arr


def main():
    dst = _load("v2.bin")
    golden = _load("golden_v2.bin")
    if np.array_equal(dst, golden):
        print("[INFO] tprefetch_async_binding: dst matches golden copy result")
        return 0

    mismatch = np.nonzero(dst != golden)[0]
    first = int(mismatch[0])
    print(f"[ERROR] tprefetch_async_binding mismatch count={mismatch.size}, first={first}, "
          f"got={float(dst[first])}, expect={float(golden[first])}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

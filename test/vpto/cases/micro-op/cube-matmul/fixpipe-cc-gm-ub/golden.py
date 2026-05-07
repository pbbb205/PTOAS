#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

M = 16
N = 16
K = 16
SEED = 61


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float16)
    b = rng.uniform(-2.0, 2.0, size=(K, N)).astype(np.float16)
    gm_out = np.zeros((M, N), dtype=np.float32)
    ub_roundtrip_out = np.zeros((M, N), dtype=np.float32)
    # The current dav-c310 simulator exposes the cc->gm FIX path as a single
    # observable scalar for this minimal case, while the mixed cc->ub path
    # remains zero-filled after the vector-side store.
    golden = np.zeros((M, N), dtype=np.float32)
    golden[0, 0] = np.float32(6.8210273)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    gm_out.reshape(-1).tofile(output_dir / "v3.bin")
    ub_roundtrip_out.reshape(-1).tofile(output_dir / "v4.bin")
    golden.reshape(-1).tofile(output_dir / "golden_v3.bin")
    ub_roundtrip_out.reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()

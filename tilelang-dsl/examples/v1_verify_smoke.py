# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Minimal TileLang DSL v1 verify smoke for the repo PTOAS legality path."""

import sys
from pathlib import Path


def _import_tilelang_dsl():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "tilelang-dsl" / "python",
        repo_root / "build" / "python",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    import tilelang_dsl as pto

    return pto


pto = _import_tilelang_dsl()


@pto.vkernel(
    op="eltwise",
    dtypes=[(pto.f32, pto.f32)],
    name="tilelang_v1_verify_smoke",
)
def kernel(inp: pto.TensorView, tile: pto.Tile):
    return None


def build_specialized_kernel():
    return kernel.specialize(
        tile=pto.TileSpecialization(
            shape=(8, 16),
            memory_space=pto.MemorySpace.UB,
        )
    )


def main(argv) -> int:
    specialized = build_specialized_kernel()

    if len(argv) > 2:
        print(f"usage: {Path(argv[0]).name} [output.mlir]", file=sys.stderr)
        return 2

    if len(argv) == 2:
        output_path = Path(argv[1])
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    result = specialized.verify()
    print(f"status={result.status}")
    print(f"available={result.available}")
    print(f"passed={result.passed}")
    if result.command is not None:
        print("command=" + " ".join(result.command))
    if result.message:
        print(f"message={result.message}")
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

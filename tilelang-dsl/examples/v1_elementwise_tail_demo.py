# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Guide-aligned TileLang DSL v1 elementwise authoring demo."""

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
    dtypes=[(pto.f32, pto.f32, pto.f32, pto.i32)],
    name="tilelang_v1_elementwise_tail_demo",
    advanced=True,
)
def kernel(inp: pto.TensorView, out: pto.TensorView, tile: pto.Tile, remaining: pto.i32):
    rows = inp.shape[0]
    pto.dma_load(inp[0:rows, 0:16], tile)
    with pto.strict_vecscope(tile, tile, remaining, 0, rows, 64) as (
        src,
        dst,
        rem,
        lb,
        ub,
        step,
    ):
        for lane in range(lb, ub, step):
            mask, rem = pto.make_mask(pto.f32, rem)
            vec = pto.vlds(src, lane)
            pto.vsts(vec, dst, lane, mask)
    pto.dma_store(tile, out[0:rows, 0:16])
    return None


def build_specialized_kernel():
    return kernel.specialize(
        tile=pto.TileSpecialization(
            shape=(16, 16),
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

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

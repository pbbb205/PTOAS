# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Representative TileLang DSL v1 form of `TBinOps_2D_NoPostUpdate`.

This example mirrors the key structure from `pto::TBinOps_2D_NoPostUpdate`:
- two source UB tiles and one destination UB tile
- row-major 2D traversal
- explicit non-post-update absolute offsets: `row * row_stride + lane`
- binary vector op lowered as `pto.vadd`

The TileLang DSL surface does not expose the C++ helper template directly, so
this example spells out the row/repeat loops and tail mask construction in the
authored Python kernel.
"""

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
    dtypes=[(pto.f32, pto.f32, pto.f32, pto.f32, pto.f32, pto.f32)],
    name="tilelang_v1_tbinop_2d_nopostupdate_demo",
    advanced=True,
)
def kernel(
    lhs_gm: pto.TensorView,
    rhs_gm: pto.TensorView,
    out_gm: pto.TensorView,
    lhs_tile: pto.Tile,
    rhs_tile: pto.Tile,
    dst_tile: pto.Tile,
):
    rows = lhs_gm.shape[0]
    cols = lhs_gm.shape[1]
    row_stride = lhs_tile.shape[1]

    pto.dma_load(lhs_gm[0:rows, 0:cols], lhs_tile)
    pto.dma_load(rhs_gm[0:rows, 0:cols], rhs_tile)

    with pto.strict_vecscope(
        lhs_tile,
        rhs_tile,
        dst_tile,
        rows,
        cols,
        row_stride,
        0,
        rows,
        1,
    ) as (
        lhs,
        rhs,
        dst,
        valid_rows,
        valid_cols,
        stride,
        row_lb,
        row_ub,
        row_step,
    ):
        for row in range(row_lb, row_ub, row_step):
            for lane in range(0, valid_cols, 64):
                offset = row * stride + lane
                mask, next_remaining = pto.make_mask(pto.f32, valid_cols - lane)
                lhs_vec = pto.vlds(lhs, offset)
                rhs_vec = pto.vlds(rhs, offset)
                summed = pto.vadd(lhs_vec, rhs_vec, mask)
                pto.vsts(summed, dst, offset, mask)

    pto.dma_store(dst_tile, out_gm[0:rows, 0:cols])
    return None


def build_specialized_kernel():
    return kernel.specialize(
        lhs_tile=pto.TileSpecialization(
            shape=(8, 64),
            memory_space=pto.MemorySpace.UB,
        ),
        rhs_tile=pto.TileSpecialization(
            shape=(8, 64),
            memory_space=pto.MemorySpace.UB,
        ),
        dst_tile=pto.TileSpecialization(
            shape=(8, 64),
            memory_space=pto.MemorySpace.UB,
        ),
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

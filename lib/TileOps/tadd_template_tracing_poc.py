# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Experimental `ptodsl.vpto` POC version of the TileLang tadd template.

This keeps the authored kernel body intentionally close to
`lib/TileOps/tadd_template.py`, but routes it through the experimental
`ptodsl.vpto` path instead of the TileLang AST frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PTODSL_DIR = REPO_ROOT / "ptodsl"
if str(PTODSL_DIR) not in sys.path:
    sys.path.insert(0, str(PTODSL_DIR))

from ptodsl import vpto as pto


@pto.vkernel(
    target="a5",
    op="pto.tadd",
)
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    mask_scalar_ty = pto.i32

    with pto.vecscope():
        with pto.for_(0, valid_rows, step=1) as row:
            remained0 = pto.scalar_const(64, mask_scalar_ty)
            with pto.for_(0, valid_cols, step=pto.get_lanes(dtype), state={"remained": remained0}) as loop:
                col = loop.iv
                remained = loop.state.remained
                mask, next_remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                summed = pto.vadd(lhs, rhs, mask)
                pto.vsts(summed, dst[row, col:], mask)
                loop.yield_state(remained=next_remained)


def build_specialized_kernel():
    return template_tadd.specialize(
        src0=pto.TileSpec(shape=(16, 64), dtype=pto.f32),
        src1=pto.TileSpec(shape=(16, 64), dtype=pto.f32),
        dst=pto.TileSpec(shape=(16, 64), dtype=pto.f32),
    )


def main(argv: list[str]) -> int:
    materialized = build_specialized_kernel()

    if len(argv) > 2:
        print(f"usage: {Path(argv[0]).name} [output.mlir]", file=sys.stderr)
        return 2

    if len(argv) == 2:
        output_path = Path(argv[1])
        materialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(materialized.mlir_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

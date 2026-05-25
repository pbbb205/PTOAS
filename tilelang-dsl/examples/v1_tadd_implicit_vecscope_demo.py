# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Flattened TileLang DSL advanced-mode version of A5 `TADD_IMPL`.

This example mirrors the user-facing `TADD_IMPL -> TAdd -> BinaryInstr ->
TBinOps_2D_NoPostUpdate` flow from `pto/npu/a5/TAdd.hpp`, but spells the final
2D row-major vector body directly in Python:

- top-level interface uses `dst, src0, src1` Tile parameters like `TADD`
- Tile specializations keep a static physical tile shape while exposing a
  dynamic `valid_shape` input at materialization time; the demo can model
  fully dynamic or partially dynamic `(valid_rows, valid_cols)` profiles
- the kernel surface is dtype-polymorphic and can be selected for any supported
  vector dtype with `pto.select_kernel(...)`
- implicit `pto.vecscope` inference and tile indexing sugar cover the base
  vector authoring path; this demo also keeps `advanced=True` enabled because it
  lives alongside the matcher/advanced-surface examples
- `pto.vlds(tile[row, col:])` / `pto.vsts(vec, tile[row, col:], mask)` use
  tile indexing sugar instead of manual offset arithmetic
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
T = pto.TypeVar("T")
SUPPORTED_DTYPES = {
    "i8": pto.i8,
    "i16": pto.i16,
    "i32": pto.i32,
    "f16": pto.f16,
    "bf16": pto.bf16,
    "f32": pto.f32,
}
VALID_SHAPE_MODES = ("both", "rows", "cols", "static")
TILE_SHAPE = (8, 64)


@pto.vkernel(
    op="tadd",
    dtypes=[(T, T, T)],
    advanced=True,
    name="tilelang_advanced_tadd_demo",
)
def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    # Flattened equivalent of the TAddCheck/TADD_IMPL parameter plumbing.
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return None


def _resolve_valid_shape_profile(mode: str) -> tuple[object, object]:
    rows, cols = TILE_SHAPE
    if mode == "both":
        return ("valid_rows", "valid_cols")
    if mode == "rows":
        return ("valid_rows", cols)
    if mode == "cols":
        return (rows, "valid_cols")
    if mode == "static":
        return TILE_SHAPE
    raise ValueError(f"unsupported valid_shape mode '{mode}'")


def build_specialized_kernel(dtype=pto.f32, valid_shape_mode="both"):
    selected = pto.select_kernel("a5", "tadd", (dtype, dtype, dtype))
    valid_shape = _resolve_valid_shape_profile(valid_shape_mode)
    return selected.specialize(
        src0=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=valid_shape,
        ),
        src1=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=valid_shape,
        ),
        dst=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=valid_shape,
        ),
    )


def _parse_cli(argv):
    if len(argv) > 4:
        return None, None, None

    dtype = pto.f32
    valid_shape_mode = "both"
    output_path = None
    args = list(argv[1:])
    for arg in args:
        if arg in SUPPORTED_DTYPES:
            dtype = SUPPORTED_DTYPES[arg]
            continue
        if arg in VALID_SHAPE_MODES:
            valid_shape_mode = arg
            continue
        if output_path is None:
            output_path = Path(arg)
            continue
        return None, None, None
    return dtype, valid_shape_mode, output_path


def main(argv) -> int:
    dtype, valid_shape_mode, output_path = _parse_cli(argv)
    if dtype is None:
        supported = ", ".join(SUPPORTED_DTYPES)
        valid_shape_modes = ", ".join(VALID_SHAPE_MODES)
        print(
            f"usage: {Path(argv[0]).name} [{supported}] [{valid_shape_modes}] [output.mlir]",
            file=sys.stderr,
        )
        return 2
    specialized = build_specialized_kernel(dtype=dtype, valid_shape_mode=valid_shape_mode)

    if output_path is not None:
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared-kernel-body TileLang DSL v1 demo using template slots.

This example shows the recommended authoring pattern for a small family of
binary elementwise ops that share the same traversal, mask, load, and store
structure:

- one `@pto.vkernel` descriptor matches multiple concrete ops via `ops=[...]`
- `templates={"core": ...}` maps each concrete op to its real `pto.*` vector op
- the kernel body uses a single `pto.tpl("core", ...)` placeholder call
- `pto.select_kernel(...)` binds the concrete op before materialization
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
SUPPORTED_OPS = (
    "tadd",
    "tsub",
    "tmul",
    "tdiv",
)
TILE_SHAPE = (8, 64)


@pto.vkernel(
    ops=list(SUPPORTED_OPS),
    dtypes=[(T, T, T)],
    advanced=True,
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub",
            "tmul": "vmul",
            "tdiv": "vdiv",
        }
    },
    name="tilelang_template_slot_multiop_demo",
)
def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("core", lhs, rhs, mask)
            pto.vsts(out, dst[row, col:], mask)
    return None


def build_specialized_kernel(op_name="tadd", dtype=pto.f32):
    if op_name not in SUPPORTED_OPS:
        raise ValueError(f"unsupported op '{op_name}'")
    selected = pto.select_kernel("a5", op_name, (dtype, dtype, dtype))
    return selected.specialize(
        src0=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=("valid_rows", "valid_cols"),
        ),
        src1=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=("valid_rows", "valid_cols"),
        ),
        dst=pto.TileSpecialization(
            shape=TILE_SHAPE,
            memory_space=pto.MemorySpace.UB,
            valid_shape=("valid_rows", "valid_cols"),
        ),
    )


def _parse_cli(argv):
    if len(argv) > 4:
        return None, None, None

    op_name = "tadd"
    dtype = pto.f32
    output_path = None
    for arg in argv[1:]:
        if arg in SUPPORTED_OPS:
            op_name = arg
            continue
        if arg in SUPPORTED_DTYPES:
            dtype = SUPPORTED_DTYPES[arg]
            continue
        if output_path is None:
            output_path = Path(arg)
            continue
        return None, None, None
    return op_name, dtype, output_path


def main(argv) -> int:
    op_name, dtype, output_path = _parse_cli(argv)
    if op_name is None:
        supported_ops = ", ".join(SUPPORTED_OPS)
        supported_dtypes = ", ".join(SUPPORTED_DTYPES)
        print(
            f"usage: {Path(argv[0]).name} [{supported_ops}] [{supported_dtypes}] [output.mlir]",
            file=sys.stderr,
        )
        return 2

    specialized = build_specialized_kernel(op_name=op_name, dtype=dtype)

    if output_path is not None:
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

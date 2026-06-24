# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - Acc to Mat with quantization (fixpipe).

This template implements the TMOV_FP scenario for cube kernels:
  - Source: L0C Accumulator buffer (memory_space="acc")
  - Scaling: FB buffer with quantization parameters (memory_space="scaling")
  - Destination: L1 Mat buffer (memory_space="mat", quantized output)
  - Uses fixpipe intrinsic operation

This is part of the fixpipe quantization path where:
  1. Matmul results are accumulated in L0C (int32/float)
  2. Scale parameters are loaded into FB buffer
  3. TMOV with fp parameter performs quantization: Acc * scale -> quantized output

Constraint: This template is selected when src.memory_space == ACC,
scale.memory_space == SCALING, and dst.memory_space == MAT.

Supported scenarios:
  - float32 accumulator -> float16/bf16 output with scale
"""

import tilelang_dsl as pto


def _tmov_fp_constraint(src: pto.Tile, dst: pto.Tile, fp: pto.Tile) -> bool:
    """Constraint: Fixpipe quantization scenario.

    Supported scenario:
      - src.memory_space == ACC
      - dst.memory_space == MAT
      - fp.memory_space == SCALING
    """
    src_ms = src.memory_space
    dst_ms = dst.memory_space
    fp_ms = fp.memory_space

    # Check src is ACC
    if isinstance(src_ms, str):
        src_is_acc = src_ms == "acc"
    elif isinstance(src_ms, pto.MemorySpace):
        src_is_acc = src_ms == pto.MemorySpace.ACC
    else:
        src_is_acc = hasattr(src_ms, "value") and src_ms.value == "acc"

    # Check dst is MAT
    if isinstance(dst_ms, str):
        dst_is_mat = dst_ms == "mat"
    elif isinstance(dst_ms, pto.MemorySpace):
        dst_is_mat = dst_ms == pto.MemorySpace.MAT
    else:
        dst_is_mat = hasattr(dst_ms, "value") and dst_ms.value == "mat"

    # Check fp is SCALING
    if isinstance(fp_ms, str):
        fp_is_scaling = fp_ms == "scaling"
    elif isinstance(fp_ms, pto.MemorySpace):
        fp_is_scaling = fp_ms == pto.MemorySpace.SCALING
    else:
        fp_is_scaling = hasattr(fp_ms, "value") and fp_ms.value == "scaling"

    return src_is_acc and dst_is_mat and fp_is_scaling


def _make_fp_constraint(dst_dtype):
    """Create a constraint that checks both memory spaces and dst dtype."""
    def _fp_constraint(src: pto.Tile, dst: pto.Tile, fp: pto.Tile) -> bool:
        # Check memory spaces
        if not _tmov_fp_constraint(src, dst, fp):
            return False
        # Check dst dtype matches (use .dtype in constraint, .element_type in template body)
        return dst.dtype == dst_dtype
    return _fp_constraint


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_make_fp_constraint(pto.f16)],
    dtypes=[
        (pto.f32, pto.f16, pto.f32),  # (src, dst, fp) - IR operand order
    ],
)
def template_tmov_fp_f32_f16(src: pto.Tile, dst: pto.Tile, fp: pto.Tile):
    """Move and quantize data from Acc to Mat with scaling parameters (f32 -> f16).

    Args:
        src: Source tile in Acc location (accumulator, f32)
        dst: Destination tile in Mat location (quantized output, f16)
        fp: Scaling tile in FB location (quantization params, f32)

    The tmov with fp parameter performs fixpipe quantization using mte_l0c_l1
    with pre_quant keyword argument.
    """
    # Get dimensions from destination tile
    m, n = dst.valid_shape
    # Strides: src is in Acc (fractal layout), dst is in Mat (row-major)
    src_stride = (m + 15) // 16 * 16  # Align to 16 blocks for fractal
    dst_stride = n  # Row-major stride

    # Use mte_l0c_l1 with pre_quant for fixpipe quantization (f32 -> f16)
    pto.mte_l0c_l1(
        src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
        pre_quant=(fp.as_ptr(), "qf322f16_pre_vec"),
    )
    return


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_make_fp_constraint(pto.bf16)],
    dtypes=[
        (pto.f32, pto.bf16, pto.f32),  # (src, dst, fp) - IR operand order
    ],
)
def template_tmov_fp_f32_bf16(src: pto.Tile, dst: pto.Tile, fp: pto.Tile):
    """Move and quantize data from Acc to Mat with scaling parameters (f32 -> bf16).

    Args:
        src: Source tile in Acc location (accumulator, f32)
        dst: Destination tile in Mat location (quantized output, bf16)
        fp: Scaling tile in FB location (quantization params, f32)

    The tmov with fp parameter performs fixpipe quantization using mte_l0c_l1
    with pre_quant keyword argument.
    """
    # Get dimensions from destination tile
    m, n = dst.valid_shape
    # Strides: src is in Acc (fractal layout), dst is in Mat (row-major)
    src_stride = (m + 15) // 16 * 16  # Align to 16 blocks for fractal
    dst_stride = n  # Row-major stride

    # Use mte_l0c_l1 with pre_quant for fixpipe quantization (f32 -> bf16)
    pto.mte_l0c_l1(
        src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
        pre_quant=(fp.as_ptr(), "qf322bf16_pre_vec"),
    )
    return
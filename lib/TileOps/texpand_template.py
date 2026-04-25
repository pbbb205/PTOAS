# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.texpands

This template implements scalar broadcast expansion for location=VEC tiles.
It fills dst.valid_shape region with the broadcasted scalar value.

Location constraint:
  - This template is designed for tiles with location=VEC (vector buffer)
  - In PTO-ISA, texpands has separate implementations for VEC and MAT locations
  - For MAT location tiles, a different template/implementation path should be used
  - Current tilelang_dsl MemorySpace only distinguishes GM and UB, where UB maps to
    both VEC and MAT locations. The constraint checks memory_space=="ub" as a proxy
  - Future enhancement: tilelang_dsl should support explicit location distinction
    (e.g., MemorySpace.VEC vs MemorySpace.MAT) for more precise constraint matching

Layout considerations:
  - PTO-ISA has both rowmajor and colmajor expands implementations
  - However, expands (scalar broadcast) is layout-agnostic: it simply fills
    the tile with a scalar value using vector stores
  - The vector store (vsts) writes data according to the tile's physical layout,
    which is handled by the underlying DMA engine
  - Therefore, this single template covers both rowmajor and colmajor cases
"""

import tilelang_dsl as pto


def _texpands_vec_location_constraint(scalar, dst) -> bool:
    """Constraint: dst tile must have location=VEC (represented as memory_space=ub).

    PTO-ISA defines texpands for both MAT and VEC locations:
      - MAT location: expands matrix tiles (different implementation path, not supported here)
      - VEC location: expands vector tiles (this template)

    Current tilelang_dsl limitation:
      MemorySpace only has UB and GM. VEC and MAT both map to UB.
      We check memory_space=="ub" as a proxy for VEC location.
      MAT tiles should use a different op/template path and won't match here.
    """
    # Check memory_space is "ub" (VEC/MAT location, not GM)
    # In current tilelang_dsl, VEC location tiles have memory_space="ub"
    ms = dst.memory_space
    if isinstance(ms, str):
        return ms == "ub"
    return hasattr(ms, "value") and ms.value == "ub"


@pto.vkernel(
    target="a5",
    op="pto.texpands",
    constraints=[_texpands_vec_location_constraint],
)
def template_texpands(scalar: pto.AnyType, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            # Use vdup for scalar broadcast
            vec = pto.vdup(scalar, mask)
            pto.vsts(vec, dst[row, col:], mask)

    return
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trecip"""

import tilelang_dsl as pto

# TODO: Add implementation for HIGH_PRECISION type
@pto.vkernel(
    target="a5",
    op="pto.trecip",
    dtypes=[(pto.f16, pto.f16), (pto.f32, pto.f32)]
)
def template_trecip(src: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vinput = pto.vlds(src[row, col:])
            if pto.constexpr(dtype == pto.f16):
                one_scalar = pto.f16(1.0)
            else:
                one_scalar = pto.f32(1.0)
            one = pto.vbr(one_scalar)
            # one = pto.vbr(dtype(1.0))
            result = pto.vdiv(one, vinput, mask)
            pto.vsts(result, dst[row, col:], mask)
    return
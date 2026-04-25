# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowprod"""

import sys
from pathlib import Path
import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.trowprod",
    advanced=True,
)
def template_trowprod(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)
    valid_rows, valid_cols = src.valid_shape
    elem_bytes = pto.bytewidth(dtype)

    # nLoop from C++ constants: TROW_PROD_LOOP_B16=7, TROW_PROD_LOOP_B32=6
    TROW_PROD_LOOP_B16 = 7
    TROW_PROD_LOOP_B32 = 6
    if pto.constexpr(dtype == pto.f16 or dtype == pto.i16):
        n_loop = TROW_PROD_LOOP_B16
    else:
        n_loop = TROW_PROD_LOOP_B32

    # Select one-point store dist based on dtype size
    if pto.constexpr(elem_bytes == 4):
        store_dist = pto.VStoreDist.ONE_POINT_B32
    elif pto.constexpr(elem_bytes == 2):
        store_dist = pto.VStoreDist.ONE_POINT_B16
    else:
        store_dist = pto.VStoreDist.ONE_POINT_B8

    mask_1, _ = pto.make_mask(dtype, 1)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
 
        one_val = dtype(1)
        v_acc = pto.vbr(one_val)
        v_one = pto.vbr(one_val)
        
        # Multiply across column chunks
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            v_src = pto.vlds(src[row, col:])
            
            # Element-wise product
            v_prod = pto.vmul(v_acc, v_src, mask)
            
            # Simulate MODE_MERGING with vsel (keep v_acc outside mask)
            v_acc = pto.vsel(v_prod, v_acc, mask)

        # Log2 reduction phase across the vector
        reduce_mask, _ = pto.make_mask(dtype, lanes) # all lanes active for inner reduction
        
        for k in range(0, n_loop, 1):
            v_intlv1, v_intlv2 = pto.vintlv(v_acc, v_one)
            v_acc = pto.vmul(v_intlv1, v_intlv2, reduce_mask)

        # Write final result at lane 0 using one-point mode
        pto.vsts(v_acc, dst[row, 0:], mask_1, dist=store_dist)
    return

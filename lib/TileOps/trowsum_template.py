# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowsum"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.trowsum",
)
def template_trowsum(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_dtype = src.element_type
    dst_dtype = dst.element_type

    # vcadd widens i16 -> i32; floats/i32 unchanged
    if pto.constexpr(src_dtype == pto.i16):
        acc_dtype = pto.i32
    else:
        acc_dtype = src_dtype

    lanes = pto.get_lanes(src_dtype)
    valid_rows, valid_cols = src.valid_shape

    # Use type-appropriate zero for accumulator initialization
    zero_val = acc_dtype(0)

    # Select one-point store dist based on dst dtype size
    elem_bytes = pto.bytewidth(dst_dtype)
    if pto.constexpr(elem_bytes == 4):
        store_dist = pto.VStoreDist.ONE_POINT_B32
    elif pto.constexpr(elem_bytes == 2):
        store_dist = pto.VStoreDist.ONE_POINT_B16
    else:
        store_dist = pto.VStoreDist.ONE_POINT_B8

    dst_mask_1, _ = pto.make_mask(dst_dtype, 1)

    for row in range(0, valid_rows, 1):
        remained = valid_cols

        acc_mask_1, _ = pto.make_mask(acc_dtype, 1)

        # Initialize the accumulator with type-appropriate zero
        v_acc = pto.vbr(zero_val)

        # Process column chunks
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(src_dtype, remained)
            v_src = pto.vlds(src[row, col:])

            # vcadd widens src_dtype to acc_dtype for integer types
            v_reduced = pto.vcadd(v_src, mask)

            # accumulate using the accumulator's mask logic
            v_acc = pto.vadd(v_acc, v_reduced, acc_mask_1)

        # Store the accumulated result safely once per row using one-point mode
        if pto.constexpr(acc_dtype != dst_dtype):
            # Truncate / Type cast before storing
            # Note: For int32 -> int16 mapping, vcvt processes it.
            acc_mask_for_cvt, _ = pto.make_mask(acc_dtype, 1)
            v_acc_casted = pto.vcvt(v_acc, dst_dtype, acc_mask_for_cvt, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.EVEN)
            pto.vsts(v_acc_casted, dst[row, 0:], dst_mask_1, dist=store_dist)
        else:
            pto.vsts(v_acc, dst[row, 0:], dst_mask_1, dist=store_dist)
    return

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Online softmax update kernel – DSL-style builder.

Generates the same IR as
  test/tilelang_st/npu/a5/src/st/testcase/softmax/softmax.pto
using the ``@pto.jit(mode="explicit")`` decorator and the ``pto.*`` namespace.

The Python maps almost line-for-line to the target MLIR:

  func.func @online_softmax_update_kernel_2d(          # function signature
      %arg0: !pto.ptr<f32, gm>, …, %arg7: i32, …)     # arg0: pto.ptr(…), …

  scf.if %has_rows {                                   # with pto.if_(has_rows):
    pto.tload ins(…) outs(…)                           #   pto.tile.load(part, tile)
    pto.vecscope {                                      #   with pto.vecscope():
      scf.for %row = … {                               #     with pto.for_(…) as row:
        %final_max, %final_sum =                       #
          scf.for %chunk = … iter_args(…) {            #       with pto.for_(…, iter_args=…) as loop:
            scf.if %has_chunk → (vreg, vreg) {         #         with pto.if_(…, results=…) as br:
              scf.yield %merged_max, %merged_sum        #           pto.yield_(…)
            } else {                                   #         with br.else_:
              scf.yield %running_max, %running_sum      #           pto.yield_(…)
            }                                          #
            scf.yield %next_max, %next_sum             #         pto.yield_(…)
          }                                            #
      }                                                #
    }                                                  #
  }                                                    #
  pto.barrier <PIPE_ALL>                               # pto.pipe_barrier(pto.Pipe.ALL)
"""

from ptodsl import pto, scalar

s = scalar  # arith shorthand alias


@pto.jit(
    name="online_softmax_update_kernel_2d",
    kernel_kind="vector",
    target="a5",
    func_attr="pto.aicore",
    mode="explicit",
)
def online_softmax_update_kernel_2d(
    arg0: pto.ptr(pto.float32, "gm"),
    arg1: pto.ptr(pto.float32, "gm"),
    arg2: pto.ptr(pto.float32, "gm"),
    arg3: pto.ptr(pto.float32, "gm"),
    arg4: pto.ptr(pto.float32, "gm"),
    arg5: pto.ptr(pto.float32, "gm"),
    arg6: pto.ptr(pto.float32, "gm"),
    arg7: pto.int32,
    arg8: pto.int32,
):
    # ── Index constants ──────────────────────────────────────────────────────
    c0   = pto.const(0)
    c1   = pto.const(1)
    c8   = pto.const(8)
    c64  = pto.const(64)
    c128 = pto.const(128)

    # ── i64 address constants (UB tile base addresses) ───────────────────────
    c0_i64    = pto.const(0,     dtype=pto.int64)
    c1_i64    = pto.const(1,     dtype=pto.int64)   # noqa: F841 (present in IR)
    c8_i64    = pto.const(8,     dtype=pto.int64)   # noqa: F841
    c16_i64   = pto.const(16,    dtype=pto.int64)   # noqa: F841
    c32_i64   = pto.const(32,    dtype=pto.int64)   # noqa: F841
    c64_i64   = pto.const(64,    dtype=pto.int64)   # noqa: F841
    c128_i64  = pto.const(128,   dtype=pto.int64)
    c256_i64  = pto.const(256,   dtype=pto.int64)
    c512_i64  = pto.const(512,   dtype=pto.int64)   # noqa: F841
    c8448_i64  = pto.const(8448,  dtype=pto.int64)
    c16640_i64 = pto.const(16640, dtype=pto.int64)
    c16768_i64 = pto.const(16768, dtype=pto.int64)
    c16896_i64 = pto.const(16896, dtype=pto.int64)

    # ── i32 constants ────────────────────────────────────────────────────────
    c1_i32  = pto.const(1,  dtype=pto.int32)
    c8_i32  = pto.const(8,  dtype=pto.int32)
    c64_i32 = pto.const(64, dtype=pto.int32)
    c0_i32  = pto.const(0,  dtype=pto.int32)

    # ── Block-level row assignment ────────────────────────────────────────────
    block_i64     = pto.get_block_idx()
    block_idx     = s.index_cast(block_i64)                         # → index
    row_base      = s.muli(block_idx, c8)
    _             = s.index_cast(pto.int32, c8)                     # block_rows_i32
    row_base_i32  = s.index_cast(pto.int32, row_base)
    remaining_rows= s.subi(arg8, row_base_i32)
    has_rows      = remaining_rows > c0_i32
    too_many_rows = remaining_rows > c8_i32
    row_count_i32 = s.select(too_many_rows, c8_i32, remaining_rows)
    row_count     = s.index_cast(row_count_i32)                     # → index
    seq           = s.index_cast(arg7)                              # → index
    rows          = s.index_cast(arg8)                              # → index
    rows_x_128    = s.muli(rows, c128)

    with pto.if_(has_rows):
        # ── Tensor views ─────────────────────────────────────────────────────
        s1   = [rows, rows, rows, c1, rows]
        s128 = [rows_x_128, rows_x_128, rows_x_128, c128, c1]
        sh1  = [c1, c1, c1, rows, c1]
        sh128= [c1, c1, c1, rows, c128]

        oldmax_view = pto.make_tensor_view(arg0, shape=sh1,   strides=s1)
        oldsum_view = pto.make_tensor_view(arg1, shape=sh1,   strides=s1)
        qk_view     = pto.make_tensor_view(arg2, shape=sh128, strides=s128)
        newmax_view = pto.make_tensor_view(arg3, shape=sh1,   strides=s1)
        newsum_view = pto.make_tensor_view(arg4, shape=sh1,   strides=s1)
        expmax_view = pto.make_tensor_view(arg5, shape=sh1,   strides=s1)
        out_view    = pto.make_tensor_view(arg6, shape=sh128, strides=s128)

        # ── Partition views ───────────────────────────────────────────────────
        off = [c0, c0, c0, row_base, c0]
        z1  = [c1, c1, c1, row_count, c1]
        zs  = [c1, c1, c1, row_count, seq]

        oldmax_part = pto.partition_view(oldmax_view, offsets=off, sizes=z1)
        oldsum_part = pto.partition_view(oldsum_view, offsets=off, sizes=z1)
        qk_part     = pto.partition_view(qk_view,     offsets=off, sizes=zs)
        newmax_part = pto.partition_view(newmax_view, offsets=off, sizes=z1)
        newsum_part = pto.partition_view(newsum_view, offsets=off, sizes=z1)
        expmax_part = pto.partition_view(expmax_view, offsets=off, sizes=z1)
        out_part    = pto.partition_view(out_view,    offsets=off, sizes=zs)

        # ── UB tile allocation ────────────────────────────────────────────────
        tile_col = pto.tile_buf_type([8,  1], pto.float32, [-1,  1], blayout="ColMajor")
        tile_w   = pto.tile_buf_type([8, 128], pto.float32, [-1, -1])

        oldmax_tile = pto.alloc_tile(tile_col, addr=c0_i64,     valid_row=row_count)
        oldsum_tile = pto.alloc_tile(tile_col, addr=c128_i64,   valid_row=row_count)
        qk_tile     = pto.alloc_tile(tile_w,   addr=c256_i64,   valid_row=row_count, valid_col=seq)
        out_tile    = pto.alloc_tile(tile_w,   addr=c8448_i64,  valid_row=row_count, valid_col=seq)
        newmax_tile = pto.alloc_tile(tile_col, addr=c16640_i64, valid_row=row_count)
        newsum_tile = pto.alloc_tile(tile_col, addr=c16768_i64, valid_row=row_count)
        expmax_tile = pto.alloc_tile(tile_col, addr=c16896_i64, valid_row=row_count)

        # ── Tile loads from GM ────────────────────────────────────────────────
        pto.tile.load(oldmax_part, oldmax_tile)
        pto.tile.load(oldsum_part, oldsum_tile)
        pto.tile.load(qk_part,     qk_tile)

        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.vecscope():
            # Materialise typed UB pointers from tile handles
            ptr_ub = pto.ptr(pto.float32, "ub")
            vf32   = pto.vreg_type(64, pto.float32)

            ub_om  = pto.as_ptr(oldmax_tile, ptr_ub)
            ub_os  = pto.as_ptr(oldsum_tile, ptr_ub)
            ub_qk  = pto.as_ptr(qk_tile,     ptr_ub)
            ub_out = pto.as_ptr(out_tile,     ptr_ub)
            ub_nm  = pto.as_ptr(newmax_tile,  ptr_ub)
            ub_ns  = pto.as_ptr(newsum_tile,  ptr_ub)
            ub_em  = pto.as_ptr(expmax_tile,  ptr_ub)

            active      = pto.pset_b32("PAT_ALL")
            one_mask, _ = pto.plt_b32(c1_i32)

            with pto.for_(c0, row_count, step=c1) as row:
                row_qk    = s.muli(row, c128)
                oldmax_bc = pto.vbrc_load(ub_om, row, vf32)
                oldsum_bc = pto.vbrc_load(ub_os, row, vf32)

                # scf.for with iter_args: accumulate (running_max, running_sum)
                with pto.for_(c0, c128, step=c64, iter_args=(oldmax_bc, oldsum_bc)) as loop:
                    chunk                    = loop.iv
                    running_max, running_sum = loop.iter_args

                    chunk_i32      = s.index_cast(pto.int32, chunk)
                    remaining_cols = s.subi(arg7, chunk_i32)
                    has_chunk      = remaining_cols > c0_i32

                    # scf.if with results – produce (next_max, next_sum)
                    with pto.if_(has_chunk, results=(vf32, vf32)) as br:
                        with br.then_:
                            chunk_mask, _      = pto.plt_b32(remaining_cols)
                            chunk_base         = s.addi(row_qk, chunk)
                            vec                = pto.vlds(ub_qk, chunk_base, vf32)
                            chunk_max          = pto.vcmax(vec, chunk_mask)
                            chunk_max_bc       = pto.vdup(chunk_max, active, position="LOWEST")
                            merged_max         = pto.vmax(running_max, chunk_max_bc, active)
                            scaled_running     = pto.vexpdif(running_max, merged_max, active)
                            running_sum_scaled = pto.vmul(scaled_running, running_sum, active)
                            chunk_exp          = pto.vexpdif(vec, merged_max, chunk_mask)
                            chunk_sum          = pto.vcadd(chunk_exp, chunk_mask)
                            chunk_sum_bc       = pto.vdup(chunk_sum, active, position="LOWEST")
                            merged_sum         = pto.vadd(running_sum_scaled, chunk_sum_bc, active)
                            pto.yield_(merged_max, merged_sum)
                        with br.else_:
                            pto.yield_(running_max, running_sum)

                    next_max, next_sum = br.results
                    pto.yield_(next_max, next_sum)

                final_max, final_sum = loop.results

                # Compute per-row expmax scalar
                raw_em  = pto.vexpdif(oldmax_bc, final_max, active)
                sc_os   = pto.vmul(raw_em, oldsum_bc, active)
                expmax  = pto.vdiv(sc_os, final_sum, active)

                pto.vsts_1pt(final_max, ub_nm, row, one_mask)
                pto.vsts_1pt(final_sum, ub_ns, row, one_mask)
                pto.vsts_1pt(expmax,    ub_em, row, one_mask)

                # Output normalisation loop
                with pto.for_(c0, c128, step=c64) as chunk2:
                    rem2      = s.subi(arg7, s.index_cast(pto.int32, chunk2))
                    has_chunk2= rem2 > c0_i32
                    with pto.if_(has_chunk2):
                        cmask2, _ = pto.plt_b32(rem2)
                        cbase2    = s.addi(row_qk, chunk2)
                        vec2      = pto.vlds(ub_qk, cbase2, vf32)
                        exp2      = pto.vexpdif(vec2, final_max, cmask2)
                        out2      = pto.vdiv(exp2, final_sum, cmask2)
                        pto.vsts(out2, ub_out, cbase2, cmask2)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)

        # Tile stores to GM
        pto.tile.store(newmax_tile, newmax_part)
        pto.tile.store(newsum_tile, newsum_part)
        pto.tile.store(expmax_tile, expmax_part)
        pto.tile.store(out_tile,    out_part)

    pto.pipe_barrier(pto.Pipe.ALL)


def build():
    return online_softmax_update_kernel_2d.mlir_module()


if __name__ == "__main__":
    print(online_softmax_update_kernel_2d)

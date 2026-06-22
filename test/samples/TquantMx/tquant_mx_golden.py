#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CI/remote-validation golden generator for the TquantMx (MXFP8, OCP) sample.

Uses validation_runtime helpers so the buffer names (v1, v2, ...) and output
names match what generate_testcase.py produces.  This is the CI/remote path;
the standalone board-validation path under npu_validation/ has its own
golden.py with self-contained file names (input.bin, golden.bin, ...).

Replicates the OCP MXFP8 quantization contract:
  src f32 [M, K] (group_size=32 along K)
    -> per-group absmax -> e8m0 shared exponent (emax=8 for e4m3)
    -> fp32 reciprocal scaling = 2^(254 - e8m0)
    -> fp8 e4m3fn output = clamp(src * scaling, -448, 448)
"""

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, write_buffers, write_golden


M = 16
K = 32
GROUP_SIZE = 32
EMAX = 8  # e4m3 max exponent


def fp32_to_fp8_element(data_abs_max, emax=EMAX):
    bits = np.uint32(np.frombuffer(np.float32(data_abs_max).tobytes(), dtype=np.uint32)[0])
    exponent_b32 = int((bits & np.uint32(0x7F800000)) >> np.uint32(23))
    mantissa_b32 = int(bits & np.uint32(0x007FFFFF))
    if exponent_b32 == 0xFF and mantissa_b32 != 0:
        return 0xFF, np.uint32(0x7FC00000).view(np.float32)
    if exponent_b32 <= emax:
        return 0x00, np.uint32(0x7F000000).view(np.float32)
    e8m0 = exponent_b32 - emax
    scale_exp = 254 - e8m0
    scaling = np.uint32(scale_exp << 23).view(np.float32)
    if scaling == 0.0:
        scaling = np.float32(2.0 ** -127)
    return e8m0, scaling


def fp32_to_fp8e4m3fn_bytes(arr):
    """Convert fp32 array to fp8 e4m3fn packed as int8 bytes (round-to-nearest-even)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = arr.view(np.uint32)
    sign = ((u32 >> np.uint32(31)) & np.uint32(1)).astype(np.int32)
    exp_f32 = ((u32 >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mant_f32 = (u32 & np.uint32(0x7FFFFF)).astype(np.int32)

    result = np.zeros(arr.shape, dtype=np.int32)
    is_normal = (exp_f32 >= 1) & (exp_f32 <= 254)
    is_subnormal_f32 = (exp_f32 == 0) & (mant_f32 != 0)
    fp8_exp = exp_f32 - 120
    fp8_exp = np.where(is_subnormal_f32, -999, fp8_exp)

    overflow = is_normal & (fp8_exp > 15)
    result = np.where(overflow, (sign << np.int32(7)) | np.int32(0x7E), result)
    inrange = is_normal & (fp8_exp >= 0) & (fp8_exp <= 15)

    m_high = mant_f32 >> np.int32(20)
    round_bit = (mant_f32 >> np.int32(19)) & np.int32(1)
    sticky = np.where((mant_f32 & np.int32(0x7FFFF)) != 0, np.int32(1), np.int32(0))
    round_up = (round_bit & (sticky | (m_high & np.int32(1)))).astype(bool)
    m_rounded = m_high + np.where(round_up, np.int32(1), np.int32(0))
    carry = (m_rounded >> np.int32(3)) & np.int32(1)
    m_rounded = m_rounded & np.int32(0x7)
    fp8_exp_r = fp8_exp + carry
    over_after_carry = inrange & (fp8_exp_r > 15)
    result = np.where(over_after_carry, (sign << np.int32(7)) | np.int32(0x7E), result)
    inrange = inrange & (fp8_exp_r <= 14)
    assembled = (sign << np.int32(7)) | (fp8_exp_r << np.int32(3)) | m_rounded
    result = np.where(inrange, assembled, result)

    is_nan = (exp_f32 == 0xFF) & (mant_f32 != 0)
    result = np.where(is_nan, np.int32(0x7F), result)
    is_inf = (exp_f32 == 0xFF) & (mant_f32 == 0)
    result = np.where(is_inf, (sign << np.int32(7)) | np.int32(0x7E), result)
    return result.astype(np.uint8).view(np.int8)


def main():
    meta = load_case_meta()
    generator = rng()

    # Identify buffers by meta: the generated main.cpp reads inputs and writes
    # outputs using v1, v2, ... names.  For tquant.mx the kernel signature is
    # (src, dst, exp, max, scaling) where src is input and the rest are outputs.
    # generate_testcase.py assigns v1=src (input), v2..v5=dst/exp/max/scaling.
    input_names = meta.inputs
    output_names = meta.outputs

    src_name = input_names[0] if input_names else "v1"
    # Outputs are ordered by tstore appearance: dst, exp, max, scaling.
    dst_name = output_names[0] if len(output_names) > 0 else "v2"
    exp_name = output_names[1] if len(output_names) > 1 else "v3"
    max_name = output_names[2] if len(output_names) > 2 else "v4"
    scaling_name = output_names[3] if len(output_names) > 3 else "v5"

    # Generate source: f32 [M, K] with values in [-10, 10].
    src = generator.uniform(-10, 10, size=M * K).astype(np.float32).reshape(M, K)

    buffers = default_buffers(meta)
    buffers[src_name] = src.reshape(-1)
    # Zero-init outputs (they will be overwritten by the kernel).
    for name in output_names:
        buffers[name] = np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])
    write_buffers(meta, buffers)

    # Compute golden: per-group absmax -> e8m0 -> scaling -> fp8.
    src_abs = np.abs(src)
    group_max = np.max(src_abs.reshape(-1, GROUP_SIZE), axis=1)  # [groups]

    e8m0s = []
    scalings = []
    for v in group_max.reshape(-1).tolist():
        e, s = fp32_to_fp8_element(v, emax=EMAX)
        e8m0s.append(e)
        scalings.append(s)
    e8m0 = np.array(e8m0s).astype(np.uint8)
    scaling_per_group = np.array(scalings).astype(np.float32)

    # Golden outputs (flat, matching meta elem_counts).
    golden_outputs = {}

    # dst: fp8 e4m3fn packed as int8, [M*K] elements.
    scaling_broadcast = np.repeat(scaling_per_group, GROUP_SIZE).reshape(M, K).astype(np.float32)
    scaled = src.astype(np.float64) * scaling_broadcast.astype(np.float64)
    scaled = np.clip(scaled, -448.0, 448.0).astype(np.float32)
    dst_bytes = fp32_to_fp8e4m3fn_bytes(scaled)
    golden_outputs[dst_name] = dst_bytes.reshape(-1)

    # exp: ui8 [groups] = e8m0 per group.
    golden_outputs[exp_name] = e8m0.reshape(-1)

    # max: f32 [groups] = per-group absmax.
    golden_outputs[max_name] = group_max.reshape(-1).astype(np.float32)

    # scaling: f32 [groups] = per-group reciprocal scale.
    golden_outputs[scaling_name] = scaling_per_group.reshape(-1)

    write_golden(meta, golden_outputs)


if __name__ == "__main__":
    main()

"""TileLang DSL template for pto.tcvt."""

import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i32),
    ],
)
def template_tcvt_f32_to_i32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.f32),
    ],
)
def template_tcvt_i32_to_f32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return

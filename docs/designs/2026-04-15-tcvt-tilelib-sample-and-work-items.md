# `pto.tcvt` TileLib 模板库设计与工作项

## 1. 目标

参考 `pto-isa` 在 A5 上现有的 `TCVT_IMPL` 实现，用 TileLang DSL 在 TileLib 中补齐 `pto.tcvt` 模板库。

当前 PTOAS 的 `pto.tcvt` 只显式携带 `rmode`，没有单独暴露 `sat_mode`。因此模板库不能只做一层简单透传，而是要在内部按 `(src_dtype, dst_dtype)` 复现 A5 的默认 `sat_mode` 选择，再为不同类型对走到正确的 VPTO 路径。


## 2. 当前语义

### 2.1 PTOAS 侧

当前 PTOAS 的 `pto.tcvt` 只有 `rmode` attribute，没有 `sat_mode` attribute。

这意味着从 PTOAS 传到 TileLib 的静态信息只有 round mode。默认饱和策略需要模板库自己补齐。

### 2.2 A5 `pto-isa` 侧

A5 侧已有多组 `TCVT_IMPL` 重载，包括：

- `TCVT_IMPL(dst, src, mode)`
- `TCVT_IMPL(dst, src, mode, satMode)`
- `TCVT_IMPL(dst, src, tmp, mode)`
- `TCVT_IMPL(dst, src, tmp, mode, satMode)`

其中 `TCVT_IMPL(dst, src, tmp, mode)` 在 A5 上只是转调无 `tmp` 的版本，`tmp` 本身不参与实现。这里保留 `tmp`，主要是为了和 A2/A3 的接口形态保持兼容。

如果只聚焦当前 `pto.tcvt` 真正需要对齐的那条入口，也就是：

```cpp
TCVT_IMPL(dst, src, mode)
```

那么 A5 `pto-isa` 里的主要过程可以概括成下面这条链路：

1. 先按 `(src_dtype, dst_dtype)` 选默认 `satMode`
   也就是这条入口本身先做一层类型分派，把当前 type pair 映射成默认
   `satMode=ON` 或 `OFF`。

2. 再转调到显式 `satMode` 的主实现入口

```cpp
TCVT_IMPL(dst, src, mode, satMode)
```

3. 在显式 `satMode` 入口里，先根据 `(src_dtype, dst_dtype, satMode)` 计算当前需要设置哪些 CTRL 位
   这里会调用 `determineSaturationCtrlBits(...)`，然后再调用
   `applySaturationCtrlBits(...)` 把这些 CTRL 位写进去。

4. CTRL 位设置完成后，再按 `round_mode` 做一层 switch 分派
   例如分到 `RoundRType` / `RoundAType` / `RoundFType` / `RoundCType` /
   `RoundZType` / `RoundOType`，最后统一调用：

```cpp
implTCVT<TileDataD, TileDataS, RoundXType>(...)
```

5. `implTCVT(...)` 内部再按 type pair 落到具体 helper
   例如：
   - `cast32to32`
   - `cast32to16`
   - `cast16to32`
   - `cast16to16`
   - `cast16to8`
   - 以及 `NonSatTorch` 那几条专门 helper

6. 最后恢复之前改过的 CTRL 位
   也就是在主实现入口的尾部调用 `restoreSaturationCtrlBits(...)`。

把这段代码实现压成一条线来看，就是：

```text
TCVT_IMPL(dst, src, mode)
  -> 按类型对选默认 satMode
  -> TCVT_IMPL(dst, src, mode, satMode)
       -> determineSaturationCtrlBits(...)
       -> applySaturationCtrlBits(...)
       -> switch(round_mode)
            -> implTCVT<RoundXType>(...)
                 -> cast helper / NonSatTorch helper
       -> restoreSaturationCtrlBits(...)
```

对 TileLib 来说，真正需要复现的就是这条框架，而不是只把 `rmode` 直接透传给某一个
`vcvt` 就结束。

因此，对当前 A5 来说，`pto.tcvt` 需要对齐的真实语义是：

1. 外部只显式给 `rmode`
2. 库内部按类型对选择默认 `sat_mode`
3. 再按类型对和 `sat_mode` 进入具体实现路径

## 3. A5 实现要点

### 3.1 默认 `sat_mode`

A5 的 round-only `TCVT_IMPL(dst, src, mode)` 对下面这些类型对默认使用 `sat_mode=OFF`：

| 源类型 | 目标类型 | 默认 `sat_mode` | 说明 |
|---|---|---|---|
| `f16` | `u8` | `OFF` | A5 现有默认行为 |
| `f16` | `i8` | `OFF` | A5 现有默认行为 |
| `f32` | `i16` | `OFF` | A5 现有默认行为 |
| `f16` | `i16` | `OFF` | A5 现有默认行为 |
| `i64` | `i32` | `OFF` | A5 现有默认行为 |
| `i32` | `i16` | `OFF` | A5 现有默认行为 |

除上表外，其余类型对默认使用 `sat_mode=ON`。

这部分规则应直接在 TileLib 模板内部复现，不应依赖 PTOAS 额外传参。

### 3.2 A5 `TCVT` 整体支持表

按三个实现维度分类：

- 是否受 `round_mode` 影响
- 是否受 `sat_mode` 影响
- 是否需要 `NonSatTorch` 对齐

这里根据 `pto-isa/include/pto/npu/a5/TCvt.hpp` 整理，不等于当前
PTOAS + TileLib 已经全部打通。

下面各表最后一列 `TileLib是否支持` 以当前
`PTOAS/lib/TileOps/tcvt_template.py` 实际实现为准。当前已打通的先标 `已支持`，
其余暂时留空。

#### 3.2.1 不受 `round_mode` / `sat_mode` 影响，也不需要 `NonSatTorch`

这组最适合优先实现，基本都是 expand / unpack 路径。

| 源类型 | 目标类型 | A5 helper 覆盖 | 备注 | TileLib是否支持 |
|---|---|---|---|---|
| `f16` | `f32` | 1D+2D，`vcvt + part` | type expand | |
| `bf16` | `f32` | 1D+2D，`vcvt + part` | type expand | |
| `i16` | `f32` / `i32` / `u32` | 1D+2D，expand helper | widening path | |
| `i32` | `i64` | 1D+2D，expand helper | | |
| `u8` | `f16` / `u16` | 1D only，expand helper | 当前只看到 1D helper | |
| `i8` | `f16` / `i16` / `i32` | 1D only，expand helper | 当前只看到 1D helper | |
| `fp8_e4m3` / `fp8_e5m2` / `h8` | `f32` | 1D+2D，expand helper | source 8-bit float | |
| `fp4_e1m2x2` / `fp4_e2m1x2` | `bf16` | 1D+2D，专用 unpack helper | 4-bit packed source | |

#### 3.2.2 受 `round_mode` 影响，不受 `sat_mode` 影响，也不需要 `NonSatTorch`

这组属于 round-only 路径。

| 源类型 | 目标类型 | A5 helper 覆盖 | 备注 | TileLib是否支持 |
|---|---|---|---|---|
| `f32` | `f32` | 1D+2D，`vtrc` | 保持 `f32`，做 integer-valued float rounding | |
| `f16` | `i32` | 1D+2D，`vcvt + part` | | |
| `i16` | `f16` | 1D+2D，`vcvt` | | |
| `i32` | `f32` | 1D+2D，`vcvt` | | `已支持` |
| `i64` | `f32` | 1D+2D，`vcvt + part` | | |
| `bf16` | `fp4_e1m2x2` / `fp4_e2m1x2` | 1D+2D，专用 packed helper | 不是普通 `vcvt` 套餐，但不吃 `sat_mode` | |

#### 3.2.3 不受 `round_mode` 影响，受 `sat_mode` 影响，不需要 `NonSatTorch`

这组主要是整数窄化。

| 源类型 | 目标类型 | A5 helper 覆盖 | 默认 `effective_sat_mode` | 备注 | TileLib是否支持 |
|---|---|---|---|---|---|
| `i16` | `u8` | 1D+2D，`vcvt + part` | `ON` | | |
| `i32` | `i16` | 1D+2D，`vcvt + part` | `OFF` | | |
| `i32` | `u16` / `u8` | 1D+2D，`vcvt + part` | `ON` | | |
| `u32` | `i16` / `u16` / `u8` | 1D+2D，`vcvt + part` | `ON` | | |
| `i64` | `i32` | 1D+2D，`vcvt + part` | `OFF` | | |

#### 3.2.4 同时受 `round_mode` 和 `sat_mode` 影响，但不需要 `NonSatTorch`

这组是常规 `tcvt` 主干路径。当前先打通的 `f32 -> i32` 就属于这一类。

| 源类型 | 目标类型 | A5 helper 覆盖 | 默认 `effective_sat_mode` | 备注 | TileLib是否支持 |
|---|---|---|---|---|---|
| `f32` | `f16` / `bf16` | 1D+2D，`vcvt + part` | `ON` | 窄化 float | |
| `f32` | `i32` | 1D+2D，`vcvt` | `ON` | 当前已先打通这一类普通路径 | `已支持` |
| `f32` | `i64` | 1D+2D，`vcvt + part` | `ON` | | |
| `f32` | `fp8_e4m3` / `fp8_e5m2` | 1D+2D，`vcvt + part` | `ON` | | |
| `f16` | `u8` | 1D+2D，`vcvt + part` | `OFF` | | |
| `bf16` | `i32` | 1D+2D，`vcvt + part` | `ON` | | |
| `bf16` | `f16` | 1D+2D，`vcvt` | `ON` | helper 内部是 `SAT_ROUND` 顺序 | |

#### 3.2.5 同时受 `round_mode` 和 `sat_mode` 影响，且需要 `NonSatTorch`

这组后面要单独收口。不能把它们直接等价成普通 `vcvt(..., sat=NOSAT)`。

| 源类型 | 目标类型 | A5 helper 覆盖 | 默认 `effective_sat_mode` | `NonSatTorch` | 备注 | TileLib是否支持 |
|---|---|---|---|---|---|---|
| `f32` | `i16` | 1D+2D，`vcvt + part` | `OFF` | 是 | `OFF` 时走 `NonSatTorch` | |
| `f16` | `i16` | 1D+2D，`vcvt` | `OFF` | 是 | | |
| `f16` | `i8` | 1D+2D，`vcvt + part` | `OFF` | 是 | | |

#### 3.2.6 专用 helper，`round_mode` 受限

这组不建议和普通路径一起排第一批。A5 helper 虽然形式上带模板参数，但当前实现实际固定在特定 round 行为上。

| 源类型 | 目标类型 | A5 helper 覆盖 | 默认 `effective_sat_mode` | 备注 | TileLib是否支持 |
|---|---|---|---|---|---|
| `f32` | `h8` | 1D+2D，专用 helper | `ON` | helper 实际固定 `ROUND_A` | |
| `f16` | `h8` | 1D+2D，专用 helper | `ON` | helper 实际固定 `ROUND_A` | |

这里再记三点：

- `f16 -> fp8_e4m3/e5m2` 当前 A5 `pto-isa` 明确未实现；`f16` 这边只提供了 `h8` 专用 helper。
- `h8`、`fp4` 这类路径不是普通 `vcvt` 套餐，后面做 TileLib 时不建议和常规 `f32/f16/bf16/int` 主干混在第一批一起做。
- 这里说“受 / 不受 `round_mode` 影响”指的是该 pair 的 A5 helper 是否真的消费 round 语义，不是说 PTOAS 这层拿不到 `rmode`。

### 3.3 `round_mode` 映射表

当前 `pto.tcvt` 这条链路里，round mode 至少会经过四层名字：

1. PTOAS op attr：`#pto<round_mode ...>`
2. `ExpandTileOp` 传给 TileLang 的上下文字符串：`round_mode`
3. TileLang DSL 前端：`pto.VcvtRoundMode.*`
4. VPTO / A5 lowering：`rnd = "R"` 这一类 token，或 `RoundMode::CAST_*`

建议文档和实现都按下面这张表统一，不要在不同层写不同别名。

| PTOAS `rmode` | `ExpandTileOp` 传值 | DSL 前端 | VPTO token | A5 / EmitC | 语义 |
|---|---|---|---|---|---|
| `NONE` | `RINT` | `pto.VcvtRoundMode.R` | `R` / `ROUND_R` | `RoundMode::CAST_RINT` | round to nearest, ties to even |
| `RINT` | `RINT` | `pto.VcvtRoundMode.R` | `R` / `ROUND_R` | `RoundMode::CAST_RINT` | round to nearest, ties to even |
| `CAST_RINT` | `RINT` | `pto.VcvtRoundMode.R` | `R` / `ROUND_R` | `RoundMode::CAST_RINT` | round to nearest, ties to even |
| `ROUND` | `ROUND` | `pto.VcvtRoundMode.A` | `A` / `ROUND_A` | `RoundMode::CAST_ROUND` | round away from zero |
| `FLOOR` | `FLOOR` | `pto.VcvtRoundMode.F` | `F` / `ROUND_F` | `RoundMode::CAST_FLOOR` | round toward negative infinity |
| `CEIL` | `CEIL` | `pto.VcvtRoundMode.C` | `C` / `ROUND_C` | `RoundMode::CAST_CEIL` | round toward positive infinity |
| `TRUNC` | `TRUNC` | `pto.VcvtRoundMode.Z` | `Z` / `ROUND_Z` | `RoundMode::CAST_TRUNC` | round toward zero |
| `ODD` | `ODD` | `pto.VcvtRoundMode.O` | `O` / `ROUND_O` | `RoundMode::CAST_ODD` | round to odd |

这里再补三条实现上要注意的点：

- `ExpandTileOp` 当前应把 `NONE` / `RINT` / `CAST_RINT` 统一归一成 `RINT`，这样模板内部只需要处理一套默认 round-to-nearest 语义。
- `PTO_IR_manual` 里对 `ROUND` 的描述偏旧，当前实现和 VPTO 规格应按 “away from zero” 理解。
- `f32 -> f32` 这条 `vtrc` 路径不能直接照抄上表全部 token。当前 VPTO `vtrc` 规格只明确列了 `R/A/F/C/Z`，`ODD` 需要单独看目标语义，不应默认跟 `vcvt` 完全等价。

### 3.4 不同类型对的处理路径

从模板实现角度看，更重要的不是 A5 内部怎么切 CTRL 位，而是不同类型对最终该走哪条路径。建议按下面这张表组织 TileLib 逻辑：

| 类型对 | 默认路径 | 备注 |
|---|---|---|
| `f32 -> f32` | `vtrc` | 这是 round-to-int-valued-float，不应走 `vcvt` |
| `f32 -> i16` 且 `sat_mode=OFF` | `NonSatTorch` helper | 需要对齐 A5 现有边界值行为 |
| `f16 -> i16` 且 `sat_mode=OFF` | `NonSatTorch` helper | 需要对齐 A5 现有边界值行为 |
| `f16 -> i8` 且 `sat_mode=OFF` | `NonSatTorch` helper | 需要对齐 A5 现有边界值行为 |
| 其余合法类型对 | `vcvt` | 具体带哪些 attr 取决于 VPTO contract |

`NonSatTorch` 这三条路径不能简单等价成普通 `vcvt(..., sat=NOSAT)`。A5 这里保留了专门实现，是为了在 `inf`、`nan`、`overflow` 这些边界值上对齐当前行为。

### 3.5 `vcvt` 的 attr 约束

TileLib 侧即使已经推导出了 `sat_mode`，也不能无条件给 `vcvt` 传 `rnd/sat/part`。这些 attr 是否应该出现，仍然要服从 VPTO `vcvt` 的 verifier 约束。

下面列几个模板里一定会碰到的典型路径：

| 类型对 | `rnd` | `sat` | `part` | 建议路径 |
|---|---|---|---|---|
| `f32 -> i32` | 需要 | 需要 | 不需要 | `vcvt` |
| `i32 -> f32` | 需要 | 不需要 | 不需要 | `vcvt` |
| `f32 -> f16/bf16` | 需要 | 需要 | 需要 | `vcvt` |
| `f16/bf16 -> f32` | 不需要 | 不需要 | 需要 | `vcvt` |
| `f32 -> f32` | 不适用 | 不适用 | 不适用 | `vtrc` |

因此，模板里最好把“默认 `sat_mode` 推导”和“`vcvt` attr 组织”拆成两层，不要混在一起写。

## 4. TileLib 设计建议

### 4.1 模板主流程

TileLib 中的 `pto.tcvt` 模板建议保持下面这个结构：

```python
@pto.vkernel(target="a5", op="pto.tcvt")
def template_tcvt(src: pto.Tile, dst: pto.Tile):
    src_dtype = src.element_type
    dst_dtype = dst.element_type

    round_mode = pto.get_op_attr("round_mode", "RINT")
    sat_mode = _a5_default_tcvt_sat_mode(src_dtype, dst_dtype)

    if _needs_nonsat_torch(src_dtype, dst_dtype, sat_mode):
        return _emit_nonsat_torch_tcvt(src, dst, round_mode)

    return _emit_regular_tcvt(src, dst, round_mode, sat_mode)
```

这里建议把逻辑拆成三个内部 helper：

- `_a5_default_tcvt_sat_mode(src_dtype, dst_dtype)`
- `_needs_nonsat_torch(src_dtype, dst_dtype, sat_mode)`
- `_emit_regular_tcvt(...)`

这样写更容易和 A5 `pto-isa` 的现有规则对齐，也方便后面做单测。

### 4.2 普通路径的分派原则

`_emit_regular_tcvt(...)` 里建议只做两件事：

1. 判断当前类型对应该走 `vtrc` 还是 `vcvt`
2. 如果走 `vcvt`，按 VPTO contract 决定是否附带 `rnd`、`sat`、`part`

不要直接按 A5 C++ helper 名称去分派 TileLang DSL。TileLib 需要对齐的是最终语义，而不是逐个复刻底层 helper 名。

### 4.3 `NonSatTorch` 的定位

`NonSatTorch` 在这里应视为模板内部实现细节，不是新的对外接口。

可以先完成普通路径，再补 `NonSatTorch`。如果目标是和当前 A5 行为严格对齐，这三条特殊路径需要在第一版就一起补上。

## 5. 工作项

### 5.1 TileLib 模板库

需要补一份 `pto.tcvt` TileLib 模板，实现以下逻辑：

| 工作项 | 说明 |
|---|---|
| 读取 `round_mode` | 通过 `pto.get_op_attr("round_mode", "RINT")` 获取 |
| 推导默认 `sat_mode` | 严格按 A5 类型对规则实现 |
| 支持 `vtrc` 路径 | 至少覆盖 `f32 -> f32` |
| 支持普通 `vcvt` 路径 | 并满足 VPTO verifier 对 attr 的要求 |
| 支持 `NonSatTorch` 路径 | 至少覆盖 `f32 -> i16`、`f16 -> i16`、`f16 -> i8` 且默认 `OFF` 的场景 |

### 5.2 DSL / ExpandHelper / `ExpandTileOp`

除了模板本身，还需要把下面几处配套能力接上：

| 模块 | 工作项 |
|---|---|
| TileLang DSL | 支持 `pto.get_op_attr("round_mode", ...)` |
| TileLang DSL | 为 `pto.vtrc` 补 round-mode surface，避免 `f32 -> f32` 卡住 |
| ExpandHelper | 传递 `round_mode` 到模板上下文 |
| `ExpandTileOp` | `SpecKey` 纳入 `round_mode`，避免不同 `rmode` 错误复用实例 |

当前没有必要把 `sat_mode` 加进 `SpecKey`，因为在现有语义下，它完全由 `(src_dtype, dst_dtype)` 决定，而这部分已经包含在操作数 specialization 里。

### 5.3 测试

建议测试按三类准备：

| 测试类型 | 关注点 |
|---|---|
| 模板选择与缓存 | 相同类型对、不同 `rmode` 不应复用同一实例 |
| 模板展开 | `round_mode` 能正确进入 `vtrc` / `vcvt` |
| 数值行为 | 默认 `OFF` 类型对、`NonSatTorch` 特殊路径、`f32 -> f32` 路径 |

最少应覆盖下面这些代表性 case：

- `f32 -> f32`
- `f32 -> i16`
- `f16 -> i16`
- `f16 -> i8`
- `f32 -> i32`
- `i32 -> f32`

## 6. 结论

这项工作的关键不是“把 `rmode` 传给一个 `vcvt`”这么简单，而是把当前 A5 `pto-isa` 在 round-only `TCVT_IMPL` 里隐含的默认 `sat_mode` 规则和类型分派规则一起带到 TileLib。

对当前 PTOAS `pto.tcvt` 而言，模板库应复现下面这条主线：

1. 从 PTOAS 读取 `round_mode`
2. 在模板内部按 `(src_dtype, dst_dtype)` 推导默认 `sat_mode`
3. 按类型对分派到 `vtrc`、普通 `vcvt` 或 `NonSatTorch` helper

这样实现出来的 TileLib 模板库，才能和 A5 `pto-isa` 现有行为保持一致。

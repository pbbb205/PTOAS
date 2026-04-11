## MODIFIED Requirements

### Requirement: TileLang DSL v1 MUST support static physical Tile shape with dynamic TensorView views and loop bounds

TileLang DSL v1 中，Tile physical shape MUST 是静态编译期常量。  
TensorView shape、slice 边界、loop bound 和 tail 相关 remaining value MAY 包含 runtime value。  
stable TensorView slicing 在本 profile 下 MUST 满足以下约束：

- 仍只支持 rank-2 TensorView slice
- `stop` MUST 显式给出
- `start` MAY 为常量或 runtime index expr
- `step` MUST 是静态正整数
- 第 0 轴 MAY 使用 `step > 1`
- 第 1 轴 MUST 保持 `step == 1`

`valid_shape` 仅可使用静态值或由 TensorView partition 直接推导。  
semantic / lowering MUST 保留标准化后的 `start/stop/step`，而不是只保留 slice extent。

#### Scenario: dynamic TensorView slice with non-zero start lowers successfully

- **WHEN** 用户在 stable kernel 中使用 `tensor[row_start:row_stop, col_start:col_stop]`，且 `row_start`、`row_stop`、`col_start`、`col_stop` 含 runtime index value
- **THEN** frontend MUST 生成合法的 authoring-form VPTO IR
- **AND** lowering MUST 能从标准化 slice 推导对应的 pointer offset 与 transfer extent
- **AND** Tile physical shape MUST 继续保持静态契约

#### Scenario: outer-axis static stepped slice is accepted for stable DMA inference

- **WHEN** 用户在 stable kernel 中使用第 0 轴带静态正步长的 TensorView slice，例如 `tensor[0:rows:2, 0:16]`
- **THEN** frontend MUST 接受该 stable slice profile
- **AND** lowering MUST 把该步长纳入 DMA stride / loop-size 推导，而不是退化成 contiguous copy

#### Scenario: unsupported slice profile is rejected before lowering

- **WHEN** 用户使用 rank > 2 slice、缺失 `stop`、dynamic `step` 或第 1 轴 `step != 1` 的 TensorView slice
- **THEN** frontend MUST 在生成 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出超出了当前 stable 2D slice / DMA profile

### Requirement: `dma_load` and `dma_store` MUST lower to VPTO DMA programming plus copy ops

TileLang DSL 的高层 `dma_load` / `dma_store` MUST 在 frontend lower 到当前合法 VPTO authoring surface：

- GM -> UB：必要的 `set_loop*_stride_outtoub` / `set_loop_size_outtoub` + `copy_gm_to_ubuf`
- UB -> GM：必要的 `set_loop*_stride_ubtoout` / `set_loop_size_ubtoout` + `copy_ubuf_to_gm`

DMA 参数 MUST 由标准化后的 TensorView slice `start/stop/step`、TensorView shape、Tile `shape/valid_shape` 和 padding/trim 配置推导。  
对 non-zero start 的 stable DMA，lowering MUST 生成等价的 pointer offset（例如 `pto.addptr`），而不是假定 source / destination 总是 zero-based contiguous view。  
对带 outer-axis 静态步长的 stable DMA，lowering MUST 把步长纳入 stride/loop-size 推导。  

`dma_load` 的 stable padding contract MUST 满足：

- `left_padding` / `right_padding` 影响 destination Tile 内的有效 copy window
- padded load MAY 在 `copy_gm_to_ubuf` 之前生成额外的 frontend prefill lowering
- `PadNull`、`PadFirstElem`、`PadValue` 的可实现子集 MUST 有明确 lowering 语义

`dma_store` 的 stable contract MUST 满足：

- `left_padding` / `right_padding` 定义 source Tile 的 interior trim window
- lowering MUST 只把 trim 后的 interior window 写回 destination TensorView slice
- 需要 GM-side fill 的 store padding 组合 MUST 显式 reject，直到 stable path 具备对应公开承载

当前 frontend-only stable path 无法真实表达的 backend-init / GM-side fill 语义 MUST 显式诊断，MUST NOT 通过 silently ignore 参数伪装成“已支持”。

#### Scenario: non-zero-start DMA lowers through inferred pointer offset and dynamic strides

- **WHEN** 用户在 stable kernel 中编写 `pto.dma_load(inp[row_start:row_stop, 0:16], tile)` 或 `pto.dma_store(tile, out[row_start:row_stop, 0:16])`
- **THEN** lowering MUST 生成对应的 DMA programming op 和 copy op
- **AND** 生成结果 MUST 反映 slice start 带来的 pointer offset 与 stride 变化
- **AND** 生成结果 MUST 符合当前 VPTO copy-family 的 authoring contract

#### Scenario: padded stable load lowers through prefill plus interior copy

- **WHEN** 用户在 stable kernel 中编写带 `pad_mode`、`left_padding`、`right_padding` 的 `pto.dma_load(src_slice, dst_tile, ...)`
- **THEN** lowering MUST 生成与该 padding 语义一致的 prefill/copy 组合
- **AND** destination Tile 的 interior copy window MUST 与 padding 后的 shape contract 一致

#### Scenario: trimmed stable store writes the interior window only

- **WHEN** 用户在 stable kernel 中编写 `pto.dma_store(src_tile, dst_slice, left_padding=1, right_padding=1)`
- **THEN** lowering MUST 只把 `src_tile` 的 interior trim window 写回 `dst_slice`
- **AND** `dst_slice` 的 extent MUST 与 trim 后的 source window 相匹配

#### Scenario: unsupported store fill or init combination is rejected explicitly

- **WHEN** 用户在 stable kernel 中请求当前 frontend-only path 尚无真实承载的 GM-side fill、store `pad_mode != PadNull` 或其他等价 backend-init 组合
- **THEN** frontend MUST 在生成 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出该组合超出了当前 stable frontend-only DMA profile

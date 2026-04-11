## ADDED Requirements

### Requirement: stable `dma_load` / `dma_store` surface MUST expose keyword DMA options and `PadMode`

TileLang DSL stable surface MUST 为 `pto.dma_load` / `pto.dma_store` 暴露正式的 keyword 参数入口，而不再把 high-level DMA 限定为 2 参数最小形态。  
stable public API MUST 提供 `PadMode`，并允许用户在 `dma_load` / `dma_store` 中通过 `pad_mode`、`pad_value`、`left_padding`、`right_padding`、`init_out_buffer` 表达高层 DMA 意图。  
frontend MUST 保留这些参数的语义信息进入后续 semantic/lowering，而不是在 AST 构建阶段静默丢弃 keyword 参数。

#### Scenario: stable DMA call accepts keyword arguments

- **WHEN** 用户在 stable kernel 中编写 `pto.dma_load(src_slice, dst_tile, pad_mode=PadMode.PadValue, pad_value=..., left_padding=2, right_padding=2)`
- **THEN** frontend MUST 接受该 call surface 的 keyword 形态
- **AND** keyword 参数 MUST 保留到后续 semantic 分析，而不是被静默忽略

#### Scenario: `PadMode` is part of the stable public surface

- **WHEN** 用户在 TileLang DSL 中引用 `PadMode.PadNull`、`PadMode.PadFirstElem` 或 `PadMode.PadValue`
- **THEN** frontend MUST 识别这些 stable surface 符号
- **AND** `PadMode` MUST NOT 继续被归类为 unsupported language construct

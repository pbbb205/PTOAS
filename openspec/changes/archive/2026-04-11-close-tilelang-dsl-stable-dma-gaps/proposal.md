# Proposal: 补齐 TileLang DSL stable 模式的 DMA 与切片契约

## 概述

`tilelang-dsl/docs/tilelang-dsl-guide.md` 已经把 stable `dma_load` / `dma_store` 描述为高层、可自动推导 DMA 参数的默认数据搬运接口，并公开了 `PadMode`、padding 参数和更宽的 TensorView slicing 语义。  
但当前 `tilelang-dsl/python/` 实现仍停留在最小 2 参数 DMA 和极窄 slice profile，导致 guide、unsupported 文档和真实 lowering 行为长期脱节。

本 change 的目标是在不切换出当前 `tilelang-dsl -> authoring-form VPTO` 前端主线的前提下，把 stable DMA/slice 契约收敛到一个真实、可测试、可文档化的 2D profile：补齐非零/dynamic start、静态正步长、DMA 参数自动推导、stable `PadMode` 入口，以及前端可安全落地的 padding / trim 行为。

## 背景与动机

当前仓库中已经存在以下明显缺口：

- guide 在 stable 章节承诺 `dma_load` / `dma_store` 会从 TensorView slice 自动推导 stride 与 loop size，但实现仍只支持 contiguous、zero-based、unit-step 的 rank-2 slice。
- guide 公开了 `PadMode`、`pad_value`、`left_padding`、`right_padding`、`init_out_buffer` 等高层接口，但当前 stable path 只接受 2 参数 DMA。
- `semantic.py` 目前把 TensorView slice 压缩成仅有 extent 的窄模型，无法忠实表达 `start/stop/step`，也无法支撑更真实的 DMA inference。
- `unsupported-features.md` 已经承认这些缺口存在，但 stable surface 长期停留在“文档已承诺、实现未补齐”的状态，会持续误导使用者并阻碍后续 sample / regression 编写。

如果不把这部分契约收敛清楚，stable 模式就无法成为可信的默认 authoring path，后续 matcher、advanced surface 和 sample 覆盖也会继续建立在不稳定边界之上。

## 目标

- 补齐 stable `dma_load` / `dma_store` 的 keyword surface，使 `PadMode`、padding 参数和相关配置能进入 frontend 语义分析。
- 把 stable TensorView slicing 扩展到可支持 non-zero/dynamic start、dynamic stop、静态正步长，并把这些信息保留到 lowering。
- 让 stable DMA inference 从 slice `start/stop/step`、TensorView shape、Tile `shape/valid_shape` 推导 pointer offset、loop stride 和 loop size，而不再只支持 full-tile contiguous profile。
- 定义 frontend-only stable padding 行为：
  - `dma_load` 支持 padding 参数与可实现的 padded-load lowering。
  - `dma_store` 支持基于 `left_padding/right_padding` 的 interior trim；GM-side fill 继续保持显式限制。
- 同步更新 tests、guide、unsupported 文档与 OpenSpec，使 stable contract 与实际实现重新一致。

## 非目标

- 不在本 change 中扩展到 rank > 2 的 TensorView slicing 或 DMA profile。
- 不支持省略 `stop` 的 Python slice 语义。
- 不支持 dynamic `step`，也不支持第 1 轴的 stepped gather/scatter DMA。
- 不借本 change 重新设计 TileLang DSL 的整体 lowering 架构，不把 stable DMA 改写为新的公开中间 IR。
- 不在本 change 中引入新的 backend capability；GM-side fill、任意 `pad_value` 直通 backend、以及 `init_out_buffer` 的完整 backend 语义不作为完成标准。

## What Changes

- 扩展 stable `dma_load` / `dma_store` public surface，支持 `pad_mode`、`pad_value`、`left_padding`、`right_padding`、`init_out_buffer` keyword 参数，并公开 `PadMode`。
- 扩展 stable TensorView slicing，支持 rank-2 profile 下的 non-zero/dynamic start、dynamic stop 和静态正步长，并把标准化后的 slice 元信息保留到 semantic layer。
- 修改 stable DMA lowering，使其从 normalized slice layout 与 Tile `valid_shape` 推导 `pto.addptr`、`set_loop*_stride_*`、`set_loop_size_*` 和 copy-family 参数。
- 为 padded load 与 trimmed store 定义 frontend-only authoring 行为，并对当前 frontend-only 路径无法真实表达的 GM-side fill / backend-init 语义给出显式诊断边界。
- 更新 `tilelang-dsl/docs/tilelang-dsl-guide.md`、`tilelang-dsl/docs/unsupported-features.md`、相关 tests 与 examples，使文档承诺、OpenSpec 契约和实现支持面保持一致。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `tilelang-dsl-surface`: 为 stable `dma_load` / `dma_store` 明确 keyword 参数和 `PadMode` public surface。
- `tilelang-dsl-vpto-lowering`: 扩展 stable 2D slicing / DMA inference / padding-trim lowering 契约，并明确 frontend-only 边界。

## 预期结果

- stable `dma_load` / `dma_store` 不再只是文档中的高层愿景，而是具备清晰、可测试的 2D authoring contract。
- stable slicing 和 DMA inference 能覆盖 guide 中最核心的 dynamic start / stride-aware authoring 场景，而不是继续被 zero-based contiguous profile 卡死。
- guide、unsupported 文档和 OpenSpec 不再对 stable DMA 能力给出互相冲突的描述。
- 对当前 frontend-only 路径暂时无法承载的行为，frontend 会给出明确、工程化的限制诊断，而不是 silently accept 或产生误导性输出。

## 成功标准

- 新增 `openspec/changes/close-tilelang-dsl-stable-dma-gaps/`，包含 `proposal.md`、`design.md`、`tasks.md`。
- 新增 spec delta：
  - `openspec/changes/close-tilelang-dsl-stable-dma-gaps/specs/tilelang-dsl-surface/spec.md`
  - `openspec/changes/close-tilelang-dsl-stable-dma-gaps/specs/tilelang-dsl-vpto-lowering/spec.md`
- proposal/design/tasks/specs 明确写清：
  - stable DMA keyword surface 和 `PadMode` 入口
  - stable 2D slice profile 的动态起点与静态步长边界
  - automatic DMA inference 的输入来源
  - padded `dma_load` 与 trimmed `dma_store` 的 frontend-only 行为
  - 当前 frontend-only 路径下继续延期的 GM-side fill / backend-init 语义

## Impact

- 受影响目录：
  - `tilelang-dsl/python/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/examples/`
  - `tilelang-dsl/docs/`
  - `openspec/specs/`
- 受影响 public API：
  - `pto.dma_load(...)`
  - `pto.dma_store(...)`
  - `PadMode`
  - TensorView slicing 的 stable accepted profile
- 受影响验证路径：
  - stable authoring kernel 的 semantic/lowering regression
  - `descriptor.mlir_text()` 输出的 DMA programming / copy-family 形态

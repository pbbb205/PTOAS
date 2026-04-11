## Context

### 范围

本 design 只覆盖 TileLang DSL stable 模式下的 2D TensorView slicing 与 high-level DMA surface：

- `pto.dma_load`
- `pto.dma_store`
- `PadMode`
- 与 DMA inference 直接相关的 TensorView slice `start/stop/step`

它不覆盖：

- matcher / registry / advanced surface
- rank > 2 的 TensorView/DMA profile
- backend 新 capability
- A5 text / LLVM emission 规则变更

### 当前状态

当前实现存在四个直接相关的事实：

1. `tilelang-dsl/docs/tilelang-dsl-guide.md` 已把 stable DMA 描述为高层自动推导接口，并公开了 `PadMode`、padding 参数以及更宽的 slicing 语义。
2. `tilelang-dsl/python/tilelang_dsl/frontend_ast.py` 的 `FrontendCallExpr` 目前只保存 positional `args`，无法承载 DMA keyword 参数。
3. `tilelang-dsl/python/tilelang_dsl/semantic.py` 当前把 TensorView slice 限定为：
   - rank-2
   - 显式 `stop`
   - `start == 0`
   - `step == 1`
   同时 `SemanticTensorSliceType` 只保留 extent，不能表达 `start/stop/step`。
4. `tilelang-dsl/python/tilelang_dsl/lowering.py` 目前把 stable DMA 展开为固定常量参数的 `set_loop_size_* + copy_*` 组合，无法从 slice layout 推导 offset / stride / trim / padding。

与此同时，repo 内部的 PTO/VPTO lowering 已经存在可参考的 shape/stride contract，但 TileLang DSL stable path 当前仍选择直接产出 authoring-form VPTO text，而不是先 materialize `pto.tload` / `pto.tstore` 再复用 backend lowering。

### 实现约束

- 本 change 保持当前 `tilelang-dsl -> authoring-form VPTO` 主线，不引入新的公开中间 IR。
- stable DMA profile 继续限定为 statically specialized rank-2 UB Tile。
- 设计必须显式区分“前端可综合的高层行为”和“当前 authoring/backend 路径无法真实承载的行为”，不能靠 silent no-op 冒充支持。
- `unsupported-features.md`、guide 和 OpenSpec 需要同步更新，避免继续出现 contract 漂移。

## Goals / Non-Goals

**Goals:**

- 让 stable DMA surface 真正接受 `PadMode` 和 keyword 参数，而不是继续停留在 2 参数最小形态。
- 让 stable TensorView slice 在 2D profile 内支持 non-zero/dynamic start、dynamic stop 和静态正步长。
- 让 lowering 基于 normalized slice layout、TensorView shape 和 Tile `valid_shape` 推导 offset / stride / loop size。
- 为 padded `dma_load` 与 trimmed `dma_store` 定义稳定、可测试的 frontend-only 行为。
- 对当前 frontend-only 路径不能真实表达的行为给出明确边界和 diagnostics。

**Non-Goals:**

- 不支持 rank > 2 slice / DMA。
- 不支持 dynamic `step`。
- 不支持第 1 轴 stepped DMA。
- 不把 stable DMA 改写为 backend-driven `pto.tload` / `pto.tstore` pipeline。
- 不在本 change 中补齐 GM-side fill 或 backend-init capability。

## Decisions

### 1. 保持 stable DMA 继续直接 lower 到 authoring-form VPTO，而不是切换到 `pto.tload` / `pto.tstore`

决策：

- `tilelang-dsl/python/tilelang_dsl/lowering.py` 继续直接生成 `pto.addptr`、`set_loop*_stride_*`、`set_loop_size_*` 和 `copy_*`。
- 不在本 change 中改变 TileLang DSL 的 lowering 架构边界。

原因：

- 当前 `descriptor.mlir_text()` / `verify()` 已经围绕 authoring-form VPTO 建立稳定路径。
- 这次 change 的核心问题是 stable contract 缺口，不是重新设计整体 lowering pipeline。

备选方案：

- 先 materialize `pto.tload` / `pto.tstore` 再复用 repo lowering
  - 放弃原因：会扩大本 change 的影响面，并把“补 stable DMA 缺口”变成“改写 DSL lowering 主线”。

### 2. 扩展前端 AST 与 semantic 模型，保留 normalized slice 与 DMA options，而不是只保存 extent

决策：

- `FrontendCallExpr` 增加 keyword 参数承载。
- stable DMA call 在 semantic 层保存：
  - `pad_mode`
  - `pad_value`
  - `left_padding`
  - `right_padding`
  - `init_out_buffer`
- TensorView slice 保存标准化后的每轴 `start/stop/step`，不再只保存 `extent`。

原因：

- 真实的 DMA inference 依赖 offset、step 和 trim 信息；仅保留 extent 无法推导 `pto.addptr` 与 loop stride。
- keyword 参数必须在 frontend 就固定下来，不能等到 lowering 再反向猜测。

备选方案：

- 继续沿用 “slice -> extents only” 的窄模型
  - 放弃原因：无法闭合 non-zero start、outer-axis step、padding/trim 等本 change 目标。

### 3. stable slicing profile 固定为“2D + 显式 stop + dynamic start/stop + 静态正步长”，且第 1 轴仍要求 `step == 1`

决策：

- 继续只支持 rank-2 TensorView slicing。
- `stop` 必须显式给出。
- `start` 可以是常量或 runtime index expr。
- `step` 必须是静态正整数。
- 第 0 轴允许 `step > 1`。
- 第 1 轴必须保持 `step == 1`。

原因：

- 这样既能覆盖 guide 中最核心的 stride-aware authoring，又不会把 stable DMA 推入当前 copy-family 无法表达的 gather/scatter 形态。

备选方案：

- 完全放开 dynamic `step`
  - 放弃原因：dynamic stride 会把 DMA legality、shape 校验和 lowering 复杂度整体抬高，不适合作为 stable 2D profile 的第一步。
- 两个轴都允许 stepped DMA
  - 放弃原因：第 1 轴 stepped copy 需要当前 stable authoring path 尚未具备的 gather/scatter 语义。

### 4. `dma_load` 采用 “prefill + interior copy” 的 frontend-only padding 方案；`dma_store` 采用 trim-and-validate 方案

决策：

- `dma_load`：
  - 先按 `pad_mode`、`pad_value`、`left_padding`、`right_padding` 决定目的 Tile 需要的 padding band。
  - padding band 由 frontend 生成稳定的 prefill lowering。
  - interior 数据仍通过 `copy_gm_to_ubuf` 进入 UB。
- `dma_store`：
  - `left_padding/right_padding` 定义 source tile 的 interior window。
  - lowering 只把 interior window 写回 destination slice。
  - GM-side fill 不属于本 change。

原因：

- `dma_load` 的 padding 可以在 UB 侧通过 frontend 合成，不需要 backend 新 contract。
- `dma_store` 若要把 padding value 主动写到 GM 边缘，会要求当前 frontend-only 路径之外的 GM-side fill 语义，因此本 change 选择 trim-and-validate。

备选方案：

- 让 `dma_store` 也尝试合成完整 GM-side fill
  - 放弃原因：会越过当前 stable authoring path 的真实边界，并引入额外 backend 依赖。
- 继续让 non-default padding 参数全部 reject
  - 放弃原因：无法收敛 guide 与 stable surface 的主要缺口。

### 5. 对 frontend-only 路径尚无稳定承载的行为保持显式诊断，而不是伪支持

决策：

- `dma_store` 的 `pad_mode != PadNull`、GM-side fill 和等价 backend-init 语义继续显式 reject。
- `init_out_buffer` 仅在能够映射到本次定义的 frontend prefill 行为时放行；超出该 profile 的组合继续报错。
- diagnostics 要直接说明“当前 stable frontend-only DMA profile 未支持的原因”，而不是退化成模糊 type error。

原因：

- 这能避免 public surface 看起来“已经支持”，但实际 lowering 只是在 silently ignore 参数。

## Risks / Trade-offs

- [Risk] 在 frontend 合成 padded-load prefill 会引入额外 lowering 复杂度  
  Mitigation：严格限定到 rank-2、静态 Tile shape、outer-axis stepped profile，并用 regression 锁定生成形态。

- [Risk] `dma_store` 仍然不能提供 guide 中最理想的 GM-side fill 语义  
  Mitigation：在 spec 和 guide 中明确把 store padding 收敛为 trim-and-validate，而不是继续保留模糊承诺。

- [Risk] dynamic start/stop 与步长推导会增加 semantic 和 lowering 状态量  
  Mitigation：统一在 semantic 层做 normalized slice 表示，避免后续多个 lowering helper 各自解释 slice。

- [Risk] `PadMode` public surface 放开后，用户更容易触碰尚未实现的组合  
  Mitigation：在 diagnostics 和 `unsupported-features.md` 中精确列出仍被限制的组合，不使用“只支持 2 参数”这类过粗表述。

## MODIFIED Requirements

### Requirement: TileLang DSL v1 MUST support the fixed elementwise lowering profile

TileLang DSL v1 lowering MUST 支持以下固定 support matrix：

- 2D `TensorView`
- 1D/2D `Tile`
- `dma_load`
- `dma_store`
- `make_mask(dtype, PAT.*)` / `make_mask(dtype, remaining)`
- `vlds`
- `vsts`
- unary：`vabs`, `vrelu`, `vexp`, `vnot`
- binary：`vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`
- vector-scalar：`vadds`, `vsubs`, `vmuls`, `vdivs`, `vmaxs`, `vmins`
- `for range(lb, ub, step)`
- `if/else`
- `set_flag`, `wait_flag`, `pipe_barrier`
- `@inline_proc` helper function 定义与 call（含返回值 call）

support matrix 外的 surface MUST 在 frontend reject。  
对 `@inline_proc`，frontend/lowering MUST 生成“入口 kernel + private helper + `func.call`”的 authoring-form module 结构；`mlir_text()` 阶段 MAY 保留这些 call 边界。

#### Scenario: representative elementwise kernel lowers to authoring-form VPTO with inline_proc calls

- **WHEN** 用户编写由 `TensorView`、`Tile`、高层 DMA、typed mask、elementwise vector op、`for`、`if`、基础 sync 和 `inline_proc` 调用组成的 kernel
- **THEN** frontend MUST 产出只包含 `func.func`、`arith`、`scf` 和合法 `pto.*` authoring surface 的 VPTO IR
- **AND** 该 IR MAY 包含标注为 inline_proc helper 的 private `func.func` 与 `func.call`

## ADDED Requirements

### Requirement: VPTO backend mainline MUST force-inline TileLang inline_proc helpers

在 `ptoas --pto-backend=vpto` 主线中，backend MUST 在早期 pass 阶段强制 inline TileLang inline_proc helper 调用。  
强制 inline 的目标 MUST 通过 helper 属性（例如 `pto.tilelang.inline_proc`）筛选，避免误作用于普通函数调用。  
backend MUST 支持“带返回值 `func.call`”内联替换，不仅是 `() -> ()` 调用。  
inline 完成后，面向 inline_proc helper 的 `func.call` MUST 被消除，且无引用 private helper MUST 被清理。

#### Scenario: backend pipeline removes inline_proc calls before downstream lowering

- **WHEN** 输入 module 包含 `pto.tilelang.instance` kernel 对 `pto.tilelang.inline_proc` private helper 的 `func.call`
- **THEN** VPTO backend 主线 MUST 在后续依赖平坦 body 的 pass 之前完成 inline
- **AND** inline 后 module MUST NOT 残留面向 inline_proc helper 的 `func.call`

#### Scenario: return-valued helper call is inlined with SSA replacement

- **WHEN** inline_proc helper 返回一个值，caller 使用 `func.call` 结果参与后续计算
- **THEN** backend inline MUST 正确替换 call result 的 SSA use 链
- **AND** 结果 IR MUST 保持类型一致并通过后续 legality 校验

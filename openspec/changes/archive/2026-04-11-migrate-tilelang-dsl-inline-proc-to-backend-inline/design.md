## Context

### 范围

本 design 覆盖 `inline_proc` 的端到端迁移：

- TileLang frontend：不再展开 inline_proc，改为保留 helper/call。
- TileLang semantic/lowering：建立 helper 函数与 `func.call` 语义模型，支持返回值调用。
- `ptoas` backend：在 VPTO backend 主线中强制 inline 并删除死 helper。
- OpenSpec/文档/测试：同步新契约。

不覆盖：

- 多模块 inline_proc 解析与跨模块可见性扩展。
- `*args` / `**kwargs` / kw-only 参数模型。
- matcher/registry 与 inline_proc 的新交互语义。

### 当前状态

1. `inline_proc` 当前在 frontend AST 阶段做语句级展开，导致宏替换行为与函数语义混杂。
2. 当前实现通过捕获校验与名字改写保障可用，但复杂度和维护成本较高。
3. `specialized.mlir_text()` 目前默认看不到 `func.call`，inline 边界与 backend 优化边界不清晰。
4. `ptoas` 已有 `PTOInlineLibCall` pass 与 TileLang template 路径，但当前 pass 对“带返回值 call”与“TileLang kernel 作为 caller”支持不完整，不足以直接承接新版 inline_proc。

### 实现约束

- 不保留 feature switch，直接替换现有 frontend-expand 语义。
- 解析范围固定为“同模块显式注册 inline_proc”。
- 必须保持 fail-fast：隐式捕获、递归/互递归、不支持参数模型在 frontend 直接报错。
- `mlir_text()` 允许 `func.call`；最终 backend 产物必须消除 inline_proc 调用。

## Goals / Non-Goals

**Goals:**

- 以函数语义重建 inline_proc：默认参数、关键字调用、返回表达式、表达式调用全部可用。
- 让 frontend 专注语义建模，inline 优化责任收敛到 `ptoas` backend 主线。
- 保证 `PTOInlineLibCall` 能内联带结果调用，并清理私有死 helper。
- 用回归测试同时锁定 frontend 语义与 backend inline 收敛行为。

**Non-Goals:**

- 不新增跨模块 inline_proc 导入调用规则。
- 不支持 `*args` / `**kwargs` / kw-only。
- 不把 inline_proc 语义下放到 emission 阶段；仍在 VPTO backend 主线 early stage 完成。

## Decisions

### 1. 前端不展开 inline_proc，改为 helper+call 模型

决策：

- `build_frontend_kernel_node` 同时产出 kernel body 与 inline_proc body（受控子集）。
- kernel/body 中 inline_proc 调用保留为命名调用节点，不在前端替换为语句块。

原因：

- 彻底消除前端宏展开复杂度，避免返回语义与作用域卫生问题反复回归。

备选方案：

- 保留 frontend 展开并继续扩展语义。
  - 放弃原因：维护复杂度高，且与“函数语义优先”目标冲突。

### 2. 参数绑定采用 Python 子集：位置参数 + 关键字 + 默认值

决策：

- 调用绑定支持 positional + keyword + defaults。
- 继续 reject：`*args` / `**kwargs` / kw-only。
- 绑定错误（重复赋值、缺参、未知关键字）在 frontend fail-fast。

原因：

- 能满足可用性诉求，同时保持实现边界可控。

备选方案：

- 一次性支持完整 Python 参数模型。
  - 放弃原因：复杂度和歧义显著上升，不适合作为本次迁移范围。

### 3. 解析范围固定为同模块显式注册

决策：

- 仅允许解析当前 kernel 所在模块内已注册的 `@inline_proc`。
- 不做全局唯一名 fallback，不做跨模块自动解析。

原因：

- 避免解析歧义，确保 descriptor 构建与 materialization 行为一致。

### 4. backend-inline 接入 VPTO backend 主线早期

决策：

- `PTOInlineLibCall` 在 `addVPTOBackendMainlinePasses` 中无条件接入（不依赖 `--enable-tile-op-expand`）。
- pass 运行位置固定在 VPTO authoring validation 之前。

原因：

- 保证后续依赖平坦 body 的 pass 不会看到 inline_proc helper 调用。

### 5. inline 目标限定为 TileLang inline helper，且支持带结果 call

决策：

- helper 函数增加专用属性（`pto.tilelang.inline_proc`）。
- `PTOInlineLibCall` 仅对受控 inlineable callee 生效。
- 扩展 `inlineCall` 支持 call result 映射与替换，不再仅限 `() -> ()`。

原因：

- 既满足功能需求，又避免误内联普通调用。

### 6. `mlir_text()` 保留 `func.call` 可观察边界

决策：

- `mlir_text()` 输出允许包含 kernel + private inline helper + `func.call`。
- “最终无调用残留”只承诺在 `ptoas` backend 主线后成立。

原因：

- 调试可观测性更好，前后端职责边界清晰。

## 测试策略

- Python 单测：
  - 默认参数、关键字调用、返回表达式、表达式调用正向覆盖。
  - 隐式捕获、递归/互递归、不支持参数模型负向覆盖。
  - `mlir_text()` 断言包含 helper 函数和 `func.call`。
- backend `lit` 回归：
  - 构造带 `pto.tilelang.inline_proc` helper call 的输入，验证 VPTO backend 主线后调用被消除。
  - 覆盖带返回值调用的 inline 正确性。

## Risks / Trade-offs

- [Risk] 增加 helper 函数后 module 结构更复杂，可能影响现有文本断言测试。  
  Mitigation：更新相关断言为“关键行为断言”，避免过度绑定无关格式。

- [Risk] `PTOInlineLibCall` 行为变更可能影响现有 OP-Lib 路径。  
  Mitigation：保持 callee 筛选策略受控；新增回归覆盖 TileLang inline 与 OP-Lib 两条路径。

- [Risk] 语义层新增 inline_proc specialization 可能引入类型绑定回归。  
  Mitigation：按 call signature specialization 建 helper，并增加表达式返回值/类型回归。

## Migration Plan

1. 先落 OpenSpec delta，冻结新契约。
2. 迁移 Python frontend/semantic/lowering 到 helper+call 模型。
3. 扩展并接线 `PTOInlineLibCall` 到 VPTO backend 主线。
4. 更新测试与文档，跑最小验证命令。

回滚策略：

- 如 backend inline 接线出现阻断，可先回滚 pass 接线与 helper attr 识别改动，保留 OpenSpec change 未归档状态并修复后再提交。

## Open Questions

- 无（本 change 边界已锁定）。

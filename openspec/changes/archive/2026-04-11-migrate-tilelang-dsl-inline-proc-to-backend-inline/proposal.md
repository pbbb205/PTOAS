# Proposal: 将 TileLang DSL `inline_proc` 迁移为 backend-inline 主线

## 概述

当前 `inline_proc` 采用 frontend AST 展开，导致参数绑定、返回语义、作用域卫生和递归检测耦合在前端重写逻辑中，维护成本高且行为容易偏离“函数调用”直觉。  
本 change 将 `inline_proc` 迁移为 backend-inline：TileLang frontend 保留 helper `func.func` 与 `func.call`，在 `ptoas` VPTO backend 主线早期强制 inline 并清理死 helper。

## 背景与动机

现状存在三类直接问题：

1. 前端展开路径把 `inline_proc` 变成“宏替换”语义，难以稳定支持默认参数、关键字参数、返回值调用与表达式调用。
2. 前端展开已引入复杂的 capture 校验与名字改写逻辑，调试与行为验证成本持续上升。
3. `mlir_text()` 与最终 backend 产物的调用边界不清晰，用户难以理解“语义正确性”和“性能内联”各由哪一层负责。

因此需要把 `inline_proc` 还原为函数语义建模，并把 inline 优化责任收敛到 `ptoas` backend pipeline。

## 目标

- 把 `inline_proc` 从 frontend 展开迁移为 backend-inline，不保留 feature switch。
- 支持 `inline_proc` 的默认参数、关键字调用、返回表达式和表达式位置调用。
- 保持 fail-fast 约束：禁止隐式捕获、禁止递归/互递归、禁止 `*args` / `**kwargs` / kw-only 参数。
- 让 `specialized.mlir_text()` 可观察 helper `func.func` 与 `func.call`。
- 在 `ptoas --pto-backend=vpto` 主线中保证 `inline_proc` helper 调用被强制消除。

## 非目标

- 不在本 change 中引入新的 matcher 或多模块解析策略；`inline_proc` 仍限定为同模块显式注册可解析。
- 不在本 change 中放开 `*args` / `**kwargs` / kw-only 参数语义。
- 不在本 change 中保留旧 frontend-expand 行为作为兼容入口。
- 不在本 change 中新增独立 capability；沿用现有 TileLang DSL capability 并追加 delta。

## What Changes

- **BREAKING**：`inline_proc` 不再在 frontend 物化阶段展开，`mlir_text()` 允许出现 helper `func.func` 与 `func.call`。
- `@inline_proc` 参数模型升级：允许默认参数与关键字调用；允许返回表达式；允许表达式位置调用。
- frontend 继续对隐式捕获、递归/互递归、不支持参数模型 (`*args/**kwargs/kw-only`) 做 source-located reject。
- semantic/lowering 增加 `inline_proc` helper function + callsite 建模，保留函数调用语义。
- `ptoas` VPTO backend 主线新增（或扩展）强制 inline 阶段，目标限定为 TileLang inline helper，inline 后清理私有死函数。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `tilelang-dsl-surface`: `inline_proc` public surface 从 statement-only frontend-expand 迁移为函数语义 + backend-inline，补齐默认参数、关键字调用、返回值和表达式调用能力。
- `tilelang-dsl-diagnostics`: 新增/调整 `inline_proc` 参数绑定、递归检测、捕获规则与不支持参数模型的 fail-fast 诊断契约。
- `tilelang-dsl-vpto-lowering`: 定义 `inline_proc` helper/call 的 lowering 形态，以及 VPTO backend 主线强制 inline 与死 helper 清理要求。

## 预期结果

- `inline_proc` 行为与 Python 函数调用直觉对齐，前端不再维护复杂 AST 宏展开。
- `mlir_text()` 保留调用边界用于调试；最终 backend 产物由 `ptoas` 主线统一消除 `inline_proc` 调用。
- 回归测试可同时覆盖 frontend 语义（参数绑定/诊断）与 backend inline 收敛行为（调用消除）。

## 成功标准

- 新增 `openspec/changes/migrate-tilelang-dsl-inline-proc-to-backend-inline/`，包含 `proposal.md`、`design.md`、`tasks.md`。
- 新增 spec delta：
  - `specs/tilelang-dsl-surface/spec.md`
  - `specs/tilelang-dsl-diagnostics/spec.md`
  - `specs/tilelang-dsl-vpto-lowering/spec.md`
- 代码层完成 backend-inline 迁移并通过最小验证：
  - `python3 -m unittest tilelang-dsl/tests/test_tilelang_dsl_v1.py -k inline_proc`
  - 覆盖 inline 主线的 `lit` 回归
  - `openspec validate migrate-tilelang-dsl-inline-proc-to-backend-inline --type change --strict --json --no-interactive`

## Impact

- 受影响目录：
  - `tilelang-dsl/python/tilelang_dsl/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/docs/user_guide/`
  - `tools/ptoas/`
  - `lib/PTO/Transforms/`
  - `openspec/changes/migrate-tilelang-dsl-inline-proc-to-backend-inline/`
- 受影响 public API：`@inline_proc` 调用与返回语义。
- 受影响 backend 行为：VPTO backend 主线新增（或扩展）强制 inline 阶段。

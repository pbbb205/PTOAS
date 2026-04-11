## ADDED Requirements

### Requirement: `@inline_proc` MUST expose function-call semantics and defer inlining to backend

TileLang DSL v1 中，`@inline_proc` MUST 采用函数语义建模，而不是 frontend AST 宏展开。  
`inline_proc` 定义 MUST 支持默认参数；调用 MUST 支持位置参数与关键字参数混合绑定；helper body MUST 允许返回表达式。  
`inline_proc` 调用 MUST 允许出现在语句位置和表达式位置。  
`specialized.mlir_text()` MUST 允许暴露入口 kernel `func.func`、private helper `func.func` 与 `func.call`，而不是强制在 frontend 阶段消除调用。

#### Scenario: inline_proc accepts defaults and keyword call syntax

- **WHEN** 用户定义 `@pto.inline_proc` helper，参数包含默认值，并在 kernel 中用关键字调用该 helper
- **THEN** frontend MUST 按 Python 子集完成参数绑定并接受该调用
- **AND** 该调用 MUST NOT 因“仅支持位置参数”而被拒绝

#### Scenario: inline_proc return expression can be used in expression position

- **WHEN** 用户在 `@inline_proc` helper 中返回表达式，并把 helper 调用放在另一个表达式上下文中
- **THEN** frontend MUST 接受该 surface 并保留调用结果语义
- **AND** `mlir_text()` 结果 MAY 包含对应的 `func.call` 结果值绑定

## MODIFIED Requirements

### Requirement: v1 MUST reject unsupported Python syntax and unsupported DSL calls before IR generation

TileLang DSL v1 frontend MUST 只接受受限 Python 子集。  
`while`、list/dict/set comprehension、arbitrary external function call、未注册 DSL op、以及其他超出 v1 surface 的 Python 结构 MUST 在 frontend 被拒绝。  
对命名调用，frontend MUST 仅放行“同模块显式注册的 `@inline_proc`”目标；其余 `namespace=None` 命名调用 MUST 继续按 unsupported external function call 拒绝。

#### Scenario: unsupported Python construct is rejected before lowering

- **WHEN** kernel body 使用 `while`、comprehension、任意非 `pto.*` function call（且不是同模块已注册 inline_proc）或未纳入 v1 support matrix 的 DSL call
- **THEN** frontend MUST 在生成任何 VPTO IR 之前报错
- **AND** 诊断 MUST 指明违规的 Python construct 或 DSL call 名称

## ADDED Requirements

### Requirement: inline_proc diagnostics MUST fail fast on capture/recursion/unsupported parameter forms

针对 `@inline_proc`，frontend diagnostics MUST fail-fast 覆盖以下语义约束：

- 隐式捕获 MUST 报错
- 递归/互递归 MUST 报错
- `*args` / `**kwargs` / kw-only 参数 MUST 报错
- 调用绑定错误（重复赋值、缺参、未知关键字）MUST 报错

#### Scenario: implicit capture in inline_proc is rejected with source location

- **WHEN** `@inline_proc` helper 体内引用了非参数、非局部定义的外部符号
- **THEN** frontend MUST 在 materialization 前报错
- **AND** 诊断 MUST 指向 helper 源位置并标明 capture 违反约束

#### Scenario: recursion and mutual recursion are rejected

- **WHEN** 某个 `inline_proc` 直接递归调用自身，或两个 helper 形成互递归调用环
- **THEN** frontend MUST 拒绝该定义/调用图
- **AND** 诊断 MUST 明确指出 recursion 或 mutual recursion

#### Scenario: unsupported inline_proc signature forms are rejected

- **WHEN** 用户为 `@inline_proc` 使用 kw-only 参数、`*args` 或 `**kwargs`
- **THEN** frontend MUST 直接报错
- **AND** 诊断 MUST 明确指出当前 v1 不支持该参数模型

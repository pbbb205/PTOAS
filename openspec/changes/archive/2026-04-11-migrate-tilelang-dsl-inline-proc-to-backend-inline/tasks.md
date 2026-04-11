## 1. OpenSpec delta 落定

- [x] 1.1 完成 `specs/tilelang-dsl-surface/spec.md`，定义 `inline_proc` 函数语义与 `mlir_text()` 可见调用边界
- [x] 1.2 完成 `specs/tilelang-dsl-diagnostics/spec.md`，定义 inline_proc fail-fast 诊断与命名调用放行边界
- [x] 1.3 完成 `specs/tilelang-dsl-vpto-lowering/spec.md`，定义 helper/call lowering 形态与 backend 强制 inline 契约

## 2. TileLang frontend/semantic/lowering 迁移

- [x] 2.1 移除 frontend AST 展开路径，保留 inline_proc 为命名调用并放开表达式位置调用
- [x] 2.2 在 frontend 参数绑定中支持 inline_proc 默认参数和关键字调用，同时保留 `*args/**kwargs/kw-only` 拒绝
- [x] 2.3 在 semantic 层建立 inline_proc helper call 语义节点，允许 `namespace=None` 的受控命名调用进入分析
- [x] 2.4 在 lowering 层渲染 kernel + private helper 多函数 module，生成 `func.call` 与返回值绑定

## 3. ptoas backend inline pass 与 pipeline 接线

- [x] 3.1 扩展 `PTOInlineLibCall` 以支持带返回值 `func.call` 的 inline 替换
- [x] 3.2 将 inline 目标限定为 TileLang inline_proc helper 属性，避免误内联普通调用
- [x] 3.4 inline 后清理无引用 private helper，并验证主线不残留 inline_proc helper call

## 4. 回归测试与文档迁移

- [x] 4.1 更新 `tilelang-dsl/tests/test_tilelang_dsl_v1.py`：默认参数/关键字/返回表达式改为正向测试
- [x] 4.2 新增 inline_proc 表达式位置调用和 `mlir_text()` helper+`func.call` 断言测试
- [x] 4.3 保留并验证负向测试：隐式捕获、递归/互递归、`*args/**kwargs/kw-only`
- [x] 4.4 增加 ptoas 侧 `lit` 回归：验证 VPTO backend 主线消除 inline_proc helper 调用（含带返回值 case）
- [x] 4.5 更新 `tilelang-dsl/docs/user_guide/08-control-flow.md`，迁移为 backend-inline 语义描述

## 5. 验证命令与结果记录

- [x] 5.1 执行 `python3 -m unittest tilelang-dsl/tests/test_tilelang_dsl_v1.py -k inline_proc`
- [x] 5.2 执行覆盖强制 inline 生效路径的 `lit`/pipeline 回归并记录结果
- [x] 5.3 执行 `openspec validate migrate-tilelang-dsl-inline-proc-to-backend-inline --type change --strict --json --no-interactive`
- [x] 5.4 在 change 记录中汇总本次实现的通过项与未覆盖项（若有）

### 5.4 验证结果记录（2026-04-10）

- 5.1 命令与结果：
  - `PYTHONPATH=tilelang-dsl/python python3 -m unittest tilelang-dsl/tests/test_tilelang_dsl_v1.py -k inline_proc`
  - `Ran 15 tests in 0.009s, OK`
- 5.2 命令与结果：
  - `llvm-lit -sv test/basic/tilelang_inline_proc_backend_inline.pto test/basic/inline_libcall_result_rewrite.pto test/basic/inline_libcall_filter_tilelang_scope.pto test/basic/vpto_mainline_inline_proc_cleanup.pto test/basic/expand_tile_op_tilelang.pto`
  - `Passed: 5/5`
- 5.3 命令与结果：
  - `openspec validate migrate-tilelang-dsl-inline-proc-to-backend-inline --type change --strict --json --no-interactive`
  - `valid: true, issues: []`
- 未覆盖项：
  - 未执行全量 `lit`/`ctest` 套件；本次仅覆盖 inline_proc 迁移相关的定向回归路径。

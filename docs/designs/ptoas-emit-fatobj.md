# `ptoas emit fatobj`

## 输入输出形式

### 输入

```mlir
module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @helper(...) {
  }
  func.func @foo(...) attributes {pto.kernel} {
    ...
  }
}

module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  func.func @helper(...) {}
  func.func @foo(...) attributes {pto.kernel} {
    ...
  }
}
```

允许只有一个 module，但是最多每个 kernel_kind 一个 module，只有vector|cube两类 module

带 `pto.kernel` 的函数在输入中保留逻辑函数名，不要求输入侧手工拼接 `_mix_aic` / `_mix_aiv` 后缀。旧属性名 `pto.aicore` 仍被兼容识别，但新输入应使用 `pto.kernel`。

### 输出

fatobj 对象文件

## 每个模块的工作

### `ptoas`

`--pto-backend=vpto` 直接输出 fatobj。这里移除的是对外的 llvm ir/bc 输出模式。所有修改发生在 vpto 路径内。emitc 路径不要动，不是我们关心的内容。

0. 对于两个并列 module，mlir 的 parser 会自动做一个嵌套。为了保证 pass pipeline 不产生分歧，单个 module 的场景也主动包裹为相同的嵌套结构。

1. 当前 `ptoas.cpp` 中进入 VPTO 路径有两处。这两处入口都需要改造并支持 fatobj 输出能力，但这里强调的是“两处都要改”，不是把它们在控制流上强行归并成一个入口：

- 第一处是 `effectiveBackend == PTOBackend::VPTO && inputIsVPTOIR && !hasTileOpsToExpand` 的直入 VPTO 分支。这条路径在 [`tools/ptoas/ptoas.cpp:1605`](/home/mouliangyu/projects/github.com/mouliangyu/PTOAS-3/tools/ptoas/ptoas.cpp#L1605) 附近，当前会在必要时先做 `inlineTilelangHelpersOnVPTOInput`，然后直接 `return emitVPTOBackendResult(...)`。
- 第二处是通用 PTO 前端 pipeline 跑完之后的 `effectiveBackend == PTOBackend::VPTO` 分支。这条路径在 [`tools/ptoas/ptoas.cpp:1670`](/home/mouliangyu/projects/github.com/mouliangyu/PTOAS-3/tools/ptoas/ptoas.cpp#L1670) 附近，当前会先打印 seam IR、再 `lowerPTOToVPTOBackend`，最后 `return emitVPTOBackendResult(...)`。

2. 进入 VPTO 路径后，首先保证 module 自动嵌套为如下形式，如果只有单个 module，就手动加一层。

```mlir
module {
    module {
    }
    module { ; 如果只有一个 module 就没有这第二个子 module
    }
}
```

3. `pto.kernel_kind` 需要位于最底层的 module。

4. 所有 pass 都通过 nest pm 驱动，不允许手动切分 module 分别跑 pass pipeline。这是嵌套 module 的统一驱动方式。

5. `ptoas` 负责统一调度，但不负责具体链接细节。它负责：

- 调用 `VPTOHostStubEmission` 生成 stub 源码字符串
- 调用 `VPTOLLVMEmitter` 生成 cube|vector 两个 llvm module 结构
- 将 stub、cube、vector 组件输入给 fatobj emission 组件，并直接把最终结果写入 `outputFile`

6. 不要修改 emitc 路径的代码。

### `VPTOHostStubEmission`

1. 负责 stub 源码字符串的生成，依据输入中 `pto.kernel` 函数的签名和符号约定生成对应 stub

2. cube 和 vector module 中的同名 `pto.kernel` 函数共享同一个 stub 函数

### `VPTOLLVMEmitter`

1. 负责 llvm module 生成工作，prepare 和 translate 合并，通过同一个 nest pm 驱动。不允许手动切分 module 分别跑 pass pipeline

2. 对外职责是接收嵌套 module 输入，并输出按 `kernel_kind` 拆分好的 vector / cube llvm module

3. 对带 `pto.kernel` 的函数，按所属 `kernel_kind` 自动补真实 device 符号后缀：

- vector 补 `_mix_aiv`
- cube 补 `_mix_aic`

输入侧只保留逻辑函数名，不在输入 IR 中手工编码这个后缀。

4. 当前文件中有很多函数本身不是 module pass，无法直接注册到 nest pm 中，需要用 pass 封装后再进入统一 pipeline

5. `runPipeline` 是这个模块内部的统一驱动入口，pass 注册集中发生在这里

### `VPTOFatobjEmission`

1. 负责和工具链、临时文件、最终 fatobj 输出打交道，将 vector、cube、stub 组件组织并产出 fatobj

2. 负责临时文件管理。这里禁止使用“临时目录托管 + 目录递归删除”的模型，而是只管理单个临时文件。原因不是实现 bug，而是目录模型本身具有更高的删除风险：一旦路径判断错误，目录删除天然带有批量删除和隐式路径解释的风险；文件级清理不存在这种大范围破坏面。

3. 将 vector/cube LLVM module 和 stub 字符串按工具链需要写入临时文件。文件落盘是统一主模型。

4. 参考 test/vpto 下的脚本搭建编译流程，并参考 LLVM/Clang driver 的工具链调用模式实现本地封装。这里需要有一组统一接口负责：

- 创建并注册临时文件，便于统一清理
- 调用 `llvm::sys::ExecuteAndWait(...)` 执行外部工具
- 在底层工具支持时，将已经落盘的临时文件内容通过标准输入重定向给子进程
- 在底层工具不支持时，回退为显式临时文件输入

5. 上述封装的目标不是消灭临时文件，而是统一管理“临时文件创建 / 注册 / 重定向 / 清理”，让 toolchain 交互方式稳定收敛。

6. 链接过程整体参考 test/vpto 下的脚本，最终输出为 `-o` 的参数。


## 测试约束

### `test/vpto` 脚本

1. `test/vpto` 中的测试脚本需要统一为使用 `ptoas` 直接吐出的 fatobj。

2. 脚本不再自己分别编译 device llvm、device obj、host stub 再手动打包，而是直接消费 `ptoas` 的 fatobj 输出结果。

3. 脚本中的 mixed / non-mixed、cube / vector、单独 `cube.pto` 等特判路径都需要移除，统一走同一种编译与链接模型。

### `test/vpto` case 组织

1. 每个 case 只保留一个 `kernel.pto`。

2. 原来 `cube.pto` 中的代码需要挪到 `kernel.pto` 里的 cube module 中。

3. `kernel.pto` 中允许同时包含 vector module 和 cube module，并通过 `pto.kernel_kind` 区分。

4. 测试数据生成、host stub、launch、compare 等配套文件继续按现有 case 目录组织保留，不在这次改造中改变其职责。

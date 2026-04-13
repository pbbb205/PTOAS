# TileLang ST 精度验证框架

## 1. 文档目标

本文从 TileLang 库开发者的视角介绍当前 `test/tilelang_st` 框架的使用方式。

这份框架的目标不是做单纯的 IR 回归，而是回答下面两个更贴近开发的问题：

1. 我新写的 TileLang 模板库实现，展开到 PTO / VPTO / LLVM IR 之后，最终在 simulator 或 NPU 上跑出来的数值是否正确。
2. 如果我要为一个新 op 增加 ST 用例，最少需要准备哪些文件，运行链路会经过哪些阶段。

当前框架已经具备下面这些能力：

- 从 `.pto` 直接驱动 `ptoas`，不需要手写 `kernel.cpp`
- 支持在一个 testcase 下放多个 case
- 支持 `sim` / `npu` 两种运行模式
- 支持单 case 过滤
- 支持把输入、golden、output 隔离到 `build/testcase/<testcase>/` 下，避免不同 testcase 之间互相覆盖

## 2. 框架定位

TileLang ST 参考了 `pto-isa` 的 ST 目录组织方式，但编译链路不同。

| 维度 | pto-isa ST | TileLang ST |
|---|---|---|
| kernel 来源 | 手写 `kernel.cpp` | 手写 `.pto`，由 `ptoas` 展开 TileLang DSL 模板 |
| 编译入口 | `bisheng -xcce kernel.cpp` | `ptoas .pto -> .ll`，再 `bisheng -x ir .ll -> device.o` |
| device 对象接入 host | 编译器一步直接生成 fatobj | 先产出 device-only `.o`，再 repack 成 host-linkable fatobj |
| 精度比较 | GTest / C++ 比较逻辑 | `compare.py` + `numpy.allclose` |
| 多 case 组织 | 多个 GTest case | 一个 testcase 下多个 kernel 函数 + host case table |

换句话说，TileLang ST 更适合验证“库模板展开后的端到端运行正确性”，而不是验证某一段单独的 CCE kernel.cpp。

## 3. 当前执行流程

统一入口是：

```bash
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd
```

完整链路如下：

```text
run_st.py
  ├─ set_env_variables()
  │   └─ 配置 simulator / NPU 运行环境
  ├─ build_project()
  │   ├─ cmake -DRUN_MODE=... -DSOC_VERSION=... -DTEST_CASE=... -DPTOAS_BIN=...
  │   ├─ ptoas: <op>.pto -> <op>_kernel.ll
  │   │    flags:
  │   │      --pto-arch=a5
  │   │      --pto-backend=vpto
  │   │      --enable-insert-sync
  │   │      --enable-tile-op-expand
  │   │      --vpto-emit-hivm-llvm
  │   ├─ bisheng -x ir: <op>_kernel.ll -> <op>_kernel_device.o
  │   ├─ repack_tilelang_kernel.sh:
  │   │    <op>_kernel_device.o -> <op>_kernel_repack.o
  │   ├─ bisheng -xcce: launch.cpp + <op>_kernel_repack.o -> lib<op>_kernel.so
  │   └─ bisheng -xc++: main.cpp -> <op>
  ├─ run_gen_data()
  │   └─ 在 build/testcase/<testcase>/ 下生成每个 case 的 input/golden
  ├─ run_binary()
  │   └─ 在 build/testcase/<testcase>/ 下执行 ../../bin/<testcase> [case]
  └─ run_compare()
      └─ 在 build/testcase/<testcase>/ 下逐 case 比较 golden/output
```

### 3.1 关于 repack 步骤

这是当前 TileLang ST 相比 `pto-isa` 手写 kernel.cpp 路径最大的区别。

`ptoas + bisheng -x ir` 产出的 `*_kernel_device.o` 是 device-only 对象，不能直接作为 host 侧共享库链接输入。框架会调用 `test/tilelang_st/npu/a5/src/st/testcase/repack_tilelang_kernel.sh` 做两件事：

1. 从 `launch.cpp` 中抽取 `extern "C" __global__ AICORE void ...` 声明，生成一个最小 stub
2. 通过 `-fcce-include-aibinary <device.o>` 把 device binary 嵌入这个 host stub，最终产出 host 可链接的 fatobj 对象

最终 `launch.cpp` 和 `*_kernel_repack.o` 一起链接成 `lib<op>_kernel.so`。

如果没有这个 repack 步骤，host 可执行文件无法通过共享库把 LLVM IR 编出来的 device kernel 注册并发射出去。

### 3.2 关于 case 的执行和比较顺序

默认情况下：

1. `gen_data.py` 会先为 testcase 下的所有 case 生成输入和 golden
2. `./bin/<testcase>` 会依次跑完所有 case
3. `compare.py` 再依次比较所有 case 的 `golden.bin` 和 `output.bin`

如果使用 `-c <case_name>`，则运行和比较都会只针对这个 case。

## 4. 目录结构与职责

当前目录结构如下：

```text
test/tilelang_st/
├── script/
│   └── run_st.py
└── npu/
    └── a5/
        └── src/st/
            ├── CMakeLists.txt
            └── testcase/
                ├── CMakeLists.txt
                ├── run_ptoas_to_file.cmake
                ├── repack_tilelang_kernel.sh
                ├── st_common.py
                ├── compare.py
                └── tadd/
                    ├── CMakeLists.txt
                    ├── cases.py
                    ├── tadd.pto
                    ├── launch.cpp
                    ├── main.cpp
                    └── gen_data.py
```

各文件职责如下：

| 文件 | 职责 |
|---|---|
| `script/run_st.py` | 统一入口，负责编译、生成数据、执行二进制、比较结果 |
| `src/st/CMakeLists.txt` | 顶层 CMake，设置编译器、环境和依赖 |
| `testcase/CMakeLists.txt` | 定义 `pto_tilelang_vec_st()` 宏，并注册所有 testcase |
| `testcase/run_ptoas_to_file.cmake` | 封装 `ptoas` 调用，把 `.pto` 编译成 LLVM IR |
| `testcase/repack_tilelang_kernel.sh` | 把 device-only `.o` 包装成 host 可链接的 fatobj |
| `testcase/st_common.py` | 所有 testcase 共享的 Python 公共模块（case 校验、数据生成辅助、精度比较、终端着色） |
| `testcase/compare.py` | 公共比较脚本，所有 testcase 共享，从 per-testcase 的 `cases.py` 导入 `CASES` 后调用 `st_common.run_compare()` |
| `testcase/<op>/cases.py` | **case 定义的单一来源**，`gen_data.py` 和 `compare.py` 均从此导入 |
| `testcase/<op>/<op>.pto` | testcase 的 kernel 描述，通常一个文件中放多个 case 对应的函数 |
| `testcase/<op>/launch.cpp` | kernel 声明和 launch wrapper |
| `testcase/<op>/main.cpp` | host driver，负责分配内存、launch kernel、回写 output（`ACL_CHECK` 宏由公共头 `test_common.h` 提供） |
| `testcase/<op>/gen_data.py` | 生成 input 与 golden，从 `cases.py` 读取 case 列表 |

## 5. 日常使用方式

### 5.0 前置条件

运行 TileLang ST 之前，建议先确认下面几件事：

- 仓库里的 `ptoas` 已经编出来，默认路径是 `build/tools/ptoas/ptoas`
- `ASCEND_HOME_PATH` 已经设置正确
- 如果需要手工跑 `ptoas`、`bisheng` 或 lit，优先先执行：

```bash
source scripts/ptoas_env.sh
```

`run_st.py` 会在运行时补充 simulator / NPU 相关环境，但它不会替你构建 `ptoas`。

### 5.1 运行已有 testcase

```bash
# simulator 上跑 tadd 全部 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd

# NPU 上跑 tadd 全部 case
python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd

# 只跑一个 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd -c f32_16x64

# 复用已有 build 目录，不重新编译
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd -w
```

### 5.2 常用参数

| 参数 | 含义 |
|---|---|
| `-r, --run-mode` | 运行模式，`sim` 或 `npu` |
| `-v, --soc-version` | SoC 版本，目前只支持 `a5` |
| `-t, --testcase` | testcase 名称，对应 `testcase/<name>/` |
| `-c, --case` | 只运行一个 case |
| `-p, --ptoas-bin` | 指定 `ptoas` 路径 |
| `-w, --without-build` | 跳过构建，直接复用已有 `build/` |

### 5.3 产物在哪

testcase 的运行时数据不再写到 `build/` 根目录，而是写到：

```text
test/tilelang_st/npu/a5/src/st/build/testcase/<testcase>/
```

以 `tadd` 为例：

```text
build/testcase/tadd/
├── gen_data.py
├── compare.py
├── f32_16x64/
│   ├── input1.bin
│   ├── input2.bin
│   ├── golden.bin
│   └── output.bin
└── f32_32x32/
    ├── input1.bin
    ├── input2.bin
    ├── golden.bin
    └── output.bin
```

这个布局的好处是：

- 不同 testcase 之间不会因为 case 同名而互相覆盖
- 方便开发者直接进入 `build/testcase/<testcase>/` 复查输入、输出和 golden
- 使用 `-w` 时，不容易把旧 testcase 的残留数据误认为当前结果

### 5.4 比较输出

`compare.py` 会对 pass/fail 做明显提示：

- pass：粗体绿色
- fail：粗体红色

比较逻辑目前使用 `numpy.allclose`。建议阈值：

| dtype | 建议 eps |
|---|---|
| `float32` | `1e-6` |
| `float16` | `1e-3` |
| `bfloat16` | `1e-2` |
| `int8/int16/int32` | `0` |

## 6. 作为库开发者，如何增加一个新 op testcase

这一节回答“我开发了一个新的 TileLang 库实现，怎么用 ST 框架验证它”。

以新增 `pto.tsub` 为例，最少需要准备下面这些文件：

| 文件 | 是否新增/修改 | 说明 |
|---|---|---|
| `testcase/tsub/CMakeLists.txt` | 新增 | 一般只有一行 `pto_tilelang_vec_st(tsub)` |
| `testcase/tsub/cases.py` | 新增 | **case 定义的单一来源**：每个 case 必须指定 `name`/`dtype`/`shape`/`valid_shape`/`eps` |
| `testcase/tsub/tsub.pto` | 新增 | 定义一个或多个 case 的 kernel 函数 |
| `testcase/tsub/launch.cpp` | 新增 | 为每个 kernel 函数声明 entry 并提供 launch wrapper |
| `testcase/tsub/main.cpp` | 新增 | host driver，负责 case table、内存拷贝、launch 和 output 落盘 |
| `testcase/tsub/gen_data.py` | 新增 | 生成每个 case 的输入和 golden，从 `cases.py` 导入 `CASES` |
| `testcase/CMakeLists.txt` | 修改 | 把 `tsub` 加入 `ALL_TESTCASES` |

通常不需要修改：

- `script/run_st.py`
- `src/st/CMakeLists.txt`
- `testcase/st_common.py`
- `testcase/compare.py`（公共脚本，所有 testcase 共享）
- `testcase/run_ptoas_to_file.cmake`
- `testcase/repack_tilelang_kernel.sh`

除非你在改框架本身，而不是新增一个 testcase。

## 7. 以 `pto.tadd` 为例，需要改哪些文件

当前仓库里 `tadd` 已经是一个完整样例。把它当成模板即可。

### 7.1 `testcase/tadd/CMakeLists.txt`

这个文件通常最简单：

```cmake
pto_tilelang_vec_st(tadd)
```

含义是让公共宏接管 `tadd.pto -> tadd_kernel.ll -> tadd_kernel_device.o -> tadd_kernel_repack.o -> libtadd_kernel.so -> tadd` 这一整条流水线。

### 7.2 `testcase/tadd/tadd.pto`

这是最核心的文件。你需要在这里写出要验证的 kernel 形态。

当前 `tadd.pto` 的特点是：

- 一个文件中包含多个 case
- 每个 case 对应一个 `func.func @TADD_<dtype>_<rows>x<cols>(...)`
- 函数体里显式写出 `make_tensor_view`、`partition_view`、`alloc_tile`、`tload`、`pto.tadd`、`tstore`

如果你在开发 `pto.tadd` 库实现，最关键的是先把你要覆盖的 case 设计好。例如：

- `f32` / `f16` / `bf16`
- 不同 tile 形状
- 边界 valid 行列不是整 tile 的情况

这里的函数命名建议统一成：

```text
TADD_<dtype>_<rows>x<cols>
```

例如：

```text
TADD_f32_16x64
TADD_f32_32x32
```

### 7.3 `testcase/tadd/launch.cpp`

这个文件的职责只有两个：

1. 声明 kernel entry
2. 为 host driver 提供 `Launch*` wrapper

当前推荐写法和 `tadd` 一致：

```cpp
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TADD_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTADD_f32_16x64(float *a, float *b, float *c, void *stream) {
    TADD_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
```

注意点：

- `launch.cpp` 不需要包含 PTO 头文件
- `AICORE` 直接本地定义为 `[aicore]`
- 这里的 kernel 声明会被 repack 脚本抽取出来生成 stub，所以必须保留 `extern "C" __global__ AICORE void ...` 这一形态
- kernel 参数顺序必须和 `.pto` 中函数签名保持一致

### 7.4 `testcase/tadd/main.cpp`

这个文件负责 host 侧调度。

你需要做的事主要有三类：

1. 声明所有 `LaunchTADD_*` wrapper
2. 在 `kCases[]` 中列出每个 case 的名字、launch 函数、shape、valid shape、元素大小
3. 在 `RunCase()` 中完成：
   - 从 `./<case>/input*.bin` 读取输入
   - `aclrtMemcpy` 把输入拷到 device
   - 调用 `tc.launch(...)`
   - `aclrtSynchronizeStream`
   - 把输出拷回 host
   - 写 `./<case>/output.bin`

当前 `tadd/main.cpp` 的 case table 形式如下：

```cpp
struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"f32_16x64", LaunchTADD_f32_16x64, 16, 64, 16, 64, sizeof(float)},
    {"f32_32x32", LaunchTADD_f32_32x32, 32, 32, 32, 32, sizeof(float)},
};
```

注意：`ACL_CHECK` 宏已移至公共头文件 `test_common.h`（需在 `acl/acl.h` 之后包含），不需要在每个 testcase 的 `main.cpp` 中重复定义。

你在新增 case 时，必须同步更新这个表，字段需与 `cases.py` 中的 `shape` / `valid_shape` 保持一致。

### 7.5 `testcase/tadd/cases.py`

这是 case 定义的**单一来源**，`gen_data.py` 和 `compare.py` 均从此导入 `CASES`。

每个 case 必须包含以下字段：

```python
CASES = [
    {
        "name": "f32_16x64",          # case 标识，对应运行时子目录和 main.cpp kCases[] 中的 name
        "dtype": np.float32,           # numpy dtype
        "shape": (16, 64),             # 分配的 tile 维度 (rows, cols)
        "valid_shape": (16, 64),       # 有效计算区域 (valid_rows, valid_cols)
        "eps": 1e-6,                   # numpy.allclose 容差
    },
]
```

`valid_shape` 为必填字段。当 valid shape 等于 tile shape 时也必须显式写出。

### 7.6 `testcase/tadd/gen_data.py`

这个文件负责为每个 case 生成输入和 golden。从 `cases.py` 导入 `CASES`，
从 `st_common.py` 导入辅助函数（`setup_case_rng`、`save_case_data`）。

以 `pto.tadd` 为例，每个 case 的核心逻辑：

```python
golden = np.zeros(shape, dtype=dtype)
vr, vc = case["valid_shape"]
golden[:vr, :vc] = (input1[:vr, :vc] + input2[:vr, :vc]).astype(dtype, copy=False)
```

golden 只在 `valid_shape` 区域内计算，区域外保持零值。

每个 case 使用独立的随机 seed（`setup_case_rng` 基于 `hash(case["name"])`），
新增或调整 case 顺序不会影响已有 case 的测试数据。

### 7.7 `testcase/compare.py`（公共，无需 per-testcase 修改）

`compare.py` 位于 `testcase/` 公共目录，所有 testcase 共享同一份：

```python
from cases import CASES
from st_common import run_compare

if __name__ == "__main__":
    run_compare(CASES)
```

`run_st.py` 运行时会将它和 per-testcase 的 `cases.py` 一起拷贝到 build 目录，
`compare.py` 通过 `from cases import CASES` 获取当前 testcase 的 case 列表。

`run_compare()` 会：
- 校验所有 case 必填字段
- 只在 `valid_shape` 区域内比较 `golden.bin` 与 `output.bin`
- 支持 `argv[1]` 作为 case filter
- exit code 2 表示失败

## 8. 如果只是在已有 `tadd` 下新增一个 case

如果 `tadd` testcase 已经存在，而你只是想加一个新 case，例如 `f32_8x128`，则通常只需要同步修改 4 个文件：

| 文件 | 必须修改的内容 |
|---|---|
| `testcase/tadd/cases.py` | 在 `CASES` 中加入新条目（含 `name`/`dtype`/`shape`/`valid_shape`/`eps`） |
| `testcase/tadd/tadd.pto` | 新增一个 `func.func @TADD_f32_8x128(...)` |
| `testcase/tadd/launch.cpp` | 新增 `extern "C"` kernel 声明和 `LaunchTADD_f32_8x128` |
| `testcase/tadd/main.cpp` | 在 `kCases[]` 中加入 `{"f32_8x128", LaunchTADD_f32_8x128, 8, 128, 8, 128, sizeof(float)}` |

不需要改：

- `testcase/tadd/gen_data.py`（自动从 `cases.py` 读取）
- `testcase/tadd/compare.py`（自动从 `cases.py` 读取）
- `testcase/tadd/CMakeLists.txt`
- `testcase/CMakeLists.txt`
- `run_st.py`

## 9. 文件之间必须保持一致的约束

这是新增 testcase 时最容易出错的地方。

### 9.1 命名一致

下面这几处名字必须严格一致：

| 位置 | 示例 |
|---|---|
| `.pto` 中的 kernel 函数名 | `@TADD_f32_16x64` |
| `launch.cpp` 中的 kernel 声明 | `TADD_f32_16x64` |
| `launch.cpp` / `main.cpp` 中的 wrapper 名 | `LaunchTADD_f32_16x64` |
| `main.cpp` 的 case 名 | `f32_16x64` |
| `gen_data.py` / `compare.py` 的 case 名 | `f32_16x64` |
| 运行时目录名 | `build/testcase/tadd/f32_16x64/` |

### 9.2 参数顺序一致

`.pto` 里 kernel 的参数顺序、`launch.cpp` 声明顺序、`main.cpp` 里 launch wrapper 的参数顺序必须一致。  
如果 `tadd` 的语义是 `(a, b) -> c`，那 host 侧和 compare 也都要按这个顺序组织。

### 9.3 shape、valid_shape 和 dtype 一致

`cases.py` 中的 `shape`/`valid_shape`/`dtype` 是 Python 侧的单一来源，`gen_data.py` 和 `compare.py` 自动从中读取。
但 C++ 侧的 `main.cpp` `kCases[]`（`rows`/`cols`/`validRows`/`validCols`/`elemSize`）和 `.pto` 中的 tile shape 仍需手动与 `cases.py` 保持一致。
否则运行能成功，结果也可能是错误的，且定位会很耗时。

## 10. 建议的开发验证节奏

作为库开发者，建议用下面的节奏迭代：

1. 先写一个最小 case，例如 `f32_16x64`
2. 在 simulator 上跑单 case：

```bash
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd -c f32_16x64
```

3. 改 `.pto` 或 host 代码后，如果确认只是小修改，可以用：

```bash
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd -c f32_16x64 -w
```

4. 单 case 稳定后，再补更多 shape / dtype case
5. 再跑全量 `tadd`
6. 最后如果需要，再切到 `-r npu`

## 11. 调试建议

### 11.1 编译失败看哪里

- `ptoas` 失败：优先看 `.pto` 本身、TileLang 模板实例化、是否缺少 `--enable-insert-sync`
- `bisheng -x ir` 失败：优先看生成的 `*_kernel.ll`
- repack 失败：优先看 `launch.cpp` 中的 kernel 声明是否符合脚本预期
- `launch.cpp` / `main.cpp` 链接失败：优先看共享库、ACL 运行时依赖和符号名一致性

### 11.2 运行失败看哪里

- `main.cpp` 报读文件失败：先确认 `build/testcase/<testcase>/<case>/input*.bin` 是否存在
- kernel 能跑但 compare fail：先看 `output.bin` 与 `golden.bin` 的差异，再看 `.pto` 语义和 host 参数顺序
- 某个 case 单独跑通过、全量跑失败：优先怀疑 case 目录隔离、host 资源释放、或者多 case 共用状态

### 11.3 典型排查文件

| 文件 | 作用 |
|---|---|
| `build/testcase/<testcase>/<testcase>_kernel.ll` | 看 `ptoas` 最终生成的 LLVM IR |
| `build/testcase/<testcase>/<case>/golden.bin` | 确认 Python 侧 oracle 是否正确 |
| `build/testcase/<testcase>/<case>/output.bin` | 确认运行时实际输出 |
| `testcase/<op>/main.cpp` | 确认 host 侧参数顺序、shape 和文件路径 |
| `testcase/<op>/compare.py` | 确认比较阈值是否合理 |

## 12. 一句话总结

对于库开发者来说，TileLang ST 框架就是一条固定好的端到端验证流水线：

```text
写 .pto -> 接入 testcase 六件套 -> run_st.py 编译运行 -> 查看 build/testcase/<op>/ 下的 input/golden/output -> 判断库实现是否正确
```

如果你想验证的是 `pto.tadd`，最重要的是把下面几处保持同步：

- `cases.py` 中的 case 定义（name/dtype/shape/valid_shape/eps）—— Python 侧的单一来源
- `tadd.pto` 中的 kernel 函数名和 tile shape
- `launch.cpp` 中的 kernel 声明与 wrapper
- `main.cpp` 中的 `kCases[]`（rows/cols/validRows/validCols 需与 `cases.py` 一致）
- `gen_data.py` 中的 golden 计算逻辑（op 语义相关，如加法/减法）

`compare.py` 和 `gen_data.py` 的 case 列表、比较阈值均自动从 `cases.py` 读取，不需要单独维护。

这几处一致，框架就能帮助你把 TileLang 库实现的”端到端正确性”稳定地跑起来。

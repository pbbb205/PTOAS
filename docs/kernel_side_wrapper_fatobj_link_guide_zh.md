# Kernel-side wrapper 链接 VPTO fatobj 指导

本文记录一个可复现的最小验证：kernel-side C++ wrapper / caller 与 separate compiled vector、cube callee object 通过 `--cce-fatobj-link -r` 合并成 `bundle.o`，再与 host `main.o` 链接并运行。

这个实验用于指导后续 pypto kernel-side C++ wrapper 调用 PTOAS-vpto 生成 fatobj 的 ABI 设计。实验里的手写 `__asm__("xxx.vector")` / `__asm__("xxx.cube")` 不是最终用户接口，而是用来模拟 PTOAS-vpto fatobj 未来需要导出的 device direct-call 符号。

## 验证目标

按本文创建 `vec_callee.cpp`、`cube_callee.cpp`、`caller.cpp` 和 `main.cpp` 后，期望运行结果为：

```text
rdc_cv_sectioned_real_demo PASS
```

验证覆盖：

- `vec_callee.o`：真实 vector `DataCopy + Add + DataCopy`
- `cube_callee.o`：真实 C310 cube `LoadData + Mmad + Fixpipe`
- `caller.o`：`__global__ [aicore]` launch target，按 `AIV/AIC` 显式分支调用 callee
- `bundle.o`：`caller.o + vec_callee.o + cube_callee.o` 经 `--cce-fatobj-link -r` 合并
- `main.o + bundle.o`：host 侧最终链接并运行

## 核心结论

kernel-side wrapper 链接 separate compiled device callee 是可行的，必要条件是 caller 需要的符号和 callee 提供的符号一致。

sectioned caller 在 AIV 分支里调用 vector callee，在 AIC 分支里调用 cube callee：

```cpp
if ASCEND_IS_AIV {
    vec_callee(a, b, tmp);
}
if ASCEND_IS_AIC {
    cube_callee(out, cubeA, cubeB);
}
```

`bisheng -dc` 编译 caller 后，device linker 需要的符号是：

```text
vec_callee.vector
cube_callee.cube
```

因此，被链接进来的 callee object / fatobj 必须提供同名符号：

```text
vec_callee.vector
cube_callee.cube
```

对 PTOAS-vpto 的含义是：

```text
vector direct-call callee: foo.vector
cube direct-call callee:   foo.cube
```

如果 vpto 当前只导出 `foo`、`foo_mix_aiv`、`foo_mix_aic` 或 `foo.vector.thread`，还不能满足 kernel-side wrapper direct-call ABI。

## 代码结构

### `vec_callee.cpp`

vector callee 显式导出 `vec_callee.vector`。这里的 `__asm__` 只是手写 demo 中模拟目标 ABI 的方式。

```cpp
#include "kernel_operator.h"

extern "C" __attribute__((used, noinline)) __aicore__ void vec_callee_vector(
    __gm__ float *a, __gm__ float *b, __gm__ float *tmp) __asm__("vec_callee.vector");

extern "C" __attribute__((used, noinline)) __aicore__ void vec_callee_vector(
    __gm__ float *a, __gm__ float *b, __gm__ float *tmp)
{
    constexpr uint32_t kCount = 1024;
    constexpr uint16_t kBlockLen = kCount * sizeof(float) / 32;

    AscendC::GlobalTensor<float> gmA;
    AscendC::GlobalTensor<float> gmB;
    AscendC::GlobalTensor<float> gmTmp;
    gmA.SetGlobalBuffer(a, kCount);
    gmB.SetGlobalBuffer(b, kCount);
    gmTmp.SetGlobalBuffer(tmp, kCount);

    AscendC::LocalTensor<float> ubA(AscendC::TPosition::VECCALC, 0, kCount);
    AscendC::LocalTensor<float> ubB(AscendC::TPosition::VECCALC, 4096, kCount);
    AscendC::LocalTensor<float> ubOut(AscendC::TPosition::VECCALC, 8192, kCount);
    AscendC::DataCopyParams copyParams{1, kBlockLen, 0, 0};

    AscendC::DataCopy(ubA, gmA, copyParams);
    AscendC::DataCopy(ubB, gmB, copyParams);
    AscendC::Add(ubOut, ubA, ubB, static_cast<int32_t>(kCount));
    AscendC::DataCopy(gmTmp, ubOut, copyParams);
}
```

### `cube_callee.cpp`

cube callee 显式导出 `cube_callee.cube`。本实验使用 C310 MX fp8 路径，host 传入 `fp8_e5m2` 原始字节矩阵，cube 侧执行 `8x16 * 16x8 -> 8x8`。

```cpp
#include "kernel_operator.h"

extern "C" __attribute__((used, noinline)) __aicore__ void cube_callee_cube(
    __gm__ float *out, __gm__ uint8_t *cubeA, __gm__ uint8_t *cubeB) __asm__("cube_callee.cube");

extern "C" __attribute__((used, noinline)) __aicore__ void cube_callee_cube(
    __gm__ float *out, __gm__ uint8_t *cubeA, __gm__ uint8_t *cubeB)
{
    constexpr uint32_t kElems = 1024;
    constexpr uint32_t kCubeElems = 256;

    AscendC::GlobalTensor<float> gmOut;
    gmOut.SetGlobalBuffer(out, kElems);
    AscendC::GlobalTensor<fp8_e5m2_t> gmA;
    AscendC::GlobalTensor<fp8_e5m2_t> gmB;
    gmA.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e5m2_t *>(cubeA), kCubeElems);
    gmB.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e5m2_t *>(cubeB), kCubeElems);

    AscendC::LocalTensor<fp8_e5m2_t> l1a(AscendC::TPosition::C1, 0, kCubeElems);
    AscendC::LocalTensor<fp8_e5m2_t> l1b(AscendC::TPosition::C1, 4096, kCubeElems);
    AscendC::LocalTensor<fp8_e5m2_t> l0a(AscendC::TPosition::A2, 0, kCubeElems);
    AscendC::LocalTensor<fp8_e5m2_t> l0b(AscendC::TPosition::B2, 0, kCubeElems);
    AscendC::LocalTensor<float> l0c(AscendC::TPosition::CO1, 0, 256);

    AscendC::LoadData2DParamsV2 gmToL1{0, 0, 1, 1, 16, 1, false, 0};
    AscendC::LoadData(l1a, gmA, gmToL1);
    AscendC::LoadData(l1b, gmB, gmToL1);

    AscendC::LoadData2DParamsV2 l1ToL0{0, 0, 1, 1, 16, 1, false, 0};
    AscendC::LoadData(l0a, l1a, l1ToL0);
    AscendC::LoadData(l0b, l1b, l1ToL0);

    AscendC::MmadBitModeParams params;
    params.SetM(8);
    params.SetN(8);
    params.SetK(16);
    params.SetUnitFlag(0);
    params.SetDisableGemv(false);
    params.SetCmatrixSource(false);
    params.SetCmatrixInitVal(true);
    AscendC::Mmad(l0c, l0a, l0b, params);

    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams{8, 8, 16, 16};
    AscendC::Fixpipe(gmOut, l0c, fixParams);
}
```

### `caller.cpp`

caller 是 host 可 launch 的 kernel entry。它在同一个 source 中声明 logical callee 名称 `vec_callee` / `cube_callee`，`bisheng -dc` 会在 AIV/AIC specialization 中分别引用 `.vector` / `.cube`。

```cpp
#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

namespace AscendC {
enum CoreType {
    AIC = 0,
    AIV = 1,
    MIX = 2,
};
}

#if defined(__DAV_CUBE__)
constexpr int g_coreType = AscendC::AIC;
#elif defined(__DAV_VEC__)
constexpr int g_coreType = AscendC::AIV;
#else
constexpr int g_coreType = AscendC::MIX;
#endif

#define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC constexpr(g_coreType == AscendC::AIC)

extern "C" __aicore__ void vec_callee(__gm__ float *a, __gm__ float *b, __gm__ float *tmp);
extern "C" __aicore__ void cube_callee(__gm__ float *out, __gm__ unsigned char *cubeA, __gm__ unsigned char *cubeB);

extern "C" __global__ [aicore] void cv_sectioned_real_launch(
    __gm__ float *a, __gm__ float *b, __gm__ float *tmp, __gm__ float *out,
    __gm__ unsigned char *cubeA, __gm__ unsigned char *cubeB)
{
    if ASCEND_IS_AIV {
        vec_callee(a, b, tmp);
    }
    if ASCEND_IS_AIC {
        cube_callee(out, cubeA, cubeB);
    }
}
```

## 编译和链接

创建并进入一个空的工作目录，然后设置 CANN 安装路径：

```bash
mkdir -p rdc_cv_sectioned_real_demo
cd rdc_cv_sectioned_real_demo

# 按实际环境修改。本文后续命令只使用该变量，不依赖固定安装路径。
export ASCEND_HOME_PATH=<path-to-cann>
```

公共 include 路径：

```bash
INC="-I${ASCEND_HOME_PATH}/aarch64-linux/asc \
-I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api \
-I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api \
-I${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api \
-I${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface \
-I${ASCEND_HOME_PATH}/aarch64-linux/asc/include \
-I${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api \
-I${ASCEND_HOME_PATH}/aarch64-linux/asc/impl"
```

编译 vector callee：

```bash
bisheng -dc -fPIC -xcce --cce-aicore-arch=dav-c310-vec \
  -std=c++17 -DTILING_KEY_VAR=0 ${INC} \
  vec_callee.cpp -o vec_callee.o
```

编译 cube callee：

```bash
bisheng -dc -fPIC -xcce --cce-aicore-arch=dav-c310-cube \
  -std=c++17 -DTILING_KEY_VAR=0 ${INC} \
  cube_callee.cpp -o cube_callee.o
```

编译 mixed caller：

```bash
bisheng -dc -fPIC -xcce --cce-aicore-arch=dav-c310 \
  -std=c++17 caller.cpp -o caller.o
```

生成 `bundle.o`：

```bash
bisheng -fPIC --cce-fatobj-link -r \
  -o bundle.o caller.o vec_callee.o cube_callee.o
```

编译 host launch：

```bash
bisheng -c -fPIC -xcce --cce-aicore-arch=dav-c310 -std=c++17 \
  -I ${ASCEND_HOME_PATH}/include \
  -I ${ASCEND_HOME_PATH}/pkg_inc \
  -I ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime \
  main.cpp -o main.o
```

最终 host link：

```bash
g++ main.o bundle.o -o rdc_cv_sectioned_real_demo \
  -L ${ASCEND_HOME_PATH}/lib64 \
  -Wl,-rpath,${ASCEND_HOME_PATH}/lib64 \
  -lprofapi -lruntime -lascendcl -ltiling_api -lplatform -lc_sec -lnnopbase \
  -lstdc++ -ldl -lpthread -lm
```

运行：

```bash
./rdc_cv_sectioned_real_demo
```

期望输出：

```text
rdc_cv_sectioned_real_demo PASS
```

## 编译选项说明

`-dc`：

用于生成可参与 device separate compilation 的 CCE object。caller 和 callee 都需要用该模式编译，否则后续 device-side direct-call link 不可靠。

`--cce-fatobj-link -r`：

对多个 CCE device object 做 relocatable fatobj link，生成 host link 可消费的 `bundle.o`。本实验中它把 `caller.o`、`vec_callee.o`、`cube_callee.o` 合并。

`--cce-aicore-arch=dav-c310-vec`：

只编译 vector callee 的 AIV 版本，提供 `vec_callee.vector`。

`--cce-aicore-arch=dav-c310-cube`：

只编译 cube callee 的 AIC 版本，提供 `cube_callee.cube`。

`--cce-aicore-arch=dav-c310`：

编译 mixed caller，生成 AIV/AIC 两个 specialization。caller 在 AIV 侧引用 `.vector`，在 AIC 侧引用 `.cube`。

`-DTILING_KEY_VAR=0`：

`kernel_operator.h` 会在未定义 `TILING_KEY_VAR` 时定义全局 `g_tilingKey`。当多个 TU separate compile 且都 include `kernel_operator.h` 时，fatobj link 可能报：

```text
ld.lld: error: duplicate symbol: g_tilingKey
>>> defined at vec_callee.cpp
>>> defined at cube_callee.cpp
```

本实验不使用 tiling key，因此用 `-DTILING_KEY_VAR=0` 禁止每个 TU 生成重复定义。

## 符号检查

检查 caller 需要的符号：

```bash
strings caller.o | grep -E 'vec_callee|cube_callee|\.vector|\.cube' | sort -u
```

期望包含：

```text
vec_callee.vector
cube_callee.cube
```

检查 callee 提供的符号：

```bash
strings vec_callee.o | grep vec_callee
strings cube_callee.o | grep cube_callee
```

期望包含：

```text
vec_callee.vector
cube_callee.cube
```

如果 callee 没有提供对应符号，`--cce-fatobj-link -r` 会失败。例如缺 vector callee 时：

```text
ld.lld: error: undefined symbol: vec_callee.vector
>>> referenced by caller.cpp
>>> /tmp/caller.o-...extract:(cv_sectioned_real_launch_mix_aiv)
```

缺 cube callee 时：

```text
ld.lld: error: undefined symbol: cube_callee.cube
>>> referenced by caller.cpp
>>> /tmp/caller.o-...extract:(cv_sectioned_real_launch_mix_aic)
```

## 对 PTOAS-vpto 的指导

pypto kernel-side C++ wrapper 链接 PTOAS-vpto fatobj 时，应复用本实验验证过的 device link 形态：

```text
wrapper/caller.o + vpto_generated_callee.fatobj.o
  --cce-fatobj-link -r
  -> bundle.o
```

但 vpto fatobj 必须导出 caller 需要的 direct-call callee 符号：

```text
foo.vector
foo.cube
```

当前已经观察到的 vpto launch-style 符号不等价：

```text
foo
foo_mix_aiv
foo_mix_aic
foo.vector.thread
```

其中：

- `foo_mix_aiv` / `foo_mix_aic` 是 launch-style specialization 名称
- `foo.vector.thread` 是 `pto.vecscope` lowering 后的内部 vector thread，不是 wrapper direct-call ABI
- `foo.vector` / `foo.cube` 才是 separate compilation device linker 需要的 callee 符号

因此，后续 PTOAS-vpto 如果要支持 pypto kernel-side wrapper direct-call，应增加 direct-call fatobj emission 模式，至少满足：

```text
vector kind: export foo.vector
cube kind:   export foo.cube
```

如果 kernel 只有 vector 实现，wrapper 也应只在 AIV 分支调用该 callee，避免 AIC 分支额外要求 `foo.cube`。

## 已知限制

本实验验证的是 compile/link/runtime 路径，不验证复杂 AIC/AIV 跨核同步协议。vector callee 写 `tmp`，cube callee 写 `out`，两边没有数据依赖，因此不需要额外 cross-core sync。

尝试在 cube callee 内用 `Fill` 初始化 `fp8_e5m2_t` L0A/L0B 时，Bisheng frontend 曾出现 `stack smashing detected` crash。本实验改为从 host 传入 `fp8_e5m2` 原始字节矩阵。

当前 cube 数值 case 是 `8x8` 输出 tile。如果要扩大到 `16x16` 或更复杂 layout，需要继续确认 C310 MX fp8 的 `M/N/Fixpipe` 参数。

# ptodsl — PTO Python IR Builders

A lightweight, pip-installable DSL package for building PTO MLIR IR modules
in Python.  The API is inspired by Triton / CuteDSL: kernels are ordinary
Python functions decorated with `@pto.to_ir`, type annotations carry PTO
types as lazy descriptors, and control-flow maps 1-to-1 to MLIR operations.

---

## Directory layout

```
ptodsl/
├── ptodsl/              # pip-installable package
│   ├── __init__.py      # exports: pto, scalar
│   ├── pto.py           # main pto.* namespace
│   ├── scalar.py        # pto.scalar.* arith helpers
│   ├── _bootstrap.py    # MLIR path setup + context factory
│   ├── _types.py        # lazy dtype descriptors and type constructors
│   ├── _ops.py          # PTO operation wrappers
│   ├── _control_flow.py # vecscope, for_, if_, yield_ context managers
│   └── _module.py       # @pto.to_ir decorator + module builders
├── examples/
│   ├── tadd_lowlevel.py    # TADD – raw MLIR Python binding calls
│   ├── tadd_dsl.py         # TADD – @pto.to_ir DSL style
│   ├── softmax_lowlevel.py # Softmax – raw MLIR Python binding calls
│   └── softmax_dsl.py      # Softmax – @pto.to_ir DSL style
├── pyproject.toml       # pip install -e .
├── check_ir.py          # IR correctness test runner
└── README.md
```

---

## Prerequisites

```bash
# Install ptoas (first time only)
cd $PTOAS_REPO_ROOT          # e.g. export PTOAS_REPO_ROOT=/workdir/ptoas_a5
bash quick_install.sh

# Set up environment in every new shell
source set_ptoas_env.sh
```

---

## Install the package

```bash
cd $PTOAS_REPO_ROOT/ptodsl
pip install -e .
```

---

## Running the IR check

```bash
# From $PTOAS_REPO_ROOT/ptodsl/
python3 check_ir.py

# From the repository root ($PTOAS_REPO_ROOT)
python3 ptodsl/check_ir.py
```

Expected output:

```
ptodsl IR check
==================================================
  PASS  TADD  low-level
  PASS  TADD  dsl-style
  PASS  softmax low-level
  PASS  softmax dsl-style
==================================================
Result: ALL PASS
```

Exit code is `0` on full pass, `1` on any failure.  A unified diff of up to
60 diverging lines is printed for each failing case.

---

## DSL-style API quick reference

```python
from ptodsl import pto
s = pto.scalar   # arith shorthand alias
```

### Kernel decorator

```python
@pto.to_ir(name="MyKernel", kernel_kind="vector", arch="a5")
def MyKernel():
    ...

@pto.to_ir(name="Softmax", kernel_kind="vector", arch="a5", func_attr="pto.aicore")
def Softmax(arg0: pto.ptr(pto.float32, "gm"), n: pto.int32):
    ...

print(MyKernel)          # prints MLIR text
mod = MyKernel.build()   # returns mlir.ir.Module
```

`func_attr="pto.aicore"` selects a flat single-module structure with the
`pto.aicore` function attribute (softmax style).  Without it, a nested
double-module is emitted (TADD style).

### Type descriptors (lazy – safe to use in annotations)

| Expression | MLIR type |
|---|---|
| `pto.float32` | `f32` |
| `pto.int32` | `i32` |
| `pto.int64` | `i64` |
| `pto.index` | `index` |
| `pto.ptr(pto.float32, "gm")` | `!pto.ptr<f32, gm>` |
| `pto.ptr(pto.float32, "ub")` | `!pto.ptr<f32, ub>` |

### Type constructors (eager – require active context)

```python
vf32     = pto.vreg_type(64, pto.float32)   # !pto.vreg<64xf32>
tile_col = pto.tile_buf_type([8,1], pto.float32, [-1,1], blayout="ColMajor")
tile_w   = pto.tile_buf_type([8,128], pto.float32, [-1,-1])
```

### Constants

```python
c0     = pto.const(0)               # index
c1_i32 = pto.const(1, dtype=pto.int32)
c64_i64= pto.const(64, dtype=pto.int64)
```

### Control flow

```python
with pto.vecscope():                # pto.vecscope { … }
    ...

with pto.for_(c0, c16, step=c1) as i:     # simple scf.for
    ...                                    # scf.yield inserted automatically

with pto.for_(c0, c128, step=c64, iter_args=(a, b)) as loop:
    x, y = loop.iter_args
    ...
    pto.yield_(nx, ny)             # scf.yield with values
fx, fy = loop.results

with pto.if_(has_rows):            # simple scf.if
    ...                             # scf.yield inserted automatically

with pto.if_(has_chunk, results=(vf32, vf32)) as br:
    with br.then_:
        ...
        pto.yield_(merged_max, merged_sum)
    with br.else_:
        pto.yield_(running_max, running_sum)
x, y = br.results
```

### Scalar arithmetic (`s = pto.scalar`)

```python
s.muli(a, b)                 # arith.muli
s.addi(a, b)                 # arith.addi
s.subi(a, b)                 # arith.subi
s.index_cast(val)            # arith.index_cast → index
s.index_cast(pto.int32, val) # arith.index_cast → i32
s.cmpi_sgt(a, b)             # arith.cmpi sgt
s.cmpi("slt", a, b)          # arith.cmpi with named predicate
s.select(cond, t, f)         # arith.select
```

### PTO operations

```python
pto.castptr(addr, ptr_type)              # pto.castptr
pto.addptr(ptr, offset)                  # pto.addptr
pto.vlds(ptr, offset, vreg_type)         # pto.vlds
pto.vbrc_load(ptr, offset, vreg_type)    # pto.vlds {dist="BRC_B32"}
pto.vsts(v, ptr, offset, mask)           # pto.vsts
pto.vsts_1pt(v, ptr, offset, mask)       # pto.vsts {dist="1PT_B32"}
pto.plt_b32(scalar)                      # → (mask, scalar_out)
pto.pset_b32("PAT_ALL")                  # pto.pset_b32 → mask
pto.vadd(a, b, mask)   # infers result type from a.type
pto.vmul / vmax / vdiv / vcmax / vcadd / vdup / vexpdif  # similarly
pto.make_tensor_view(ptr, shape=…, strides=…)    # type inferred
pto.partition_view(tv, offsets=…, sizes=…)        # type inferred
pto.alloc_tile(tile_type, addr=…, valid_row=…, valid_col=…)
pto.tload(part, tile)
pto.tstore(tile, part)
pto.tile_ptr(tile, ptr_type)
pto.get_block_idx()           # → i64
pto.set_flag("MTE2", "V", event_id=0)
pto.wait_flag("MTE2", "V", event_id=0)
pto.barrier_all()
```

### Experimental `vpto` POC

For early experiments around AST-free tracing of TileLang-style tile templates,
`ptodsl` also exposes an experimental namespace:

```python
from ptodsl import vpto as pto

@pto.vkernel(target="a5", op="pto.tadd")
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    with pto.vecscope():
        with pto.for_(0, valid_rows, step=1) as row:
            remained0 = pto.scalar_const(64, pto.i32)
            with pto.for_(0, valid_cols, step=pto.get_lanes(dtype), state={"remained": remained0}) as loop:
                col = loop.iv
                remained = loop.state.remained
                mask, next_remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                out = pto.vadd(lhs, rhs, mask)
                pto.vsts(out, dst[row, col:], mask)
                loop.yield_state(remained=next_remained)
```

Current limitations:

- pybinding-backed POC only; it still covers a narrow TileLang-shaped subset
- supports only static 2D `Tile` parameters
- supports only a narrow vector subset needed by `tadd_template.py`
- requires explicit builder-style `vecscope()` / `for_()` rather than Python `for range(...)`

Reference script:

```bash
python3 lib/TileOps/tadd_template_tracing_poc.py
```

---

## How the IR check works

```
generated IR  ──┐
                ├── Module.parse() → canonical string ──── == ──── PASS/FAIL
reference .pto ──┘  (strips comments, normalises SSA names and attr order)
```

Constant declaration order is preserved after the round-trip; builders must
emit constants in the same order as the reference.  The diff output makes any
mismatch immediately visible.

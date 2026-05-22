# 3. Kernel Entry and Sub-Kernels

PTODSL provides one host-visible kernel decorator (`@pto.jit`) and three
compute-unit sub-kernel decorators (`@pto.cube`, `@pto.simd`, `@pto.simt`),
plus matching context managers for inline use. This chapter covers the kernel
entry, the two programming models, sub-kernel reference, parameter contracts,
and boundary constraints.

## 3.1 `@pto.jit` — the only kernel entry

Decorator overview:

```text
@pto.jit(mode="auto")       tile-first authoring, compiler-managed staging
@pto.jit(mode="explicit")   micro-instruction authoring, user-managed staging
@pto.cube                   Cube-unit matrix sub-kernel
@pto.simd                   SIMD-unit vector sub-kernel
@pto.simt                   SIMT-unit scalar sub-kernel
```

### Role

`@pto.jit` marks a function as a launchable PTO kernel. It owns compilation
(tracing + lowering), caching, and runtime launch binding. This is the only
decorator that can be invoked directly from the host; the compute-unit
decorators define sub-kernels that are called from within `@pto.jit`.

### Signature

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel_name","compile":{"CONST_A":128,"CONST_B":64}} -->
```python
@pto.jit(target="a5", mode="auto")
def kernel_name(
    tensor_arg_1: pto.tensor_spec(rank=1, dtype=pto.f32),  # Python-native tensor (positional)
    tensor_arg_2: pto.tensor_spec(rank=1, dtype=pto.f32),  # Python-native tensor (positional)
    *,
    CONST_A: pto.constexpr = 128,  # compile-time constant (keyword-only)
    CONST_B: pto.constexpr = 64,   # compile-time constant (keyword-only)
):
    # ... tensor views, tile allocation, and kernel logic ...
    return
```

**Positional parameters** are Python-native tensors — they arrive from NumPy,
torch-npu, or any framework with `.shape` and `.strides`. Inside the body, wrap
them with `make_tensor_view` to create GM descriptors.

**Keyword-only parameters** annotated with `pto.constexpr` are compile-time
constants. They must be provided at `.compile()` time and cannot change between
launches of the same compiled kernel. Use them for tile sizes, algorithmic knobs
(e.g., `CAUSAL`), and other values that the compiler can specialize against.

### `mode`: auto vs explicit

`mode` is a keyword on the decorator, not a function parameter. It selects the
programming model:

- `mode="auto"` (the default) is **tile-centric**. You write kernels in terms
  of tiles and Tile Ops. The compiler manages staging, scheduling, and
  synchronization around the tile abstraction.
- `mode="explicit"` adds the full **micro-instruction** surface — MTE ops,
  explicit synchronization, and direct pointer manipulation — on top of
  everything available in `auto`.

Section 3.2 covers the two models in detail.

### Compilation and launch

<!-- ptodsl-doc-pending: host-side compile-and-launch flow is documented but not covered by compile-only docs contract -->
```python
# Compile (traces the body, lowers through PTOAS, caches the result)
compiled = kernel_name.compile(CONST_A=128, CONST_B=64)

# Launch on NPU
compiled[grid, stream](tensor_1, tensor_2, ...)
```

- `.compile(**constexprs)` — traces the kernel body with the given constexpr
  values, lowers the IR, and returns a compiled handle. Subsequent calls with
  the same specialization key (function identity, tensor ABI signature,
  constexpr values) hit the cache.
- `compiled[grid, stream](args...)` — launches the compiled kernel. `grid` is
  the number of SPMD blocks (an integer); `stream` is the NPU stream (`None`
  for default).

### SPMD built-ins

Available inside a `@pto.jit` body:

| Built-in | Returns | Description |
|----------|---------|-------------|
| `pto.get_block_idx()` | `int` | Index of the current block (0-based) |
| `pto.get_block_num()` | `int` | Total number of blocks in the grid |
| `pto.get_subblock_idx()` | `int` | Index of the current sub-block |
| `pto.get_subblock_num()` | `int` | Total number of sub-blocks |

### Typical body (auto mode)

```python
@pto.jit(target="a5", mode="auto")
def my_kernel(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    B: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=B.shape, strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        b_part = pto.partition_view(b_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)
        pto.tile.add(a_tile, b_tile, o_tile)
        pto.tile.store(o_tile, o_part)
```

### Custom sub-kernels

When Tile Ops don't cover the computation you need — a custom softmax, a
specialized activation, per-element blending — you write a sub-kernel in
`@pto.simd`, `@pto.simt`, or `@pto.cube` and call it directly from
`@pto.jit`. In auto mode, data movement stays with Tile Ops
(`tile.load`/`tile.store`) and PTOAS handles the synchronization between Tile
Ops and the sub-kernel:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.direct_l3_call","symbol":"kernel_entry_direct_l3_call_probe","compile":{"BLOCK":128}} -->
```python
@pto.simd
def add_rows(
    a_tile: pto.Tile,
    b_tile: pto.Tile,
    o_tile: pto.Tile,
    rows: pto.index,
    cols: pto.index,
):
    VEC = pto.elements_per_vreg(pto.f32)
    initial_remained = scalar.index_cast(pto.i32, cols)
    with pto.for_(0, rows, step=1) as r:
        col_loop = pto.for_(0, cols, step=VEC).carry(remained=initial_remained)
        with col_loop:
            c = col_loop.iv
            remained = col_loop.remained
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_tile[r, c:])
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_tile[r, c:], mask)
            col_loop.update(remained=remained)

@pto.jit(target="a5", mode="auto")
def my_kernel(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    B: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=B.shape, strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        b_part = pto.partition_view(b_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)

        add_rows(a_tile, b_tile, o_tile, 1, cols)

        pto.tile.store(o_tile, o_part)
```

Sub-kernels are the mechanism for custom compute in PTODSL — when Tile Ops
cover your needs, you don't need one; when they don't, a sub-kernel gives you
direct access to the hardware unit. In auto mode, a sub-kernel's parameters
are restricted to `Tile` and PTO scalar types — the compiler owns staging and
sync. In explicit mode, sub-kernels may also accept `PartitionTensorView` and
`pto.ptr` parameters, matching the richer type surface available there.
Section 3.3 covers each sub-kernel decorator in detail.

## 3.2 Programming models: auto vs explicit

`@pto.jit` exposes a single entry with two programming models. The entry's
host ABI, compilation flow, and launch mechanism are identical in both — the
difference is what you can write inside the kernel body.

### `mode="auto"` — tile-centric

In auto mode you think in tiles. You allocate tiles, partition GM views, move
data with `tile.load` and `tile.store`, compute with Tile Ops like
`tile.add` and `tile.exp`, and call sub-kernels for hardware-specific compute.
The compiler handles the lowering of tiles to micro-instructions: inferring
staging, inserting synchronization between Tile Ops and sub-kernels, and
managing tile-level scheduling.

Use auto mode for the majority of kernels. It gives you the full performance
of the NPU without requiring you to reason about instruction-level ordering.

### `mode="explicit"` — tile + micro-instruction

Explicit mode extends auto mode with direct micro-instruction access. You keep
everything available in auto — tiles, Tile Ops, sub-kernels — and additionally
gain access to MTE ops, explicit synchronization, and pointer manipulation.
When you need precise control over individual instructions and phase ordering,
you can drop below the tile abstraction without leaving the `@pto.jit` entry.

The richer type surface also applies to sub-kernels: in auto mode, a
sub-kernel's parameters are restricted to `Tile` and PTO scalar types; in
explicit mode they may also accept `PartitionTensorView` and `pto.ptr`,
matching the types available in the enclosing orchestration code. Organize
orchestration logic into helper functions that accept these types:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.explicit_signature","symbol":"kernel_entry_explicit_signature_probe","compile":{"BLOCK":16}} -->
```python
def my_orchestration_helper(
    part: pto.PartitionTensorView,   # GM partition descriptors
    tile: pto.Tile,                  # UB tile buffers
    scratch: pto.Tile,               # cube-local scratch (LEFT, RIGHT, ...)
    ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),  # typed UB pointers
    scalar_value: pto.i32,           # PTO scalar values
):
    return
```

**Typical pattern**: GM↔UB movement uses ptr-based `mte_load`/`mte_store`
rather than `tile.load`/`tile.store`. The user places `pipe_barrier` at phase
boundaries and explicitly sequences sub-kernel calls:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.explicit_body","symbol":"kernel_entry_explicit_body_probe","compile":{"ROWS":8,"COLS":16}} -->
```python
def process_block(q_tile, k_part, v_part, k_tile, v_tile,
                  s_tile, o_tile, o_part, rows: pto.i32, cols: pto.i32):
    in_row_bytes = cols * pto.bytewidth(pto.f16)
    out_row_bytes = cols * pto.bytewidth(pto.f32)
    gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f16)
    ub_row_stride = k_tile.shape[1] * pto.bytewidth(pto.f16)

    # Stage current block from GM to UB
    pto.mte_load(k_part.as_ptr(), k_tile.as_ptr(), 0, in_row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.mte_load(v_part.as_ptr(), v_tile.as_ptr(), 0, in_row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.pipe_barrier(pto.Pipe.ALL)

    # Dispatch sub-kernels
    qk_matmul(q_tile, k_tile, s_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    online_softmax(s_tile, o_tile, rows, cols)
    pto.pipe_barrier(pto.Pipe.ALL)

    # Write result back
    pto.mte_store(o_tile.as_ptr(), o_part.as_ptr(), out_row_bytes,
                  nburst=(rows, ub_row_stride, gm_row_stride))
```

Sub-kernel calls and inline sub-kernel scopes (`with pto.simd():`, etc.) work
identically in both modes.

### Choosing between modes

| | `mode="auto"` | `mode="explicit"` |
|---|---|---|
| Abstraction | Tiles | Tiles + micro-instructions |
| Data movement | `tile.load` / `tile.store` | `mte_load` / `mte_store` (ptr-based) |
| Sync | Compiler-managed | User-authored |
| Use case | Most kernels | Hand-tuned instruction scheduling |

Start with auto. Move to explicit when you need to control the exact sequence
of micro-instructions — for example, to overlap DMA and compute with
double-buffering, or to hand-optimize a phase boundary that the compiler
doesn't fuse as aggressively as you need.

## 3.3 Sub-kernels

Sub-kernels are functions decorated with `@pto.cube`, `@pto.simd`, or
`@pto.simt` that execute on a specific NPU compute unit. They can be invoked
in two ways:

1. **As decorated functions** — reusable, named sub-kernels called from
   `@pto.jit`.
2. **As context managers** (`with pto.cube():`, etc.) — inline blocks for
   one-off snippets (see Section 3.4).

### 3.3.1 `@pto.cube` — Cube unit

**Role**: `@pto.cube` marks a function that executes on the Cube unit (matrix
multiplication engine). It consumes UB-resident tiles and explicit cube-local
scratch buffers.

**Signature**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.cube_signature","symbol":"kernel_entry_cube_signature_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.cube
def my_cube_kernel(
    input_tile: pto.Tile,            # UB tile (source data)
    output_tile: pto.Tile,           # UB tile (destination)
    left_scratch: pto.Tile,          # LEFT buffer (cube-local)
    right_scratch: pto.Tile,         # RIGHT buffer (cube-local)
    acc_scratch: pto.Tile,           # ACC buffer (cube-local)
):
    return
```

All parameters are `Tile` references. Tiles marked as cube-local must be
allocated with the appropriate `memory_space` (e.g., `pto.MemorySpace.LEFT`,
`pto.MemorySpace.ACC`).

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.cube_helper","symbol":"data_movement_cube_helper_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.cube
def qk_matmul(
    q_tile: pto.Tile,
    k_tile: pto.Tile,
    q_l0a: pto.Tile,
    k_l0b: pto.Tile,
    s_acc: pto.Tile,
    s_tile: pto.Tile,
):
    m = q_tile.valid_shape[0]
    k = q_tile.valid_shape[1]
    n = k_tile.valid_shape[1]

    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

Cube-local state (LEFT, RIGHT, ACC, BIAS) never leaks into UB — it is the
caller's responsibility to allocate scratch buffers and pass them in
explicitly.

**Invocation modes**: can be called from `@pto.jit` in either mode, or used
inline with `with pto.cube():` (Section 3.4).

### 3.3.2 `@pto.simd` — SIMD unit

**Role**: `@pto.simd` marks a function that executes on the SIMD unit (vector
engine). It operates on vector registers (`vreg`) loaded from UB tiles and
stores results back to UB tiles. Vector registers are local to the function
and never cross its boundary.

**Signature**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.simd_signature","symbol":"kernel_entry_simd_signature_probe","compile":{"BLOCK":128}} -->
```python
@pto.simd
def my_simd_kernel(
    input_tile: pto.Tile,            # UB tile
    output_tile: pto.Tile,           # UB tile
    rows: pto.i32,                   # PTO scalar
    cols: pto.i32,                   # PTO scalar
):
    return
```

Parameters are UB `Tile` references and PTO scalar values (`pto.i32`,
`pto.f32`, etc.). Scalar parameters may come from `lds` reads or compile-time
constants.

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.simd_body","symbol":"kernel_entry_simd_body_probe","compile":{"BLOCK":128}} -->
```python
@pto.simd
def add_rows(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
             rows: pto.index, cols: pto.index):
    VEC = pto.elements_per_vreg(pto.f32)
    initial_remained = scalar.index_cast(pto.i32, cols)
    with pto.for_(0, rows, step=1) as r:
        col_loop = pto.for_(0, cols, step=VEC).carry(remained=initial_remained)
        with col_loop:
            c = col_loop.iv
            remained = col_loop.remained
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_tile[r, c:])
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_tile[r, c:], mask)
            col_loop.update(remained=remained)
```

The boundary contract: `vreg` values (`a_vec`, `b_vec`, `o_vec`) are local to
the function. The only way to persist data across a `@pto.simd` call is to
write it back to a UB tile via `vsts` (or `psts`, etc.).

**Invocation modes**: can be called from `@pto.jit` in either mode, or used
inline with `with pto.simd():` (Section 3.4).

### 3.3.3 `@pto.simt` — SIMT unit

**Role**: `@pto.simt` marks a function that executes on the SIMT unit. SIMT
(Single Instruction, Multiple Threads) is a programming model where you write
instructions in scalar syntax, and the hardware executes them in parallel
across many threads — analogous to how a GPU SM runs a CUDA kernel. Each
instruction appears to operate on a single element (`lds`, `sts`, `a + b`),
but the same instruction is issued across a large number of work-items
simultaneously.

**Signature**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.simt_signature","symbol":"kernel_entry_simt_signature_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def my_simt_kernel(
    tile: pto.Tile,                  # UB tile
    ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),  # typed UB pointer
    scalar_value: pto.i32,           # PTO scalar
):
    return
```

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_blend","symbol":"flash_attention_simt_blend_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile, pv_tile: pto.Tile,
    alpha_tile: pto.Tile, beta_tile: pto.Tile,
    o_next_tile: pto.Tile,
    row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
):
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])
        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])
```

SIMT kernels read and write individual scalar elements from tiles. The unit
executes the same scalar instruction across many work-items in parallel, making
it efficient for per-element operations.

**Invocation modes**: can be called from `@pto.jit` in either mode, or used
inline with `with pto.simt():` (Section 3.4).

## 3.4 Inline context manager syntax

In addition to the decorator form, each sub-kernel unit provides a context
manager: `with pto.cube():`, `with pto.simd():`, and `with pto.simt():`. These
open inline blocks without requiring a separate named function — useful for
quick prototyping, one-off hardware-unit snippets, or code that is too small to
extract. Inline scopes are supported in top-level `@pto.jit` bodies.

### Syntax

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_simd_scope","symbol":"kernel_entry_inline_simd_scope_probe","compile":{"BLOCK":128}} -->
```python
with pto.simd():
    a_vec = pto.vlds(a_tile[r, c:])
    b_vec = pto.vlds(b_tile[r, c:])
    o_vec = pto.vadd(a_vec, b_vec, mask)
    pto.vsts(o_vec, o_tile[r, c:], mask)
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_simt_scope","symbol":"kernel_entry_inline_simt_scope_probe","compile":{"BLOCK":8}} -->
```python
with pto.simt():
    alpha = scalar.load(alpha_tile[row, 0])
    beta = scalar.load(beta_tile[row, 0])
    o_next = alpha * o_prev + beta * pv_val
    scalar.store(o_next, o_next_tile[row, col])
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_cube_scope","symbol":"kernel_entry_inline_cube_scope_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
with pto.cube():
    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

### Semantics

- Inside the `with` block, instructions execute on the corresponding hardware
  unit.
- `vreg` values created inside `with pto.simd():` are scoped to the block —
  they do not escape.
- Cube-local scratch (`l0a`, `l0b`, `acc`) must be allocated by the caller
  before entering the block.
- The context manager form is equivalent to an inline anonymous sub-kernel. The
  compiler treats it identically to a named `@pto.simd` / `@pto.cube` /
  `@pto.simt` function.

### Comparison

| | Decorator form | Context manager form |
|---|---|---|
| Reuse | Named, callable from multiple sites | Inline, single-use |
| Readability | Good for complex, multi-step logic | Good for short (3-10 line) snippets |
| Testing | Can be unit-tested independently | Tested only through the enclosing kernel |
| Cube-local args | Explicit parameters | Captured from enclosing scope |

The two forms can be freely mixed in the same `@pto.jit` body.

## 3.5 Boundary contracts

Data crosses decorator boundaries only through UB-backed tiles or typed UB
pointers:

| Boundary | Allowed |
|----------|---------|
| Host → `@pto.jit` | Python-native tensors |
| `@pto.jit(mode="auto")` → sub-kernel | `Tile`, PTO scalars (compiler handles staging + sync) |
| `@pto.jit(mode="explicit")` → sub-kernel | `Tile`, `PartitionTensorView`, `pto.ptr`, PTO scalars |
| `@pto.jit` → `with pto.{cube,simd,simt}:` | `Tile` captured from enclosing scope |
| Sub-kernel → sub-kernel | Not allowed (go through UB tiles via the caller) |
| `@pto.simd` → caller | Only via `vsts`/`psts` to UB tiles; `vreg` cannot escape |
| Cube-local → UB | Only via `mte_l0c_ub`; LEFT/RIGHT/ACC/BIAS are private |

## 3.6 `pto.constexpr`

`pto.constexpr` marks a `@pto.jit` keyword-only parameter as a compile-time
constant. The compiler specializes the kernel for each combination of constexpr
values, and the compiled artifact is cached by specialization key together with
the kernel's tensor ABI contract.

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def kernel(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
    DTYPE: pto.constexpr = pto.f32,
):
    # ... use BLOCK / DTYPE in tile shapes, loop bounds, or dtype-specialized paths ...
    return
```

- Must appear as a keyword-only argument (after `*`).
- Must have a default value.
- Must be provided at `.compile()` time if the caller needs to override the
  default.
- Cannot change between launches of the same compiled instance — compile a new
  variant for a different value.

`pto.constexpr` parameters can be used anywhere in the kernel body where a
Python value is expected: tile shapes, loop bounds that are known at compile
time, dtype arguments, etc. They are evaluated at trace time, so `for i in
range(BLOCK)` would unroll `BLOCK` times.

In contrast, values derived from runtime tensor shapes (e.g., `A.shape[0]`)
are dynamic — they vary per launch and should be used with `pto.for_` to
produce device-side loops.

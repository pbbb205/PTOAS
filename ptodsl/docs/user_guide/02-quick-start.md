# 2. Quick Start

This chapter walks through a minimal but complete PTODSL kernel — a tiled copy from one GM tensor to another — covering the essential concepts you need to start writing your own kernels.

## 2.1 A first kernel: tiled copy

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"tile_copy","compile":{"BLOCK":128}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def tile_copy(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    """Copy one 2D tensor tile from A to O."""

    rows = A.shape[0]
    cols = A.shape[1]

    # Describe the GM tensors.
    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)

    # Allocate UB tiles for one row-strip block.
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    # Partition the GM views to cover the current logical slice.
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])

    # Load from GM into UB, then store back out.
    pto.tile.load(a_part, a_tile)
    pto.tile.store(o_tile, o_part)
```

Let us step through each piece.

### The entry point

```python
@pto.jit(target="a5")
def tile_copy(A, O, *, BLOCK: pto.constexpr = 128):
```

`@pto.jit` marks this function as a launchable PTO kernel. The positional parameters `A` and `O` are Python-native tensors — they arrive from NumPy, torch-npu, or any framework that provides a shape and strides. Their ABI contract is declared with `pto.tensor_spec(...)`. The keyword-only argument `BLOCK` is a compile-time constant declared with `pto.constexpr`; the compiler specializes the kernel for each tile width.

### Describing GM tensors

```python
a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
```

`make_tensor_view` wraps a Python tensor into a `TensorView` — a descriptor that tells the kernel how to address the tensor in global memory. You provide the logical shape and the stride (in elements) of each dimension.

### Allocating on-chip buffers

```python
a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
```

`alloc_tile` reserves space in the Unified Buffer (UB). A `Tile` is a 2D buffer that lives on-chip during kernel execution. Every tile has a `shape` and a `dtype`.

### Partitioning GM views

```python
a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
```

`partition_view` creates a sub-view of a `TensorView` at a given offset and size. It describes *which part* of the GM tensor a `tile.load` or `tile.store` should operate on. For this simple whole-tensor example the offset is zero and the size matches the logical tensor extent; in a blocked kernel you would slide the offset through a loop.

### Moving data: tile.load and tile.store

```python
pto.tile.load(a_part, a_tile)   # GM → UB
pto.tile.store(o_tile, o_part)  # UB → GM
```

`tile.load` copies a block of data from GM (described by a partition) into a UB tile. `tile.store` copies a UB tile back to GM. These are **Tile Ops** — they operate on entire tile buffers at once.

### Why start with copy

```python
pto.tile.load(a_part, a_tile)
pto.tile.store(o_tile, o_part)
```

A copy kernel strips the example down to the essential PTODSL boundary objects:

- host tensors entering `@pto.jit`
- `TensorView` descriptors over GM tensors
- UB `Tile` allocation
- `PartitionTensorView` slices
- tile-level movement with `tile.load` / `tile.store`

Once these pieces are clear, arithmetic and sub-kernel orchestration become much easier to layer on.

## 2.2 A blocked version with a loop

The kernel above touches one logical slice directly. To introduce device-side control flow, we can iterate over the rows of a 2D tensor and copy one row-strip at a time:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"blocked_copy","compile":{"BLOCK":128}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def blocked_copy(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]

    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)

    tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, tile)
        pto.tile.store(tile, o_part)
```

Here `rows` and `cols` are dynamic — they come from `A.shape` and can differ across launches. The loop bound depends on `rows`, so `pto.for_` records a structured loop in the IR rather than unrolling at trace time. The `BLOCK` parameter stays `constexpr` because it is a tuning knob, not data-dependent. Chapter 5 covers this distinction in detail.

## 2.3 Compile and launch

Once the kernel is defined, you compile it and then launch it:

<!-- ptodsl-doc-pending: host-side compile-and-launch flow is documented but not covered by compile-only docs contract -->
```python
# Compile once, cache the result.
compiled = blocked_copy.compile(BLOCK=128)

# Allocate or obtain input/output tensors (NumPy, torch-npu, ...).
import numpy as np
A = np.random.randn(4, 128).astype(np.float32)
O = np.empty_like(A)

# Launch on the NPU.
compiled[1, None](A, O)
```

- `.compile(**constexprs)` traces the kernel body, lowers it through the PTOAS pipeline, and returns a compiled handle. Repeated calls with the same tensor ABI contract and constexpr configuration hit the cache.
- `compiled[grid, stream](args...)` launches the compiled kernel. `grid` is the number of SPMD blocks; `stream` is the NPU stream (or `None` for the default).

## 2.4 SPMD launch

For workloads that can be parallelized across multiple blocks, specify a grid:

```python
# Process batch * heads slices in parallel.
compiled[batch * heads, stream](Q, K, V, O)
```

Inside the kernel, each block queries its index:

```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
```

This lets you map different data slices to different blocks — for example, one block per (batch, head) pair in flash attention.

## 2.5 Adding sub-kernels and explicit orchestration

The examples above used Tile Ops (`tile.load` / `tile.store` here, and
arithmetic Tile Ops in later chapters), which operate on entire tiles at once.
When you need finer control — for instance, writing a custom softmax or an
activation that maps directly to vector hardware — you can keep the same
`@pto.jit` entry and add sub-kernels. If you also need micro-instruction control,
switch that kernel to `mode="explicit"`:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"vec_add_micro","compile":{"BLOCK":128}} -->
```python
# SIMD sub-kernel — vector instructions on individual rows.
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

# Single kernel entry in explicit mode — micro-instruction staging plus SIMD sub-kernel.
@pto.jit(target="a5", mode="explicit")
def vec_add_micro(
    A: pto.tensor_spec(rank=1, dtype=pto.f32),
    B: pto.tensor_spec(rank=1, dtype=pto.f32),
    O: pto.tensor_spec(rank=1, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    N = A.shape[0]
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    num_blocks = (N + BLOCK - 1) // BLOCK
    with pto.for_(0, num_blocks, step=1) as i:
        offset = i * BLOCK
        this_block = scalar.min(N - offset, BLOCK)
        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[this_block])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])
        row_bytes = this_block * pto.bytewidth(pto.f32)
        pto.mte_load(a_part.as_ptr(), a_tile.as_ptr(), 0, row_bytes,
                     nburst=(1, 0, 0))
        pto.mte_load(b_part.as_ptr(), b_tile.as_ptr(), 0, row_bytes,
                     nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)
        add_rows(a_tile, b_tile, o_tile, 1, this_block)
        pto.pipe_barrier(pto.Pipe.ALL)
        pto.mte_store(o_tile.as_ptr(), o_part.as_ptr(), row_bytes,
                      nburst=(1, 0, 0))
```

- **`@pto.jit(mode="explicit")`**: allocates tiles, partitions the GM views,
  loops over blocks, and directly authors the micro-instruction schedule for
  each block.

- **`@pto.simd` sub-kernel**: the top-level kernel calls a SIMD sub-kernel
  for the row-wise vector work while keeping instruction staging in the
  explicit entry body.

- **Inside `@pto.simd`**: the outer `pto.for_` iterates over rows, the inner
  `pto.for_` iterates over column chunks of the hardware vector width
  (`elements_per_vreg`). Each iteration loads a vector-width slice into a
  `vreg`, does the addition under a mask (for tail elements), and stores the
  result back. Both loops are recorded as structured control flow IR — the
  compiler decides whether to keep them or unroll them.

The same pattern also has an `auto` counterpart: keep `@pto.jit` in its
default mode and replace the explicit `mte_*` sequence with `tile.load` /
`tile.store`. Chapter 3 covers the full entry model; Chapters 7–10 cover each
operation family in detail.

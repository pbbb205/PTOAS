# TileLang DSL Unsupported And Partial Features

## Scope

This document records the gap between the broad language surface described in
`tilelang-dsl-guide.md` and what the current standalone `tilelang_dsl` package
actually implements under:

- `tilelang-dsl/python/tilelang_dsl/`
- `tilelang-dsl/tests/`

Use this file as a quick "what is still missing" index. For the implemented
contract, treat these as the source-of-truth companion documents:

- `v1-surface.md`
- `v1-lowering.md`
- `matcher-and-advanced-surface-migration.md`

## Status Labels

- `Unsupported`: the public surface is documented but not exported or not
  accepted by the frontend at all.
- `Partial`: the concept exists, but only a narrower subset works in the
  current implementation.

## Unsupported Features

### Missing Public Type Constructors And Aliases

The guide documents a richer type-construction surface that is not exported by
the current package:

- `pto.tile(...)`
- `SyncOpType`

Today, the public package exports annotation markers (`TensorView`, `Tile`),
scalar dtypes, `ptr(...)`, `PadMode`, `TileConfig`, matcher APIs, and a small
set of enums. The list above covers the remaining missing public constructors
and aliases from the guide.

### Missing Tile/Tensor Utility Methods

The following guide surfaces are not implemented as public APIs:

- `tile.slice(...)`
- `tile.reshape(...)`
- `pto.tile_from_ptr(...)`
- `pto.tile_with_strides(...)`
- `pto.tile_config(...)`

### Missing Vector Load/Store Families

The current package supports the core v0.3 load/store subset:

- `pto.vlds(...)`
- `pto.vsts(...)`
- `pto.vldsx2(...)`
- `pto.vstsx2(...)`
- `pto.load_scalar(...)`
- `pto.store_scalar(...)`

The following documented load/store families are still unsupported:

- `pto.vsld(...)`
- `pto.vstu(...)`

### Missing Direct Predicate Constructor/Compare APIs

The implementation expects users to go through `pto.make_mask(...)` rather than
call the underlying mask ops directly. These guide-documented APIs are not part
of the supported authoring surface:

- `pto.pset_b8(...)`, `pto.pset_b16(...)`, `pto.pset_b32(...)`
- `pto.pge_b8(...)`, `pto.pge_b16(...)`, `pto.pge_b32(...)`
- `pto.plt_b8(...)`, `pto.plt_b16(...)`, `pto.plt_b32(...)`

### Missing Extended Vector Arithmetic Families

The previously missing `11-vector-arithmetic-operations.md` gap list is now
implemented in the current package surface (including fused ops, broadcast/index
generation, reduction-flavored ops, and rearrangement/sort groups).

### Deferred Surface

`pto.vreduce(...)` is still explicitly deferred and remains rejected even in
`advanced=True` kernels.

## Partial Features

### Scalar Constants And Literal Typing

The guide describes automatic `float -> pto.f32` literal typing.

Literal support currently includes:

- `bool`
- `int`
- `str`
- `None`

### TensorView Attribute Model

`TensorView` currently supports only a narrow attribute subset:

- `shape`
- `strides`
- `element_type`
- `valid_shape`

The following documented attributes are not implemented:

- `offset`

In practice, `TensorView` is now modeled as a fixed 5D GM view in the current
profile, but the DMA-oriented slicing/lowering path remains narrower than the
full guide:

- `shape` / `valid_shape` exposure follows the 5D descriptor
- `strides` lower through hidden stride parameters carried alongside TensorView shape
- fewer written slice axes are right-aligned onto the trailing physical axes
- DMA-oriented slicing/lowering still only accepts rank-2 TensorView slices

### Tile Attribute Model

`Tile` currently exposes the documented metadata/query surface used by the user guide:

- `shape`
- `element_type`
- `memory_space`
- `valid_shape`
- `config`
- `rank`

Current constraints still apply:

- only statically specialized rank-1/rank-2 UB tiles are supported
- `TileConfig` is queryable metadata, but lowering still renders the fixed baseline
  layout contract unless later backend work teaches richer layout semantics

### Tile Config Semantics

`TileConfig` can be attached during specialization, but lowering does not yet
honor the rich layout/padding semantics described in the guide. The rendered
tile type is effectively fixed to a hard-coded baseline:

- `blayout=row_major`
- `slayout=none_box`
- `fractal=512`
- `pad=0`

So this is currently metadata storage rather than full behavioral support.

### TensorView Slicing

The guide presents general Python slicing with dynamic starts and strides. The
current stable DMA-oriented implementation is still a narrower 2D profile:

- slice `stop` must be explicit on all dimensions
- slice `start` may be a compile-time constant or runtime index expression
- slice `step` must be a static positive integer
- dimension 0 may use `step > 1`
- dimension 1 must keep `step == 1` (current DMA restriction)

Dynamic bounds are supported within those constraints.


### Tile Indexing Sugar

Tile indexing sugar is partially implemented on the stable authoring path.

Currently supported:

- rank-1: `tile[start:]`
- rank-2: `tile[row, col:]`
- rank-2 column-major: `tile[row_start:, col_index]`
- for `pto.vlds(...)`, `pto.vsts(...)`, `pto.vldsx2(...)`, and `pto.vstsx2(...)`

Not currently supported from the guide's broader indexing model:

- single-element syntax such as `tile[row, col]` and `tile[pos]`
- explicit slice `stop`
- stepped tile vector slices
- the remaining wider indexed op family gap (`vsld`)

### Control-Flow Result Merging

The frontend does analyze loop-carried values and merged `if` results, but
lowering still has a hard limit:

- at most one loop-carried binding per loop
- at most one merged `if`/`else` binding per conditional

So the language feature exists conceptually, but multi-value merge cases are
not fully lowered yet.

### Tile Profile Breadth

The guide discusses Tile memory spaces in more general terms, but bare Tile
specialization still only accepts:

- rank-1 or rank-2 Tiles
- static physical shape
- `MemorySpace.UB`

So GM Tiles and more general profiles are not supported yet.

## Currently Implemented Core Surface

For quick orientation, the current package head is strongest in these areas:

- matcher-driven kernel selection
- `templates={...}` and `pto.tpl(...)`
- `ptr(...)`, `pto.castptr(...)`, `pto.addptr(...)`
- low-level DMA config/copy ops
- runtime block queries (`pto.get_block_idx`, `pto.get_block_num`, ...)
- `pto.make_mask(...)`
- `pto.vlds(...)`, `pto.vsts(...)`, `pto.vldsx2(...)`, `pto.vstsx2(...)`
- `pto.load_scalar(...)` and `pto.store_scalar(...)`
- base unary/binary/vector-scalar vector ops
- advanced compare/select/carry/rearrangement families

If you need the exact supported boundary for implementation work, prefer the
source files and tests over the broader guide:

- `tilelang-dsl/python/tilelang_dsl/support_matrix.py`
- `tilelang-dsl/python/tilelang_dsl/semantic.py`
- `tilelang-dsl/python/tilelang_dsl/lowering.py`
- `tilelang-dsl/tests/test_tilelang_dsl_v1.py`

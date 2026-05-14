# ptodsl `vpto` POC Proposal

## Background

Today we have two very different authoring paths for VPTO-related Python DSLs:

- `ptodsl` executes Python directly and builds IR through tracing-style wrappers.
- `tilelang-dsl` captures Python source as AST, then runs `frontend_ast -> semantic -> lowering`.

This split is especially visible for tile templates such as
[`lib/TileOps/tadd_template.py`](/home/zhangzhendong/ptoas-workspace/PTOAS/lib/TileOps/tadd_template.py),
whose body is conceptually simple but currently depends on the full AST frontend.

For the longer-term direction, we want VPTO-level authoring to converge on the
same tracing-style route as `ptodsl`, while preserving as much of the
TileLang-style surface as practical.

## Problem Statement

The current AST route gives us good source diagnostics and broad surface
coverage, but it also has clear costs:

- Every new surface feature needs to be added in three layers:
  frontend node building, semantic typing, and text lowering.
- Reusing mature `ptodsl` builder idioms is difficult because authored Python is
  no longer the execution model.
- Simple tile templates still pay the cost of a compiler frontend even when the
  kernel body is static, structured, and already close to the desired VPTO form.

For team discussion, the concrete question is:

Can we execute a TileLang-style tile template directly and emit useful VPTO IR
without going through AST capture?

## Proposal

Introduce an experimental `ptodsl.vpto` namespace as a tracing-oriented POC for
TileLang-style tile templates.

### Design Goals

- Reuse the authored Python function body directly.
- Keep the POC independent from `tilelang-dsl` internals.
- Preserve the most recognizable TileLang surface where it is cheap:
  `@pto.vkernel`, `Tile`, `dst.element_type`, `dst.valid_shape`,
  `tile[row, col:]`, `get_lanes`, `make_mask`, `vlds`, `vadd`, `vsts`.
- Keep the implementation minimal and explicit enough that the team can judge
  whether the tracing route is viable before we invest in broader migration.

### Non-Goals for This POC

- No attempt to replace `tilelang-dsl` in-place.
- No matcher, multi-dtype registry, template slots, inline-proc, or cube
  surface.
- No source-diagnostic parity with the AST frontend.
- No requirement to generalize beyond the minimal pybinding-backed subset
  needed for `tadd_template.py`.

## POC Scope

The POC is intentionally limited to a single template shape:

- Target template: `tadd_template.py`
- Supported parameter kind: bare static 2D `Tile`
- Supported control flow: explicit builder-style `vecscope()` and `for_()`
- Supported ops: `make_mask`, `vlds`, `vadd`, `vsts`
- Supported lowering shape: one `pto.vecscope` containing nested `scf.for`,
  `pto.tile_buf_addr`, and vector micro-ops

This means the first implementation validates the core idea:

1. specialize bare `Tile` parameters with static shape + dtype
2. execute the authored Python body directly
3. trace tile slice accesses such as `src0[row, col:]`
4. emit structured VPTO IR with `scf.for` and no AST capture

## Why This Cut Is Useful

This is not yet the final architecture, but it answers the most important
migration question with low implementation risk:

- If the POC is too awkward even for `tadd_template.py`, we should not try to
  move the main TileLang route onto tracing.
- If the POC stays small and readable, then we have evidence that a tracing
  backend can carry at least a meaningful subset of tile templates.

This cut also forces one important architectural decision early:

- The tracing route should standardize on explicit builder-style control flow.
  Reconstructing `scf.for` from raw Python `for range(...)` would pull us back
  toward AST capture or source transformation, which defeats the purpose of the
  experiment.

## Proposed Architecture

Add a new lightweight module:

- [`ptodsl/ptodsl/vpto.py`](/home/zhangzhendong/ptoas-workspace/PTOAS/ptodsl/ptodsl/vpto.py)

Core pieces:

- `Tile` annotation marker
- `TileSpec(shape, dtype, memory_space="ub")`
- `@vkernel(target="a5", op="pto.tadd")`
- `TracingKernelDescriptor.specialize(...)`
- proxy `Tile` arguments that expose:
  - `.element_type`
  - `.valid_shape`
  - `tile[row, col:]`
- a trace builder that emits structured MLIR objects through Python bindings

The key idea is that `tile[row, col:]` is not lowered from AST. Instead, it is
captured at runtime through a proxy object and immediately converted into a
traced tile-slice value.

## Expected Output Shape

For the `tadd_template.py`-style kernel body, the POC emits:

- tile-buffer arguments
- one `pto.vecscope`
- nested `scf.for` for rows and columns
- `pto.tile_buf_addr` for each referenced tile
- `pto.plt_b32`
- `pto.vlds`
- `pto.vadd`
- `pto.vsts`
- `scf.yield` for loop-carried `remained`

This is intentionally close to the already documented tile-op expand form, but
keeps structured control flow instead of concretely unrolling the loops.

## Tradeoffs

### Advantages

- Very small implementation surface for the first proof point.
- No dependency on AST parsing or source capture.
- Easy to compare source body and emitted IR side by side.
- Makes it clear which parts of TileLang syntax are “real execution” versus
  “frontend-only sugar”.
- Produces IR that is much closer to a future scalable frontend than the
  original fully unrolled POC.

### Limitations

- No rich diagnostics or semantic model yet.
- No integration with the existing `tilelang-dsl` package entrypoint.
- Current output is deliberately narrow and only covers the pybinding-backed
  operations needed by the first POC template.
- Control flow currently needs explicit `vecscope()` / `for_()` builders instead
  of raw Python `for range(...)`.

These are acceptable for the first experiment because the goal is not feature
completeness; it is to validate the tracing execution model on a real tile
template.

## Rollout Path If The POC Works

If the POC proves maintainable, the next steps should be:

1. Add a slightly broader vector subset beyond `tadd`.
2. Replace any remaining POC-specific glue with shared `ptodsl`/MLIR builders.
3. Introduce a reusable runtime contract layer for dtype, mask, and slice
   checks.
4. Decide whether `tilelang-dsl` should:
   - keep AST frontend and optionally target the tracing backend, or
   - expose a parallel tracing-first authoring mode for static templates.

## Deliverables In This Change

- This proposal document
- an experimental `ptodsl.vpto` namespace
- a minimal `tadd_template.py`-oriented POC example

The change is intentionally framed as a team discussion artifact plus a narrow
executable proof-of-concept, not as a replacement plan for the current
TileLang frontend.

# PTO-DSL SIMT Micro-op API Design

## 1. Scope

This document records the PTO-DSL frontend design plan for the SIMT micro-op
surface that is already supported by VPTO on `main`.

The design is intentionally frontend-first:

- expose Python PTO-DSL wrappers for existing VPTO SIMT operations;
- keep wrapper names and parameters close to VPTO IR;
- avoid backend changes unless the frontend generates valid IR that the
  backend incorrectly rejects;
- document open questions before changing lowering, verifiers, or backend
  passes.

The first implementation batch focuses on SIMT launch and query operations.
Later batches are listed for context so the API direction stays consistent.

## 2. References

- SIMT ISA documentation: `docs/isa/micro-isa/17-simt.md`
- VPTO operation definitions: `include/PTO/IR/VPTOOps.td`
- VPTO verifier behavior: `lib/PTO/IR/VPTO.cpp`
- Existing PTO-DSL operation wrappers: `ptodsl/ptodsl/_ops.py`
- Existing PTO-DSL subkernel lowering: `ptodsl/ptodsl/_subkernels.py`
- Existing PTO-DSL tracing session: `ptodsl/ptodsl/_tracing/session.py`
- Existing PTO-DSL SIMT docs: `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
- Existing scalar docs: `ptodsl/docs/user_guide/06-scalar-and-pointer-ops.md`
- Existing SIMT VPTO lit tests: `test/lit/vpto/simt_*`
- Existing SIMT runtime samples: `test/vpto/cases/micro-op/simt/*`

## 3. Current PTO-DSL State

Current PTO-DSL already has a narrow SIMT surface:

- `@pto.simt` decorator and `with pto.simt():` inline scope.
- `pto.store_vfsimt_info(dim_z, dim_y, dim_x)`.
- `pto.get_tid_x()`, `pto.get_tid_y()`, `pto.get_tid_z()`.
- `scalar.load(...)` and `scalar.store(...)` for plain scalar element access.

Current `@pto.simt` helper calls lower to:

```mlir
%dim_z = arith.constant 1 : i32
%dim_y = arith.constant 1 : i32
%dim_x = arith.constant 1 : i32
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @simt_body(...)
```

That path emits a reusable helper function marked with `pto.simt_entry`, but it
does not yet expose user-controlled launch dimensions and does not use
`pto.simt_launch`.

## 4. Full Migration Plan

The full SIMT micro-op PTO-DSL surface can be migrated in staged batches.

### Batch 1: Launch and Query Ops

Expose launch configuration and nullary thread/lane query wrappers:

- `pto.simt_launch(...)`
- `pto.store_vfsimt_info(...)`
- `pto.get_tid_x/y/z()`
- `pto.get_block_dim_x/y/z()`
- `pto.get_grid_dim_x/y/z()`
- `pto.get_block_idx_x/y/z()`
- `pto.get_veccoreid()`
- `pto.get_clock32()`
- `pto.get_clock64()`
- `pto.get_laneid()`
- `pto.get_lanemask_eq/le/lt/ge/gt()`

### Batch 2: Lane Collective Ops

Expose direct wrappers for:

- `pto.vote_all/any/uni/ballot(pred)`
- `pto.shuffle_idx/up/down/bfly(value, control, *, width=32)`
- `pto.redux_add/max/min(value, *, signedness=None)`

### Batch 3: SIMT Scalar Memory and Atomics

Expose direct wrappers for:

- `pto.ldg(ptr, offset=0, *, l1cache="cache", l2cache="nmfv")`
- `pto.stg(value, ptr, offset=0, *, l1cache="cache", l2cache="nmfv")`
- `pto.atomic_exch/add/sub/min/max/and/or/xor(ptr, value, *, l2cache="nmfv", signedness=None)`
- `pto.atomic_cas(ptr, compare, value, *, l2cache="nmfv", signedness=None)`

Plain scalar memory remains available through `scalar.load(...)` and
`scalar.store(...)`.

### Batch 4: SIMT Scalar Math, Convert, Sync, and State

Expose direct wrappers for:

- `pto.prmt(...)`
- `pto.mulhi(...)`
- `pto.mul_i32toi64(...)`
- `pto.absf(...)`, `pto.sqrt(...)`, `pto.exp(...)`, `pto.log(...)`,
  `pto.pow(...)`, `pto.ceil(...)`, `pto.floor(...)`, `pto.rint(...)`,
  `pto.round(...)`, `pto.fmin(...)`, `pto.fmax(...)`, `pto.fma(...)`
- `pto.convert(...)`
- `pto.syncthreads()`, `pto.threadfence()`, `pto.threadfence_block()`
- `pto.keep(...)`, `pto.resume(...)`

`pto.sqrt/exp/log` are VPTO SIMT micro-ops. They are not the same API layer as
the existing `scalar.sqrt/exp/log` helpers, which currently emit generic
`math.*` operations.

## 5. Batch 1 Detailed Design

### 5.1 Goals

Batch 1 should make SIMT launch dimensions and all nullary SIMT runtime queries
authorable from PTO-DSL.

The implementation should:

- keep micro-op names aligned with VPTO op names;
- preserve the low-level `store_vfsimt_info(dim_z, dim_y, dim_x)` order;
- add an ergonomic launch wrapper that uses the launch-site `x, y, z` order;
- preserve current `@pto.simt` helper behavior for existing code;
- avoid backend changes.

### 5.2 Non-goals

Batch 1 should not implement lane collectives, atomics, GM scalar cache policy,
scalar math, conversion, keep/resume, or runtime/ST coverage.

Batch 1 should not change the semantics of `scalar.load/store`.

### 5.3 Operation Mapping

| PTO-DSL API | VPTO IR op | Return |
|---|---|---|
| `pto.store_vfsimt_info(dim_z, dim_y, dim_x)` | `pto.store_vfsimt_info` | `None` |
| `pto.simt_launch(body, *args, dims=(dim_x, dim_y, dim_z))` | `pto.simt_launch` or equivalent `store_vfsimt_info + func.call` | `None` |
| `pto.get_tid_x()` | `pto.get_tid_x` | `i32` |
| `pto.get_tid_y()` | `pto.get_tid_y` | `i32` |
| `pto.get_tid_z()` | `pto.get_tid_z` | `i32` |
| `pto.get_block_dim_x()` | `pto.get_block_dim_x` | `i32` |
| `pto.get_block_dim_y()` | `pto.get_block_dim_y` | `i32` |
| `pto.get_block_dim_z()` | `pto.get_block_dim_z` | `i32` |
| `pto.get_grid_dim_x()` | `pto.get_grid_dim_x` | `i32` |
| `pto.get_grid_dim_y()` | `pto.get_grid_dim_y` | `i32` |
| `pto.get_grid_dim_z()` | `pto.get_grid_dim_z` | `i32` |
| `pto.get_block_idx_x()` | `pto.get_block_idx_x` | `i32` |
| `pto.get_block_idx_y()` | `pto.get_block_idx_y` | `i32` |
| `pto.get_block_idx_z()` | `pto.get_block_idx_z` | `i32` |
| `pto.get_veccoreid()` | `pto.get_veccoreid` | `i32` |
| `pto.get_clock32()` | `pto.get_clock32` | `i32` |
| `pto.get_clock64()` | `pto.get_clock64` | `i64` |
| `pto.get_laneid()` | `pto.get_laneid` | `i32` |
| `pto.get_lanemask_eq()` | `pto.get_lanemask_eq` | `i32` |
| `pto.get_lanemask_le()` | `pto.get_lanemask_le` | `i32` |
| `pto.get_lanemask_lt()` | `pto.get_lanemask_lt` | `i32` |
| `pto.get_lanemask_ge()` | `pto.get_lanemask_ge` | `i32` |
| `pto.get_lanemask_gt()` | `pto.get_lanemask_gt` | `i32` |

### 5.4 Launch API

#### Signature

```python
pto.simt_launch(
    body: pto.SubkernelTemplate,
    *args,
    dims: tuple[int | Scalar, int | Scalar, int | Scalar] = (1, 1, 1),
) -> None
```

`dims` uses `(dim_x, dim_y, dim_z)` order. This matches the textual
`pto.simt_launch @body<<<x, y, z>>>(...)` order and the common launch-site
mental model.

The existing low-level API keeps its backend order:

```python
pto.store_vfsimt_info(dim_z, dim_y, dim_x) -> None
```

This asymmetry is intentional:

- `store_vfsimt_info` is a direct wrapper over the backend operation and should
  not rename or reorder operands.
- `simt_launch` is launch-site sugar and should match the IR sugar order
  `x, y, z`.

#### Example

```python
from ptodsl import pto, scalar


@pto.simt
def write_tid(dst: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    scalar.store(tid, dst, idx)


@pto.jit(target="a5")
def kernel(dst: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    pto.simt_launch(write_tid, dst, dims=(32, 1, 1))
```

Expected source-level IR shape for Batch 1:

```mlir
%dim_x = arith.constant 32 : i32
%dim_y = arith.constant 1 : i32
%dim_z = arith.constant 1 : i32
pto.simt_launch @write_tid<<<%dim_x, %dim_y, %dim_z>>>(%dst)
  : (!pto.ptr<i32, ub>) -> ()
```

Batch 1 emits VPTO `pto.simt_launch` directly. The existing backend
`vpto-expand-wrapper-ops` pass expands it to `pto.store_vfsimt_info + func.call`.

### 5.5 `@pto.simt` Decorator Attributes

SIMT entry functions may carry optional VPTO attributes:

- `pto.simt_max_threads`
- `pto.simt_max_regs`

Proposed PTO-DSL decorator extension:

```python
@pto.simt(max_threads=256, max_regs=48)
def body(...):
    ...
```

Lowering:

```mlir
func.func @body(...) attributes {
  pto.simt_entry,
  pto.simt_max_threads = 256 : i32,
  pto.simt_max_regs = 48 : i32
}
```

Both decorator arguments should be optional. When omitted, PTO-DSL should emit
no explicit attributes and let backend defaults apply.

Validation:

- values must be Python integers known at trace time;
- values must be positive;
- these attributes must only be attached to functions that are already marked
  `pto.simt_entry`.

This extension is useful for launch-envelope documentation and resource
control, but it is not required to expose query ops. It can be implemented in
the same batch or as a small follow-up.

### 5.6 Query API Behavior

All query APIs are nullary wrappers and return a wrapped MLIR SSA value.

Implementation pattern:

```python
def get_laneid():
    return wrap_surface_value(_pto.GetLaneIdOp().result)
```

No Python-side context check is required for the first version. The backend
already knows which operations are legal in `pto.simt_entry` when applicable.
Adding a frontend context check can be considered later if it improves error
messages without hiding backend semantics.

### 5.7 Type Handling for Launch Dimensions

Launch dimensions are VPTO `i32` operands. PTO-DSL should accept:

- Python integer literals;
- PTO scalar values that are already `i32`;
- index-like runtime values when they can be explicitly cast to `i32`.

Proposed normalization rule:

- Python `int` is materialized as signless `i32` constant.
- A runtime scalar with type `i32` is accepted unchanged.
- A runtime scalar with type `index` may be cast to `i32` if existing PTO-DSL
  scalar casting helpers provide a clear path.
- Other types should raise a clear Python `TypeError`.

The implementation should not silently accept `i64` or arbitrary integers by
truncation.

### 5.8 Interaction With Existing `@pto.simt` Calls

Current code can call a SIMT subkernel directly:

```python
write_tid(dst)
```

Today that direct call lowers to launch dimensions `(1, 1, 1)`.

To preserve compatibility, direct `SubkernelTemplate.__call__` behavior should
remain valid and keep its current default launch dimensions. `pto.simt_launch`
is the explicit launch-dimension surface for new code.

Future ergonomic options:

```python
write_tid.launch(dst, dims=(32, 1, 1))
```

This method is not required for Batch 1. If added, it should call the same
lowering path as `pto.simt_launch(...)` and should not create a second semantic
route.

### 5.9 Implementation Sketch

Frontend files likely touched:

- `ptodsl/ptodsl/_ops.py`
  - add nullary query wrappers;
  - add `_coerce_i32_dim(...)` helper if existing helpers are not sufficient;
  - add `simt_launch(...)` wrapper or delegate to tracing runtime.
- `ptodsl/ptodsl/pto.py`
  - export new wrappers.
- `ptodsl/ptodsl/_subkernels.py`
  - optionally extend `simt(..., max_threads=None, max_regs=None)`.
- `ptodsl/ptodsl/_tracing/session.py`
  - add a reusable lowering method for explicit SIMT launches;
  - optionally attach `pto.simt_max_threads` and `pto.simt_max_regs` attrs when
    creating helper functions.
- `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
  - document explicit `pto.simt_launch(...)` and optional decorator attrs.
- `ptodsl/docs/user_guide/06-scalar-and-pointer-ops.md` or a new SIMT section
  - document query ops if we want user-guide coverage in the same PR.
- `ptodsl/tests/support/docs_fragment_fixtures.py`
  - update only if new docs snippets are executable docs-as-tests.
- `ptodsl/tests/test_jit_compile.py`
  - add compile smoke tests for query wrappers and explicit launch dims.

Backend files should not be touched for Batch 1 unless frontend-generated IR is
valid but rejected by existing VPTO code.

### 5.10 Test Plan

Minimum Python/frontend tests:

1. Existing direct `@pto.simt` call still emits `pto.store_vfsimt_info` and a
   single reusable `pto.simt_entry` function.
2. `pto.simt_launch(body, dst, dims=(32, 1, 1))` emits either:
   - `pto.simt_launch @body<<<...>>>`, or
   - an equivalent `pto.store_vfsimt_info` with dimensions reordered to
     `z, y, x` followed by `func.call @body`.
3. All query wrappers compile inside a SIMT body and emit the expected op names.
4. `get_clock64()` returns an `i64` value; all other query wrappers in Batch 1
   return `i32`.
5. Invalid launch dimensions raise Python errors before backend verification
   when the type is clearly unsupported.

Suggested lit/frontend assertions:

- `func.func @body(...) attributes {pto.simt_entry}`
- `pto.get_tid_x`
- `pto.get_block_dim_x`
- `pto.get_grid_dim_x`
- `pto.get_block_idx_x`
- `pto.get_veccoreid`
- `pto.get_clock32`
- `pto.get_clock64`
- `pto.get_laneid`
- `pto.get_lanemask_lt`
- explicit launch dimensions are present in the generated IR.

Runtime/ST validation is not required for the first frontend API PR unless a
later implementation changes runtime behavior.

### 5.11 Open Questions

1. Should `pto.simt_launch(...)` directly emit VPTO `SimtLaunchOp`, or should
   it lower immediately to `store_vfsimt_info + func.call` in PTO-DSL tracing?

   Batch 1 uses direct `SimtLaunchOp` emission. This matches the ISA and keeps
   the frontend surface one-to-one with VPTO. Expansion remains owned by the
   existing backend wrapper-expansion pass.

2. Should direct `@pto.simt` calls remain fixed at `(1, 1, 1)` forever, or
   should they accept launch dims later through a method such as
   `body.launch(..., dims=(...))`?

   Batch 1 preserves current direct-call behavior. A method can be added later
   as pure sugar over `pto.simt_launch(...)`.

3. Should PTO-DSL enforce "query ops only inside `pto.simt_entry`" at Python
   tracing time?

   Batch 1 relies on backend verification. A frontend context check may improve
   diagnostics later, but it should not invent semantics different from VPTO.

4. Should `@pto.simt(max_threads=..., max_regs=...)` be included in Batch 1?

   These attributes are part of the SIMT entry contract and are cheap to expose,
   but they are not necessary for query wrappers. Batch 1 leaves them for a
   follow-up.

## 6. Backend Change Guardrail

Before changing `include/PTO/IR/*`, `lib/PTO/IR/*`, or
`lib/PTO/Transforms/*` for this work, answer:

- Is PTO-DSL generating IR that matches `docs/isa/micro-isa/17-simt.md` and
  `include/PTO/IR/VPTOOps.td`?
- Does the existing backend reject that valid IR?
- Did existing VPTO lit tests already cover the intended backend behavior?
- Can the issue be fixed by wrapper normalization, tracing, docs, or tests?
- If a backend change is still needed, can it be covered by a narrow lit test?

The default answer for Batch 1 should be no backend changes.

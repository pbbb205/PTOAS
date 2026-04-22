### Synchronization & Buffer Control

Operations for pipeline synchronization and buffer management.

#### Enum Types for Synchronization

The following enum types provide type-safe parameter specification for synchronization operations:

- **`BarrierType`**: Memory barrier types for `pto.mem_bar`
  - `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST`: vector→vector barriers
  - `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST`: vector→scalar barriers
  - `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST`: scalar→vector barriers

- **`Pipe`**: Hardware pipeline identifiers
  - `MTE2`: Memory Transfer Engine 2 pipeline
  - `V`: Vector pipeline
  - `MTE3`: Memory Transfer Engine 3 pipeline
  - `ALL`: All pipelines (for barrier operations)

- **`Event`**: Event identifiers for synchronization
  - `ID0`, `ID1`, `ID2`, `ID3`, ..., `ID31`: Event IDs 0-31 (A5 supports 32 event IDs, 0-15 for subblock 0, 16-31 for subblock 1)

#### `pto.set_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Sets a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.wait_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Waits for a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.pipe_barrier(pipes: PIPE) -> None`

**Description**: Executes a barrier across specified pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `PIPE` | Pipeline specification (e.g., `PIPE.ALL`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE

pto.pipe_barrier(PIPE.ALL)
```

#### `pto.get_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Acquire buffer slot for inter-pipeline double-buffering coordination.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Acquisition mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Acquire buffer for MTE2 pipeline
pto.get_buf(Pipe.MTE2, 0, 0)
```

#### `pto.rls_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Release buffer slot to allow other pipeline to proceed.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Release mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Release buffer for MTE2 pipeline
pto.rls_buf(Pipe.MTE2, 0, 0)
```

#### `pto.mem_bar(barrier_type: BarrierType) -> None`

**Description**: Memory barrier for pipeline synchronization within vector scope. Required when UB addresses alias between vector load/store operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `barrier_type` | `BarrierType` | Barrier type controlling prior/subsequent instruction ordering. Supported values are `BarrierType.VV_ALL`, `BarrierType.VST_VLD`, `BarrierType.VLD_VST`, `BarrierType.VST_VST`, `BarrierType.VS_ALL`, `BarrierType.VST_LD`, `BarrierType.VLD_ST`, `BarrierType.VST_ST`, `BarrierType.SV_ALL`, `BarrierType.ST_VLD`, `BarrierType.LD_VST`, and `BarrierType.ST_VST`. |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import BarrierType

# Ensure stores are visible before loads to same UB region
pto.mem_bar(BarrierType.VST_VLD)
```

#### `pto.set_cross_core(core_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event to another core (cross-core synchronization).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Target/source core identifier (platform-specific mapping) |
| `event_id` | `Event` | Cross-core event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 to core 0
pto.set_cross_core(0, Event.ID0)
```

#### `pto.set_intra_block(block_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event within a block (A5). Specifies trigger pipe. 1:1 per subblock.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying trigger pipe |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 on block/pipeline 0
pto.set_intra_block(0, Event.ID0)
```

#### `pto.set_intra_core(config: pto.i32) -> None`

**Description**: Configures intra-core synchronization settings.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `pto.i32` | Configuration value for intra-core synchronization |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_intra_core(3)
```

#### `pto.wait_flag_dev(core_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event from another core. SU-level blocking — entire core stalls.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Core identifier |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 from core 0
pto.wait_flag_dev(0, Event.ID0)
```

#### `pto.wait_intra_core(block_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event within block (A5). Specifies which pipeline should wait — only that pipe stalls, SU and other pipes continue.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying which pipeline should wait |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 on block/pipeline 0
pto.wait_intra_core(0, Event.ID0)
```

### DMA Programming [Advanced Tier]

This section covers Direct Memory Access (DMA) operations for transferring data between Global Memory (GM) and Unified Buffer (UB). DMA operations are performance-critical and require careful configuration of stride parameters and transfer sizes.

**Key Concepts:**
- **DMA Configuration**: Set stride parameters and loop sizes using `set_loop*_stride_*` and `set_loop_size_*` operations.
- **DMA Execution**: Perform transfers using `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, and `copy_ubuf_to_ubuf` operations.
- **GM→UB Padding**: Optionally fill out-of-bounds regions with a specified value when copying from GM to UB. See [Pad Fill Semantics](#pad-fill-semantics) for details.

**Usage Flow:**
1. Configure DMA parameters (strides, loop sizes)
2. Execute the DMA transfer operation
3. Optionally enable padding for GM→UB transfers

**Note**: All DMA operations in this section are part of the **Advanced Tier** and require explicit buffer management and pointer arithmetic. For basic tile-based authoring, refer to the [Basic Authoring Mode](01-introduction.md#basic-vs-advanced-authoring-modes) documentation.

#### Manual Configuration Example

```python
# DMA configuration example (requires careful parameter tuning)
pto.set_loop2_stride_outtoub(src_stride=32, dst_stride=128)  # Outer loop strides
pto.set_loop1_stride_outtoub(src_stride=1, dst_stride=32)    # Inner loop strides
pto.set_loop_size_outtoub(loop1=16, loop2=16)                # Transfer size
pto.copy_gm_to_ubuf(src=gm_ptr, dst=ub_ptr, n_burst=16, len_burst=128, gm_stride=128, ub_stride=128)

```

#### Pad Fill Semantics

When copying data from Global Memory (GM) to Unified Buffer (UB), you can enable padding to fill out-of-bounds regions with a specified value. This is useful when the source data dimensions don't perfectly match the destination tile allocation, or when you need to handle boundary conditions in tiled computations.

##### How Padding Works

1. **Configure the hardware pad register**: Call `pto.set_mov_pad_val` to set the pad value in the hardware register. This must be done before any `pto.copy_gm_to_ubuf` operation with padding enabled.

2. **Enable padding in the DMA operation**: Set `enable_ub_pad=True` in the `pto.copy_gm_to_ubuf` call to activate the padded transfer path. The pad value from the hardware register will be used for filling out-of-bounds regions.

3. **Hardware mapping**: The `pto.set_mov_pad_val` operation corresponds directly to the low-level VPTO instruction that configures the hardware pad register. There is no automatic translation from tile `PadValue` descriptors—you must explicitly set the pad register before padded DMA transfers.

##### Example Workflow

Configure the hardware pad register using `pto.set_mov_pad_val`, then perform the DMA transfer with padding enabled:

```python
# First, configure the hardware pad register with a scalar value
# For zero fill, use an appropriate scalar type based on your data
pto.set_mov_pad_val(pto.f32(0.0))  # Zero fill for float32 data

# Then perform the DMA transfer with padding enabled
pto.copy_gm_to_ubuf(
    src=gm_ptr,
    dst=ub_ptr,
    n_burst=32,
    len_burst=200,
    gm_stride=200,
    ub_stride=256,
    enable_ub_pad=True,  # Enable padded transfer
)
```

##### Accessing Pad Values in Kernel Code

Tile `PadValue` descriptors can be used within kernel code for computation purposes (e.g., initializing vectors with a specific fill value). However, note that **these descriptors are not automatically used for DMA padding**—you must still call `pto.set_mov_pad_val` explicitly to configure the hardware pad register for GM→UB transfers.

To access a pad value from a tile descriptor in kernel code:

```python
# Get the pad descriptor from the destination tile
pad_desc = dst.pad_value

# Check if a valid pad value is configured
if pto.constexpr(pad_desc != pto.PadValue.NULL):
    # Materialize the scalar value
    pad_scalar = pad_desc.eval()
    
    # Use the scalar value (e.g., for vector duplication)
    mask = pto.make_mask(pto.f32, PAT.ALL)
    pad_vector = pto.vdup(pad_scalar, mask)
```

##### Important Notes

- The `PadValue.NULL` descriptor indicates no pad value is configured. Attempting to call `.eval()` on `PadValue.NULL` will raise a frontend error.
- Custom pad values currently support only 32-bit float payloads (`PadValue.custom_f32(...)`).
- Padding only affects GM→UB transfers (`pto.copy_gm_to_ubuf`). UB→GM and UB→UB transfers do not support padding.
- The padded region is determined by the difference between the tile's `valid_shape` and its full `shape`. Ensure your tile is configured with appropriate dimensions.
- Tile `PadValue` descriptors are not automatically used for DMA padding. You must call `pto.set_mov_pad_val` explicitly to configure the hardware pad register for padded GM→UB transfers.

##### `pto.set_mov_pad_val` Operation [Advanced Tier]

The `pto.set_mov_pad_val` operation configures the hardware pad register used for GM→UB transfers when padding is enabled. This operation must be called explicitly before any `pto.copy_gm_to_ubuf` operation with `enable_ub_pad=True`, as the TileLang DSL v1 does not automatically translate tile `PadValue` descriptors to hardware register configurations.

**Operation Signature**:
```python
pto.set_mov_pad_val(pad_value: ScalarType) -> None
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pad_value` | `ScalarType` | Scalar value used for padding. Supported types: any 8/16/32-bit integer scalar (`pto.i8`, `pto.si8`, `pto.ui8`, `pto.i16`, `pto.si16`, `pto.ui16`, `pto.i32`, `pto.si32`, `pto.ui32`) plus `pto.f16`, `pto.bf16`, and `pto.f32`. The value's bit pattern is encoded into the hardware pad register. Integer inputs are automatically normalized to the corresponding signless hardware operand width during lowering, so no manual cast is required before calling `pto.set_mov_pad_val`. For standard pad values, use `PadValue.eval(...)` to obtain the appropriate scalar: `0` or `0.0` for `PadValue.ZERO`, dtype-aware maximum for `PadValue.MAX`, dtype-aware minimum for `PadValue.MIN`. |

**Returns**: None (side-effect operation)

**Example**:

Using a scalar value directly:
```python
# Configure the hardware pad register for zero fill using an integer scalar
pto.set_mov_pad_val(pto.i32(0))  # Zero fill for integer types

# Or using a float scalar for floating-point padding
pto.set_mov_pad_val(pto.f32(0.0))  # Zero fill for float types

# Perform DMA transfer with padding enabled
pto.copy_gm_to_ubuf(
    src=gm_ptr,
    dst=ub_ptr,
    n_burst=32,
    len_burst=200,
    gm_stride=200,
    ub_stride=256,
    enable_ub_pad=True,
)
```

Using a tile's pad value descriptor:
```python
# Get the pad value from a tile configuration
pad_desc = tile.pad_value  # PadValue enum
if pto.constexpr(pad_desc != pto.PadValue.NULL):
    pad_scalar = pad_desc.eval()  # Materializes to a scalar value
    pto.set_mov_pad_val(pad_scalar)
    
    # Perform padded DMA transfer
    pto.copy_gm_to_ubuf(
        src=gm_ptr,
        dst=ub_ptr,
        n_burst=32,
        len_burst=200,
        gm_stride=200,
        ub_stride=256,
        enable_ub_pad=True,
    )
```

Using a standalone `PadValue` with an explicit dtype:
```python
pad_scalar = pto.PadValue.MAX.eval(pto.f32)
pto.set_mov_pad_val(pad_scalar)
```

For integer tile dtypes such as `pto.ui16` or `pto.si32`, `pad_desc.eval()` can be passed directly to `pto.set_mov_pad_val`. TileLang DSL v1 will automatically insert the required same-width bitcast to the signless hardware operand type during lowering.

**Important**: You are responsible for ensuring the pad register is properly configured before any `pto.copy_gm_to_ubuf` operation with `enable_ub_pad=True`. The pad register configuration persists until changed by another `pto.set_mov_pad_val` call.

**Future Improvement**: Future versions of TileLang DSL may provide an implicit approach that automatically translates `PadValue` descriptors from tile configurations to hardware register configurations, similar to DMA syntax sugar features.

#### `pto.set_loop2_stride_outtoub(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for GM → UB transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_outtoub(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for GM → UB transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_outtoub(loop1: pto.i64, loop2: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for GM → UB transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop1` | `pto.i64` | Inner loop trip count |
| `loop2` | `pto.i64` | Outer loop trip count |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size_outtoub(loop1=1, loop2=1)
```

#### `pto.set_loop2_stride_ubtoout(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for UB → GM transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_ubtoout(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for UB → GM transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_ubtoout(loop1: pto.i64, loop2: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for UB → GM transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop1` | `pto.i64` | Inner loop trip count |
| `loop2` | `pto.i64` | Outer loop trip count |

**Returns**: None (side-effect operation)

#### `pto.set_loop(loop_id: pto.i32, src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for a generic loop.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop_id` | `pto.i32` | Loop identifier (e.g., 1 for inner loop, 2 for outer loop) |
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop(1, src_stride=32, dst_stride=64)
```

#### `pto.set_loop_size(loop_id: pto.i32, size: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for a generic loop.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop_id` | `pto.i32` | Loop identifier (e.g., 1 for inner loop, 2 for outer loop) |
| `size` | `pto.i64` | Loop trip count |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size(1, 16)
```

#### DMA Execution Operations

**Note**: These operations execute DMA transfers but require manual configuration of DMA parameters (loop strides, loop sizes) using the `set_loop*_stride_*` and `set_loop_size_*` operations described above.

The following operations provide direct control over DMA transfers but require manual stride and size configuration.

#### `pto.copy_gm_to_ubuf(src: GMPtr, dst: UBPtr, sid: pto.i64 = 0, n_burst: pto.i64, len_burst: pto.i64, left_padding_count: pto.i64 = 0, right_padding_count: pto.i64 = 0, enable_ub_pad: pto.i1 = False, l2_cache_ctl: pto.i64 = 0, gm_stride: pto.i64, ub_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data from Global Memory (GM) to Unified Buffer (UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GMPtr` | Source GM pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `sid` | `pto.i64` | DMA stream/control operand, defaults to `0` |
| `n_burst` | `pto.i64` | Number of bursts |
| `len_burst` | `pto.i64` | Bytes copied by each burst |
| `left_padding_count` | `pto.i64` | Left padding count, defaults to `0` |
| `right_padding_count` | `pto.i64` | Right padding count, defaults to `0` |
| `enable_ub_pad` | `pto.i1` | Convenience alias for `data_select_bit`, defaults to `False` |
| `l2_cache_ctl` | `pto.i64` | L2 cache control operand, defaults to `0` |
| `gm_stride` | `pto.i64` | GM-side stride in bytes |
| `ub_stride` | `pto.i64` | UB-side stride in bytes |

**Returns**: None (side-effect operation)

**Notes**:
- **Keyword arguments**: The keyword form shown above is the recommended public API surface. Use named arguments for clarity.
- **Padding control**: Set `enable_ub_pad=True` to enable padded GM→UB transfers. The pad value must be configured separately using `pto.set_mov_pad_val` before the DMA operation (see [Pad Fill Semantics](#pad-fill-semantics) for details).
- **Pad value source**: When padding is enabled, the fill scalar comes from the hardware pad register configured by `pto.set_mov_pad_val`. You must call this operation explicitly before the DMA transfer.
- **ABI compatibility**: The lowering preserves the underlying PTO operand order while providing a more ergonomic keyword interface.

**Example**:
```python
pto.copy_gm_to_ubuf(
    src=gm_ptr,
    dst=ub_ptr,
    n_burst=32,
    len_burst=128,
    gm_stride=128,
    ub_stride=128,
    enable_ub_pad=False,
)
```

**Padding Example**:
```python
# First configure the hardware pad register with a scalar value
pto.set_mov_pad_val(pto.f32(0.0))  # Zero fill for float32 data

# Then perform padded DMA transfer
pto.copy_gm_to_ubuf(
    src=gm_ptr,
    dst=ub_ptr,
    n_burst=32,
    len_burst=200,
    gm_stride=200,
    ub_stride=256,
    enable_ub_pad=True,
)
```

#### `pto.copy_ubuf_to_ubuf(src: UBPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data within Unified Buffer (UB → UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

#### `pto.copy_ubuf_to_gm(src: UBPtr, dst: GMPtr, sid: pto.i64 = 0, n_burst: pto.i64, len_burst: pto.i64, reserved: pto.i64 = 0, gm_stride: pto.i64, ub_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data from Unified Buffer (UB) to Global Memory (GM).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `GMPtr` | Destination GM pointer |
| `sid` | `pto.i64` | DMA stream/control operand, defaults to `0` |
| `n_burst` | `pto.i64` | Number of bursts |
| `len_burst` | `pto.i64` | Bytes copied by each burst |
| `reserved` | `pto.i64` | Reserved operand, defaults to `0` |
| `gm_stride` | `pto.i64` | GM-side stride in bytes |
| `ub_stride` | `pto.i64` | UB-side stride in bytes |

**Returns**: None (side-effect operation)

**Notes**:
- In TileLang DSL, the keyword form above is the recommended public surface.
- `gm_stride`/`ub_stride` are ergonomic aliases for the low-level `burst_dst_stride`/`burst_src_stride` operands.
- The lowering still maps to the underlying low-level PTO operand ABI in positional order.

**Example**:
```python
pto.copy_ubuf_to_gm(
    src=ub_ptr,
    dst=gm_ptr,
    n_burst=32,
    len_burst=128,
    gm_stride=128,
    ub_stride=128,
)
```

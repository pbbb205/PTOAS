### Unary Vector Operations

Element-wise unary operations on vector registers.

#### `pto.vabs(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Absolute value of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match vector element type) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Absolute values |

**Constraints**:
- Mask granularity must match vector element type (e.g., `f32` requires `mask_b32`)

**Example**:
```python
abs_vec = pto.vabs(vec_f32, mask32)
```

#### `pto.vexp(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Exponential of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Exponential values |

#### `pto.vln(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Natural logarithm of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Natural logarithm values |

#### `pto.vsqrt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Square root of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Square root values |

#### `pto.vrec(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Reciprocal of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reciprocal values |

#### `pto.vrelu(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: ReLU activation (max(0, x)) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated values |

#### `pto.vnot(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitwise NOT of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise NOT values |

#### `pto.vcadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex addition of vector elements (treating pairs as complex numbers).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex addition result |

#### `pto.vcmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex maximum of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex maximum result |

#### `pto.vbcnt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bit count (population count) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bit count values |

#### `pto.vneg(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Negation of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match vector element type) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Negated values |

**Constraints**:
- Mask granularity must match vector element type

**Example**:
```python
neg_vec = pto.vneg(vec_f32, mask32)
```

#### `pto.vcls(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Count leading sign bits of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Count of leading sign bits |

**Constraints**:
- Operates on integer vector types only

#### `pto.vcmin(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex minimum of vector elements (treating pairs as complex numbers).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex minimum result |

#### `pto.vrsqrt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Reciprocal square root of vector elements (1/√x).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reciprocal square root values |

**Constraints**:
- For floating-point vector types only

#### `pto.vprelu(vec: VRegType, alpha: VRegType, mask: MaskType) -> VRegType`

**Description**: Parametric ReLU activation of vector elements: `x if x >= 0 else alpha * x`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `alpha` | `VRegType` | Slope parameter for negative values |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Parametric ReLU activated values |

#### `pto.vmov(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Vector move (data movement).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Copied vector |

#### `pto.vsunpack(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Signed unpack of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Unpacked signed values |

**Constraints**:
- Operates on integer vector types only

#### `pto.vzunpack(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Zero-extended unpack of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Unpacked zero-extended values |

**Constraints**:
- Operates on integer vector types only

#### `pto.vusqz(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Unsigned squeeze (compression) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Compressed unsigned values |

**Constraints**:
- Operates on integer vector types only

#### `pto.vsqz(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Signed squeeze (compression) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Compressed signed values |

**Constraints**:
- Operates on integer vector types only

#### `pto.vexpdiff(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Exponential difference of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Exponential difference values |

**Constraints**:
- For floating-point vector types only

### Binary Vector Operations

Element-wise binary operations on vector registers.

#### `pto.vadd(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise addition of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum of vectors |

**Example**:
```python
sum_vec = pto.vadd(vec_a, vec_b, mask32)
```

#### `pto.vsub(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise subtraction of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference of vectors |

#### `pto.vmul(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise multiplication of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Product of vectors |

#### `pto.vdiv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise division of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Quotient of vectors |

#### `pto.vmax(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise maximum |

#### `pto.vmin(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise minimum |

#### `pto.vand(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise AND of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise AND result |

#### `pto.vor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise OR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise OR result |

#### `pto.vxor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise XOR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise XOR result |

#### `pto.vshl(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift left (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshr(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift right (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vaddrelu(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Addition with ReLU activation (max(0, vec1 + vec2)).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated sum of vectors |

#### `pto.vaddreluconv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Convolution addition with ReLU activation (convolution-specific fused operation).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated convolution sum |

**Constraints**:
- Optimized for convolution-specific patterns

#### `pto.vsubrelu(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Subtraction with ReLU activation (max(0, vec1 - vec2)).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated difference of vectors |

#### `pto.vaxpy(alpha: VRegType, x: VRegType, y: VRegType, mask: MaskType) -> VRegType`

**Description**: BLAS AXPY operation (αx + y).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | `VRegType` | Scaling factor |
| `x` | `VRegType` | Input vector x |
| `y` | `VRegType` | Input vector y |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result of αx + y |

#### `pto.vmulconv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Convolution multiplication (convolution-specific multiplication).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Convolution product |

**Constraints**:
- Optimized for convolution-specific patterns

#### `pto.vmull(vec1: VRegType, vec2: VRegType, mask: MaskType) -> (VRegType, VRegType)`

**Description**: Widening multiply with split low/high results (extended arithmetic).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Low part of widened product (`r & 0xFFFFFFFF`) |
| `high` | `VRegType` | High part of widened product (`r >> 32`) |

**Constraints**:
- Current A5 documented form is native `i32/u32` 32x32->64 widening multiply
- Result is split into two vector outputs instead of a single widened vector

**Example**:
```python
low, high = pto.vmull(lhs_i32, rhs_i32, mask32)
```

#### `pto.vmula(vec1: VRegType, vec2: VRegType, vec3: VRegType, mask: MaskType) -> VRegType`

**Description**: Fused multiply-add (vec1 * vec2 + vec3).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector (multiplier) |
| `vec2` | `VRegType` | Second input vector (multiplicand) |
| `vec3` | `VRegType` | Third input vector (addend) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result of vec1 * vec2 + vec3 |

### Vector-Scalar Operations

Operations between vectors and scalars.

#### `pto.vmuls(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector multiplied by scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar multiplier |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Scaled vector |

**Example**:
```python
scaled = pto.vmuls(vec_f32, pto.f32(2.0), mask32)
```

#### `pto.vadds(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector plus scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar addend |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

#### `pto.vmaxs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Maximum values |

#### `pto.vmins(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Minimum values |

#### `pto.vlrelu(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Leaky ReLU activation (max(αx, x)).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Alpha coefficient |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Leaky ReLU activated values |

#### `pto.vshls(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift left by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshrs(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift right by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vands(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise AND of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise AND result |

**Constraints**:
- Operates on integer vector types only

#### `pto.vors(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise OR of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise OR result |

**Constraints**:
- Operates on integer vector types only

#### `pto.vxors(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise XOR of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise XOR result |

**Constraints**:
- Operates on integer vector types only

#### `pto.vsubs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector minus scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar subtrahend |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference vector |

#### `pto.vbr(value: ScalarType) -> VRegType`

**Description**: Broadcast scalar to all vector lanes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` | Scalar source |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Vector whose active lanes all carry `value` |

**Constraints**:
- Supported scalar types are the 8/16/32-bit integer families (`i*`, `si*`, `ui*`) plus `f16`, `bf16`, and `f32`.
- For integer types, only the low bits of the scalar source are consumed according to the bit width (8, 16, or 32 bits).

**Example**:
```python
# Broadcast scalar constant to vector
zero_vec = pto.vbr(0.0)
one_vec = pto.vbr(1.0)

# Reduction seed with explicit floating dtype
rowmax_seed_f32 = pto.vbr(pto.f32("-inf"))
rowmax_seed_f16 = pto.vbr(pto.f16("0xFC00"))
```

#### `pto.vdup(input: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vdup(input: VRegType, mask: MaskType, position: PositionMode = PositionMode.LOWEST) -> VRegType`

**Description**: Duplicate a scalar value or one selected vector element into
the active lanes of a destination vector.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `ScalarType` or `VRegType` | Input scalar or source vector |
| `mask` | `MaskType` | Predicate mask controlling which lanes are written |
| `position` | `PositionMode` | Optional enum for the vector-input overload, selecting the source vector element to duplicate (default: `PositionMode.LOWEST`) |

**Position Mode Enum**: The `PositionMode` enum provides type-safe source-lane
selection for `pto.vdup`. `LOWEST` selects the lowest-index element of the
source vector and `HIGHEST` selects the highest-index element. The enum is only
used by the vector-input overload.

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Vector whose active lanes receive the duplicated value |

**Constraints**:
- `mask` granularity must match the destination vector element type. For
  example, `f32`/`i32`/`si32`/`ui32` vectors require `mask_b32`.
- When `input` is a scalar, the scalar value is duplicated to every active lane.
- When `input` is a vector, `position` selects a single source element and that
  value is duplicated to every active lane.
- The scalar overload does not accept `position`.
- Inactive lanes follow VPTO predicate semantics and are not guaranteed to carry
  meaningful values for subsequent masked-off use.
- Supported scalar types are the 8/16/32-bit integer families (`i*`, `si*`, `ui*`) plus `f16`, `bf16`, and `f32`.
- `position` is only meaningful for vector input. TileLang DSL currently exposes
  `PositionMode.LOWEST` and `PositionMode.HIGHEST`, matching VPTO v0.3.

**Example**:
```python
mask32 = pto.make_mask(pto.f32, pto.PAT.ALL)

# Duplicate a scalar into all active lanes.
broadcast = pto.vdup(3.14, mask32)

# Use dtype constructors for floating-point special values.
seed = pto.vdup(pto.f32("-inf"), mask32)
seed_f16 = pto.vdup(pto.f16("0xFC00"), pto.make_mask(pto.f16, pto.PAT.ALL))

# Assume `vec` is an existing `f32` vector register value.
vec = pto.vlds(src, 0)

# Duplicate the lowest source lane to all active lanes.
dup_lowest = pto.vdup(vec, mask32)  # position defaults to "LOWEST"

# Duplicate the highest source lane to all active lanes.
dup_highest = pto.vdup(vec, mask32, pto.PositionMode.HIGHEST)
```

**Type Safety Note**:
- For floating-point seeds, prefer `pto.f16(...)` / `pto.bf16(...)` / `pto.f32(...)` constructors.
- Do not pass integer bit-pattern literals directly (for example `0xFF800000`) when a floating vector type is intended.

### Carry & Select Operations

Operations with carry propagation and selection.

**Comparison Mode Enum**: The `CmpMode` enum provides type-safe comparison mode specification for `pto.vcmp` and `pto.vcmps` operations. It includes the following values: `EQ` (equal), `NE` (not equal), `LT` (less than), `LE` (less than or equal), `GT` (greater than), `GE` (greater than or equal).

Implemented current-package carry/select surface also includes:
- `pto.vselr(vec0, vec1) -> VRegType`
- `pto.vselrv2(vec0, vec1) -> VRegType`
- `pto.vaddcs(vec0, vec1, carry_in, mask) -> (VRegType, MaskType)`
- `pto.vsubcs(vec0, vec1, carry_in, mask) -> (VRegType, MaskType)`

#### `pto.vcmp(vec0: VRegType, vec1: VRegType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Element-wise vector comparison with seed mask. Compares two vectors element-wise and generates a predicate mask based on the specified comparison mode.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec0` | `VRegType` | First input vector |
| `vec1` | `VRegType` | Second input vector |
| `seed_mask` | `MaskType` | Seed mask that determines which lanes participate in the comparison |
| `cmp_mode` | `CmpMode` | Comparison mode enum: `CmpMode.EQ` (equal), `CmpMode.NE` (not equal), `CmpMode.LT` (less than), `CmpMode.LE` (less than or equal), `CmpMode.GT` (greater than), `CmpMode.GE` (greater than or equal) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Generated predicate mask based on element-wise comparison |

**Constraints**:
- Only lanes enabled by `seed_mask` participate in the comparison
- The two input vectors must have the same element type and vector length
- The output mask granularity matches the input vector element type

**Example**:
```python
# Compare two vectors for less-than relation
all_mask = pto.make_mask(pto.f32, PAT.ALL)
lt_mask = pto.vcmp(vec_a, vec_b, all_mask, CmpMode.LT)
```

#### `pto.vcmps(vec: VRegType, scalar: ScalarType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Vector-scalar comparison with seed mask. Compares each element of a vector against a scalar value and generates a predicate mask based on the specified comparison mode.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value to compare against (must match vector element type) |
| `seed_mask` | `MaskType` | Seed mask that determines which lanes participate in the comparison |
| `cmp_mode` | `CmpMode` | Comparison mode enum: `CmpMode.EQ`, `CmpMode.NE`, `CmpMode.LT`, `CmpMode.LE`, `CmpMode.GT`, `CmpMode.GE` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Generated predicate mask based on vector-scalar comparison |

**Constraints**:
- Only lanes enabled by `seed_mask` participate in the comparison
- The scalar type must match the vector element type
- The output mask granularity matches the input vector element type

**Example**:
```python
# Check which elements are greater than zero
all_mask = pto.make_mask(pto.f32, PAT.ALL)
positive_mask = pto.vcmps(values, pto.f32(0.0), all_mask, CmpMode.GT)
```

#### `pto.vaddc(vec1: VRegType, vec2: VRegType, mask: MaskType) -> (VRegType, MaskType)`

**Description**: Vector addition with carry output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum vector |
| `carry_out` | `MaskType` | Output carry mask |

#### `pto.vsubc(vec1: VRegType, vec2: VRegType, mask: MaskType) -> (VRegType, MaskType)`

**Description**: Vector subtraction with borrow output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference vector |
| `borrow_out` | `MaskType` | Output borrow mask |

#### `pto.vsel(true_vec: VRegType, false_vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Vector select based on mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `true_vec` | `VRegType` | Vector selected when mask bit is 1 |
| `false_vec` | `VRegType` | Vector selected when mask bit is 0 |
| `mask` | `MaskType` | Selection mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

**Example**:
```python
result = pto.vsel(scaled_vec, original_vec, mask32)
```

### Reduction Operations

Reduction operations across vector lanes or channels.

#### `pto.vcgadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Cross-group addition reduction (reduction across VLanes).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced sum across groups |

#### `pto.vcgmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Cross-group maximum reduction (reduction across VLanes).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced maximum across groups |

#### `pto.vcgmin(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Cross-group minimum reduction (reduction across VLanes).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced minimum across groups |

#### `pto.vcpadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Cross-channel addition reduction (reduction across channels).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced sum across channels |

### Data Rearrangement

Operations for rearranging data within vectors.

Predicate rearrangement ops `pto.pdintlv_b8` and `pto.pintlv_b16` are documented in `10-predicate-operations.md` because they operate on predicate masks rather than vector registers.

Implemented current-package rearrangement surface also includes:
- `pto.vintlvv2(vec0, vec1, part) -> VRegType`
- `pto.vdintlvv2(vec0, vec1, part) -> VRegType`

#### `pto.vintlv(vec1: VRegType, vec2: VRegType) -> (VRegType, VRegType)`

**Description**: Interleave two vectors and return the low/high results.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Low interleaved result |
| `high` | `VRegType` | High interleaved result |

#### `pto.vdintlv(vec0: VRegType, vec1: VRegType) -> (VRegType, VRegType)`

**Description**: Deinterleave a pair of vectors into low/high results.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec0` | `VRegType` | First input vector |
| `vec1` | `VRegType` | Second input vector |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | First deinterleaved vector |
| `vec2` | `VRegType` | Second deinterleaved vector |

#### `pto.vpack(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Vector packing (combine elements from two vectors).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Packed vector |

#### `pto.vperm(vec: VRegType, indices: VRegType, mask: MaskType) -> VRegType`

**Description**: Vector permutation (reorder elements according to index vector).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `indices` | `VRegType` | Permutation indices |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Permuted vector |

#### `pto.vshift(vec: VRegType, shift_amount: ScalarType, mask: MaskType) -> VRegType`

**Description**: Generic vector shift (shift all elements by same amount).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift_amount` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted vector |

#### `pto.vslide(vec: VRegType, window_size: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector sliding window (create overlapping windows).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `window_size` | `ScalarType` | Size of sliding window |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sliding window result |

#### `pto.vsort32(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: 32-element sorting of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (32 elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sorted vector |

**Constraints**:
- Input vector must have exactly 32 elements

#### `pto.vmrgsort(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Merge sort of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Merged and sorted vector |

#### `pto.vtranspose(dest: ptr, src: ptr, config: pto.i64) -> None`  [Advanced Tier]

**Description**: UB-to-UB transpose operation. This op works on UB memory directly (not `vreg -> vreg`).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dest` | `ptr` | Destination pointer in UB memory space |
| `src` | `ptr` | Source pointer in UB memory space |
| `config` | `pto.i64` | ISA control/config operand that encodes transpose layout behavior |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `None` | `None` | Side-effect operation that writes transposed data to `dest` |

**Constraints**:
- `dest` and `src` must be UB pointers
- Correctness depends on the `config` encoding and UB layout contract

**Example**:
```python
pto.vtranspose(dst_ub_ptr, src_ub_ptr, config_word)
```

### Conversion & Special Operations

Type conversion and specialized operations.

#### `pto.vtrc(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Truncate vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Truncated vector |

#### `pto.vcvt(vec: VRegType, to_type: Type, mask: MaskType, rnd: pto.VcvtRoundMode | None = None, sat: pto.VcvtSatMode | None = None, part: pto.VcvtPartMode | None = None) -> VRegType`

**Description**: Convert vector elements between supported float and integer
families. This is the TileLang DSL surface for the VPTO `pto.vcvt` conversion
family.

**Attribute Enums**:
- `pto.VcvtRoundMode`: `R`, `A`, `F`, `C`, `Z`, `O`
- `pto.VcvtSatMode`: `SAT`, `NOSAT`
- `pto.VcvtPartMode`: `EVEN`, `ODD`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `to_type` | `Type` | Target scalar dtype symbol for the result vector element type |
| `mask` | `MaskType` | Predicate mask selecting active source lanes. Its granularity must match the source vector family, not the destination family |
| `rnd` | `pto.VcvtRoundMode` \| `None` | Optional rounding-mode attribute lowered to VPTO `rnd` |
| `sat` | `pto.VcvtSatMode` \| `None` | Optional saturation attribute lowered to VPTO `sat` |
| `part` | `pto.VcvtPartMode` \| `None` | Optional even/odd packing selector lowered to VPTO `part` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Converted vector with the vreg shape implied by `to_type` |

**Constraints**:
- Current TileLang DSL v1 accepts exactly three positional arguments:
  `pto.vcvt(vec, to_type, mask)`. Optional attributes are exposed as keyword
  arguments: `rnd=...`, `sat=...`, `part=...`.
- The underlying VPTO op family is the fuller
  `pto.vcvt %input, %mask {rnd, sat, part}` surface, and the DSL keywords map
  directly to those VPTO attributes.
- `mask` always follows the source vector family:
  `f32`/`i32`/`si32`/`ui32` use `mask_b32`;
  `f16`/`bf16`/`i16`/`si16`/`ui16` use `mask_b16`;
  `i8`/`si8`/`ui8` use `mask_b8`.
- The enum form is preferred. For compatibility, canonical strings such as
  `"R"`, `"SAT"`, and `"EVEN"` are also accepted.
- Only backend-supported source/destination type pairs are legal. For the full
  A5 `vcvt` type matrix, width-changing packing rules, and attribute-sensitive
  forms, refer to
  [`../vpto_spec/vpto-spec-current.md`](../vpto_spec/vpto-spec-current.md).
- VPTO does not define a `mask_b64` form. Conversions that produce `si64`
  results still use the typed mask granularity of the source vector family.
- Width-changing conversions continue to follow VPTO packing semantics even on
  the simplified DSL surface. For example, `f16 -> f32` uses an `f16`-family
  `mask_b16`, because the mask is attached to the source vector family.

**Example**:
```python
mask16 = pto.make_mask(pto.f16, PAT.ALL)
vec_f16 = pto.vlds(src, 0)
vec_f32 = pto.vcvt(vec_f16, pto.f32, mask16)

mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_i32 = pto.vcvt(vec_f32, pto.si32, mask32)

vec_f16_narrow = pto.vcvt(
    vec_f32,
    pto.f16,
    mask32,
    rnd=pto.VcvtRoundMode.R,
    sat=pto.VcvtSatMode.SAT,
    part=pto.VcvtPartMode.ODD,
)
```

#### `pto.vbitsort(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitonic sort of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sorted vector |

#### `pto.vmrgsort4(vec1: VRegType, vec2: VRegType, vec3: VRegType, vec4: VRegType, mask: MaskType) -> VRegType`

**Description**: 4-way merge sort of vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `vec3` | `VRegType` | Third input vector |
| `vec4` | `VRegType` | Fourth input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Merged and sorted vector |

**Order Mode Enum**: The `OrderMode` enum provides type-safe order selection for `pto.vci` operations. Currently only `ASC` (ascending order) is supported, with more order options planned for future releases.

#### `pto.vci(index: ScalarType, order: OrderMode = OrderMode.ASC) -> VRegType`

**Description**: Generate a lane-index vector from a scalar seed/index value (DSA/SFU operation).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `ScalarType` | Scalar seed or base index value |
| `order` | `OrderMode` | Order mode enum (default: `OrderMode.ASC` for ascending order) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Generated index vector |

**Constraints**:
- This is an index-generation family, not a numeric conversion
- The `order` parameter and result element type together determine how indices are generated
- Currently only ascending order (`OrderMode.ASC`) is supported

**Example**:
```python
# Generate ascending indices starting from 0
indices = pto.vci(pto.i32(0), OrderMode.ASC)

# Keyword form for the optional order argument is also supported
indices_kw = pto.vci(pto.i32(0), order=OrderMode.ASC)
```

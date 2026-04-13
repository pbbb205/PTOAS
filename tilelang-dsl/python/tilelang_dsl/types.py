# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public type markers for the TileLang DSL v1 surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __repr__(self) -> str:
        return self.name


_INTEGER_DTYPE_WIDTHS = {
    "i8": 8,
    "si8": 8,
    "ui8": 8,
    "i16": 16,
    "si16": 16,
    "ui16": 16,
    "i32": 32,
    "si32": 32,
    "ui32": 32,
    "i64": 64,
    "si64": 64,
    "ui64": 64,
}

_INTEGER_DTYPE_SIGNS = {
    "i8": "signless",
    "si8": "signed",
    "ui8": "unsigned",
    "i16": "signless",
    "si16": "signed",
    "ui16": "unsigned",
    "i32": "signless",
    "si32": "signed",
    "ui32": "unsigned",
    "i64": "signless",
    "si64": "signed",
    "ui64": "unsigned",
}

_FLOAT_DTYPE_WIDTHS = {
    "f16": 16,
    "bf16": 16,
    "f32": 32,
}

_DTYPE_BYTE_WIDTHS = {
    name: bits // 8 for name, bits in _INTEGER_DTYPE_WIDTHS.items()
}
_DTYPE_BYTE_WIDTHS.update({name: bits // 8 for name, bits in _FLOAT_DTYPE_WIDTHS.items()})


class TensorView:
    """Bare TensorView annotation marker for TileLang DSL v1."""


class PartitionTensorView:
    """Bare PartitionTensorView annotation marker for TileLang DSL v1."""


class Tile:
    """Bare Tile annotation marker for TileLang DSL v1."""


@dataclass(frozen=True)
class PointerType:
    element_dtype: ScalarType
    memory_space: "MemorySpace"

    def __repr__(self) -> str:
        return f"ptr({self.element_dtype!r}, {self.memory_space!r})"


@dataclass(frozen=True)
class VRegType:
    element_dtype: ScalarType
    lanes: int

    def __repr__(self) -> str:
        return f"vreg({self.element_dtype!r})"


@dataclass(frozen=True)
class MaskType:
    granularity: str

    def __repr__(self) -> str:
        return f"mask_{self.granularity}"


@dataclass(frozen=True)
class AlignType:
    def __repr__(self) -> str:
        return "align"


@dataclass(frozen=True)
class WildcardType:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVariable:
    name: str

    def __repr__(self) -> str:
        return f"TypeVar({self.name!r})"


class MemorySpace(str, Enum):
    GM = "gm"
    UB = "ub"


class Pipe(str, Enum):
    MTE1 = "PIPE_MTE1"
    MTE2 = "PIPE_MTE2"
    V = "PIPE_V"
    MTE3 = "PIPE_MTE3"
    ALL = "PIPE_ALL"


class Event(str, Enum):
    ID0 = "EVENT_ID0"
    ID1 = "EVENT_ID1"
    ID2 = "EVENT_ID2"
    ID3 = "EVENT_ID3"
    ID4 = "EVENT_ID4"
    ID5 = "EVENT_ID5"
    ID6 = "EVENT_ID6"
    ID7 = "EVENT_ID7"
    ID8 = "EVENT_ID8"
    ID9 = "EVENT_ID9"
    ID10 = "EVENT_ID10"
    ID11 = "EVENT_ID11"
    ID12 = "EVENT_ID12"
    ID13 = "EVENT_ID13"
    ID14 = "EVENT_ID14"
    ID15 = "EVENT_ID15"
    ID16 = "EVENT_ID16"
    ID17 = "EVENT_ID17"
    ID18 = "EVENT_ID18"
    ID19 = "EVENT_ID19"
    ID20 = "EVENT_ID20"
    ID21 = "EVENT_ID21"
    ID22 = "EVENT_ID22"
    ID23 = "EVENT_ID23"
    ID24 = "EVENT_ID24"
    ID25 = "EVENT_ID25"
    ID26 = "EVENT_ID26"
    ID27 = "EVENT_ID27"
    ID28 = "EVENT_ID28"
    ID29 = "EVENT_ID29"
    ID30 = "EVENT_ID30"
    ID31 = "EVENT_ID31"


class BarrierType(str, Enum):
    VV_ALL = "VV_ALL"
    VST_VLD = "VST_VLD"
    VLD_VST = "VLD_VST"


class MaskPattern(str, Enum):
    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    EVEN = "PAT_EVEN"
    ODD = "PAT_ODD"
    VL16 = "PAT_VL16"
    VL32 = "PAT_VL32"


class PadMode(str, Enum):
    PadNull = "PadNull"
    PadFirstElem = "PadFirstElem"
    PadValue = "PadValue"


class DeinterleaveDist(str, Enum):
    DINTLV = "DINTLV"
    BDINTLV = "BDINTLV"
    B8 = "DINTLV"
    B16 = "DINTLV"
    B32 = "DINTLV"
    BD = "BDINTLV"


class InterleaveDist(str, Enum):
    INTLV = "INTLV"
    B8 = "INTLV"
    B16 = "INTLV"
    B32 = "INTLV"


class PositionMode(str, Enum):
    LOWEST = "LOWEST"
    HIGHEST = "HIGHEST"


class OrderMode(str, Enum):
    ASC = "ORDER_ASC"


class PostUpdateMode(str, Enum):
    POST_UPDATE = "POST_UPDATE"
    NO_POST_UPDATE = "NO_POST_UPDATE"


@dataclass(frozen=True)
class TileConfig:
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TileConfig":
        return cls(tuple(sorted(mapping.items())))


@dataclass(frozen=True)
class TileSpecialization:
    shape: tuple[int, ...]
    memory_space: MemorySpace
    config: TileConfig | None = None
    valid_shape: tuple[int | None, ...] | None = None


i1 = ScalarType("i1")
i8 = ScalarType("i8")
si8 = ScalarType("si8")
ui8 = ScalarType("ui8")
i16 = ScalarType("i16")
si16 = ScalarType("si16")
ui16 = ScalarType("ui16")
i32 = ScalarType("i32")
si32 = ScalarType("si32")
ui32 = ScalarType("ui32")
i64 = ScalarType("i64")
si64 = ScalarType("si64")
ui64 = ScalarType("ui64")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
f32 = ScalarType("f32")
PIPE = Pipe
EVENT = Event
PAT = MaskPattern
AnyFloat = WildcardType("AnyFloat")
AnyInt = WildcardType("AnyInt")
AnyType = WildcardType("AnyType")
AnyMask = WildcardType("AnyMask")
mask_b8 = MaskType("b8")
mask_b16 = MaskType("b16")
mask_b32 = MaskType("b32")
align = AlignType()


def TypeVar(name: str) -> TypeVariable:
    if not isinstance(name, str) or not name:
        raise TypeError("TypeVar name must be a non-empty string")
    return TypeVariable(name)


def ptr(dtype: ScalarType, memory_space: MemorySpace) -> PointerType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("ptr() expects a TileLang scalar dtype")
    if not isinstance(memory_space, MemorySpace):
        raise TypeError("ptr() expects a TileLang MemorySpace")
    return PointerType(element_dtype=dtype, memory_space=memory_space)


def vreg(dtype: ScalarType) -> VRegType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("vreg() expects a TileLang scalar dtype")
    return VRegType(element_dtype=dtype, lanes=get_lanes(dtype))


def integer_bitwidth(dtype: ScalarType) -> int | None:
    if not isinstance(dtype, ScalarType):
        return None
    return _INTEGER_DTYPE_WIDTHS.get(dtype.name)


def integer_signedness(dtype: ScalarType) -> str | None:
    if not isinstance(dtype, ScalarType):
        return None
    return _INTEGER_DTYPE_SIGNS.get(dtype.name)


def is_integer_dtype(dtype: ScalarType) -> bool:
    return integer_bitwidth(dtype) is not None


def is_float_dtype(dtype: ScalarType) -> bool:
    return isinstance(dtype, ScalarType) and dtype.name in _FLOAT_DTYPE_WIDTHS


def bytewidth(dtype: ScalarType) -> int:
    if not isinstance(dtype, ScalarType):
        raise TypeError("bytewidth expects a TileLang scalar dtype")
    width = _DTYPE_BYTE_WIDTHS.get(dtype.name)
    if width is None:
        raise TypeError(f"dtype `{dtype.name}` is not supported by bytewidth")
    return width


def get_lanes(dtype: ScalarType) -> int:
    return 256 // bytewidth(dtype)


def elements_per_vreg(dtype: ScalarType) -> int:
    return get_lanes(dtype)


def constexpr(value: bool) -> bool:
    return value


__all__ = [
    "ScalarType",
    "WildcardType",
    "TypeVariable",
    "TypeVar",
    "TensorView",
    "PartitionTensorView",
    "Tile",
    "PointerType",
    "VRegType",
    "MaskType",
    "ptr",
    "vreg",
    "MemorySpace",
    "Pipe",
    "Event",
    "PIPE",
    "EVENT",
    "MaskPattern",
    "PAT",
    "BarrierType",
    "PadMode",
    "DeinterleaveDist",
    "InterleaveDist",
    "PositionMode",
    "OrderMode",
    "PostUpdateMode",
    "TileConfig",
    "TileSpecialization",
    "i1",
    "i8",
    "si8",
    "ui8",
    "i16",
    "si16",
    "ui16",
    "i32",
    "si32",
    "ui32",
    "i64",
    "si64",
    "ui64",
    "f16",
    "bf16",
    "f32",
    "AnyFloat",
    "AnyInt",
    "AnyType",
    "AnyMask",
    "mask_b8",
    "mask_b16",
    "mask_b32",
    "constexpr",
    "bytewidth",
    "get_lanes",
    "elements_per_vreg",
]

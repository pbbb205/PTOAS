# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared user-facing diagnostics for PTODSL tracing misuse."""

from __future__ import annotations


class PTODSLTracingMisuseError(TypeError):
    """Raised when authored Python misuses PTODSL runtime values during tracing."""


def _format_source_context(function_name: str | None, source_file: str | None, source_line: int | None) -> str:
    details = []
    if function_name:
        details.append(f"kernel {function_name!r}")
    if source_file is not None:
        location = source_file
        if source_line is not None:
            location = f"{location}:{source_line}"
        details.append(location)
    if not details:
        return ""
    return f" ({', '.join(details)})"


def native_python_control_flow_error(usage: str) -> PTODSLTracingMisuseError:
    """Return one actionable diagnostic for native Python control-flow misuse."""
    return PTODSLTracingMisuseError(
        f"native Python {usage} cannot consume a PTODSL runtime value during tracing. "
        "This value is a device-side SSA/runtime-metadata value, not a Python bool/int. "
        "Use pto.if_(...) or pto.for_(...) for device-side control flow, or keep the "
        "bound/condition in pto.constexpr."
    )


def host_tensor_metadata_error(message: str, *, param_name: str | None = None) -> TypeError:
    """Return one actionable diagnostic for unsupported host-tensor metadata."""
    prefix = "host tensor metadata is incomplete or unsupported"
    if param_name is not None:
        prefix = f"@pto.jit host tensor '{param_name}' metadata is incomplete or unsupported"
    return TypeError(f"{prefix}: {message}")


def subkernel_host_tensor_boundary_error(role: str, name: str) -> TypeError:
    """Return one diagnostic for host-tensor usage outside the JIT boundary."""
    return TypeError(
        f"@pto.{role} parameter '{name}' uses a host tensor value, but host tensors only belong "
        "at the @pto.jit boundary. Pass PTODSL device-side values such as Tile, "
        "PartitionTensorView, typed pointers, or PTO scalars instead."
    )


def subkernel_signature_boundary_error(role: str, name: str) -> TypeError:
    """Return one diagnostic for illegal host-tensor formal annotations on a subkernel."""
    return TypeError(
        f"@pto.{role} parameter '{name}' cannot be annotated with pto.tensor_spec(...). "
        "Host tensors are only valid as @pto.jit positional parameters."
    )


def illegal_subkernel_placement_error(role: str, outer_role: str | None) -> RuntimeError:
    """Return one diagnostic for a subkernel call placed outside the supported layer graph."""
    if role == "simt":
        return RuntimeError(
            "@pto.simt helper materialization is only supported from the top-level @pto.jit body; "
            f"it cannot be materialized inside @pto.{outer_role}."
        )
    return RuntimeError(
        f"@pto.{role} may only be called from the top-level @pto.jit body; "
        f"nested invocation inside @pto.{outer_role} is not part of the PTODSL layer contract."
    )


def illegal_inline_subkernel_placement_error(role: str, outer_role: str | None) -> RuntimeError:
    """Return one diagnostic for an inline subkernel scope placed outside the supported layer graph."""
    return RuntimeError(
        f"inline pto.{role}() may only be used from the top-level @pto.jit body; "
        f"nested use inside @pto.{outer_role} is not part of the PTODSL layer contract."
    )


def simd_value_escape_error(type_text: str) -> RuntimeError:
    """Return one diagnostic for transient SIMD values escaping a simd subkernel boundary."""
    return RuntimeError(
        f"@pto.simd cannot return transient SIMD values across the subkernel boundary "
        f"(got {type_text}). Write the value back to a Tile/UB buffer instead."
    )


def tile_row_alignment_error(*, shape, dtype, row_bytes: int, required_alignment: int) -> TypeError:
    """Return one diagnostic for authored tile shapes violating row-byte alignment."""
    return TypeError(
        "alloc_tile(shape=...) physical row layout is invalid for the current PTODSL tile contract: "
        f"shape={list(shape)!r} with dtype={dtype!r} gives a row byte size of {row_bytes}, "
        f"but row-major none-box tiles must be {required_alignment}-byte aligned. "
        "For logical column tiles such as [Br, 1], prefer blayout='ColMajor' instead of authoring them "
        "as row-major narrow tiles. If row-major is truly required, keep the physical tile shape explicitly "
        "aligned and express the logical tail with valid_shape=[...]."
    )


def explicit_mode_required_error(surface: str, current_mode: str | None) -> RuntimeError:
    """Return one diagnostic for explicit-only surfaces used outside explicit mode."""
    observed_mode = "unknown" if current_mode is None else current_mode
    return RuntimeError(
        f"{surface} is an auto-mode contract violation: it is only available in "
        f'@pto.jit(mode="explicit"); current kernel mode is {observed_mode!r}. '
        "Move the kernel to explicit mode before authoring this surface."
    )


def explicit_mode_required_with_context_error(surface: str, module_spec) -> RuntimeError:
    """Return one diagnostic for explicit-only surfaces used outside explicit mode with source context."""
    observed_mode = getattr(module_spec, "mode", None)
    context = _format_source_context(
        getattr(module_spec, "function_name", None),
        getattr(module_spec, "source_file", None),
        getattr(module_spec, "source_line", None),
    )
    observed_mode = "unknown" if observed_mode is None else observed_mode
    return RuntimeError(
        f"{surface} is an auto-mode contract violation{context}: it is only available in "
        f'@pto.jit(mode="explicit"); current kernel mode is {observed_mode!r}. '
        "Move the kernel to explicit mode before authoring this surface."
    )


def invalid_jit_mode_error(
    mode: str,
    *,
    function_name: str | None = None,
    source_file: str | None = None,
    source_line: int | None = None,
) -> ValueError:
    """Return one diagnostic for unsupported ``@pto.jit(mode=...)`` values."""
    context = _format_source_context(function_name, source_file, source_line)
    return ValueError(
        f"unsupported PTODSL jit mode {mode!r}{context}; expected 'auto' or 'explicit'"
    )


def removed_ukernel_surface_error() -> AttributeError:
    """Return one diagnostic for the removed ``pto.ukernel`` public surface."""
    return AttributeError(
        'pto.ukernel has been removed from the PTODSL public surface. '
        'Use @pto.jit(mode="explicit") for explicit DMA orchestration, and call or inline '
        "@pto.simd/@pto.simt/@pto.cube directly from that kernel."
    )


__all__ = [
    "PTODSLTracingMisuseError",
    "explicit_mode_required_error",
    "explicit_mode_required_with_context_error",
    "host_tensor_metadata_error",
    "illegal_inline_subkernel_placement_error",
    "illegal_subkernel_placement_error",
    "invalid_jit_mode_error",
    "native_python_control_flow_error",
    "removed_ukernel_surface_error",
    "simd_value_escape_error",
    "subkernel_host_tensor_boundary_error",
    "subkernel_signature_boundary_error",
    "tile_row_alignment_error",
]

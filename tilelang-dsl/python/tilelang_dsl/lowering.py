# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Authoring-form VPTO lowering skeleton for TileLang DSL v1."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .semantic import (
    SemanticAlignStoreStmt,
    SemanticAlignType,
    SemanticAssignStmt,
    SemanticAttributeAccess,
    SemanticBinaryExpr,
    SemanticBindingRef,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaLoadStmt,
    SemanticDmaStoreStmt,
    SemanticExpr,
    SemanticExprStmt,
    SemanticForStmt,
    SemanticGetBufStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticIfResult,
    SemanticKernel,
    SemanticLiteralExpr,
    SemanticMemBarStmt,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticMetaType,
    SemanticPipeBarrierStmt,
    SemanticPredicateStoreStmt,
    SemanticPtrType,
    SemanticReturnStmt,
    SemanticRlsBufStmt,
    SemanticScalarStoreStmt,
    SemanticScalarType,
    SemanticSetCrossCoreStmt,
    SemanticSetFlagStmt,
    SemanticSetIntraBlockStmt,
    SemanticSetIntraCoreStmt,
    SemanticShapeType,
    SemanticStmt,
    SemanticVecscopeStmt,
    SemanticStrictVecscopeStmt,
    SemanticSubscriptAccess,
    SemanticSymbolExpr,
    SemanticTensorSliceExpr,
    SemanticTensorViewType,
    SemanticPartitionTensorViewType,
    SemanticTileType,
    SemanticType,
    SemanticTupleExpr,
    SemanticTupleType,
    SemanticVRegType,
    SemanticVectorPairStoreStmt,
    SemanticVectorStoreStmt,
    SemanticWaitFlagDevStmt,
    SemanticWaitFlagStmt,
    SemanticWaitIntraCoreStmt,
)
from .types import (
    MaskPattern,
    ScalarType,
    bytewidth,
    get_lanes,
    integer_bitwidth,
    integer_signedness,
    is_integer_dtype,
)


_I1_TYPE = SemanticScalarType(dtype=ScalarType("i1"))
_I32_TYPE = SemanticScalarType(dtype=ScalarType("i32"))
_I64_TYPE = SemanticScalarType(dtype=ScalarType("i64"))


def _format_symbol_name(symbol_name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$.]*", symbol_name):
        return f"@{symbol_name}"
    escaped = symbol_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}"'


@dataclass(frozen=True)
class AuthoringModule:
    """Lowering result that owns authoring-form VPTO text emission."""

    kernel: SemanticKernel

    def render(self) -> str:
        kernel_text = _AuthoringRenderer(self.kernel).render()
        if not self.kernel.inline_helpers:
            return kernel_text

        base_lines = kernel_text.splitlines()
        module_close_index = max(
            (index for index, line in enumerate(base_lines) if line == "}"),
            default=-1,
        )
        if module_close_index < 0:
            return kernel_text

        merged_lines = base_lines[:module_close_index]
        for helper in self.kernel.inline_helpers:
            helper_lines = _extract_single_function_lines(
                _AuthoringRenderer(helper).render()
            )
            if not helper_lines:
                continue
            helper_lines[0] = _rewrite_inline_helper_attrs(helper_lines[0])
            merged_lines.extend(helper_lines)

        merged_lines.append("}")
        merged_lines.append("")
        return "\n".join(merged_lines)


def _extract_single_function_lines(rendered_text: str) -> list[str]:
    lines = rendered_text.splitlines()
    try:
        function_start = next(
            index for index, line in enumerate(lines) if line.lstrip().startswith("func.func ")
        )
    except StopIteration:
        return []
    module_close_index = max(
        (index for index, line in enumerate(lines) if line == "}"),
        default=-1,
    )
    if module_close_index <= function_start:
        return []
    return lines[function_start:module_close_index]


def _rewrite_inline_helper_attrs(function_line: str) -> str:
    kernel_attr = "attributes { pto.tilelang.instance }"
    helper_attr = 'attributes { sym_visibility = "private", pto.tilelang.inline_proc }'
    if kernel_attr in function_line:
        return function_line.replace(kernel_attr, helper_attr)
    if "attributes {" in function_line:
        return function_line
    if function_line.rstrip().endswith("{"):
        stripped = function_line.rstrip()
        return stripped[:-1] + f" {helper_attr} {{"
    return function_line


@dataclass(frozen=True)
class _RenderedValue:
    name: str
    type: SemanticType


@dataclass(frozen=True)
class _RenderedTextualType(SemanticType):
    text: str


@dataclass(frozen=True)
class _DmaTransferConfig:
    n_burst: _RenderedValue
    len_burst: _RenderedValue
    copy_src_stride: _RenderedValue
    copy_dst_stride: _RenderedValue
    loop_src_stride: _RenderedValue
    loop_dst_stride: _RenderedValue


@dataclass(frozen=True)
class _DmaLoadPaddingProfile:
    pad_mode_name: str
    left_padding: int
    right_padding: int
    init_out_buffer: bool
    pad_value: SemanticExpr | None


@dataclass(frozen=True)
class _DmaStoreTrimProfile:
    left_padding: int
    right_padding: int


class _AuthoringRenderer:
    def __init__(self, kernel: SemanticKernel):
        self.kernel = kernel
        self._constant_lines: list[str] = []
        self._constant_cache: dict[tuple[str, object], str] = {}
        self._castptr_cache: dict[tuple[str, str], str] = {}
        self._tile_memref_cache: dict[str, _RenderedValue] = {}
        self._tile_valid_dim_cache: dict[tuple[str, int], _RenderedValue] = {}
        self._used_tile_buffers = self._collect_used_tile_buffers(kernel.body)
        self._temp_counter = 0
        self._loop_counter = 0

    def render(self) -> str:
        parameter_list = ", ".join(
            f"{param.ssa_name}: {self._render_type(param.type)}"
            for param in self.kernel.parameters
            if param.kind != "tile_valid_shape"
        )
        env = {
            param.name: _RenderedValue(name=param.ssa_name, type=param.type)
            for param in self.kernel.parameters
            if param.kind != "tile_valid_shape"
        }
        entry_lines: list[str] = []
        for param in self.kernel.parameters:
            if param.kind != "tile":
                continue
            if param.name in self._used_tile_buffers:
                self._materialize_tile_memref(
                    env[param.name],
                    indent=4,
                    into=entry_lines,
                )
        body_lines = self._render_block(self.kernel.body, env, indent=4)

        lines = [
            f"// tilelang.target = {self.kernel.target}",
            f"// tilelang.op = {self.kernel.op}",
            f"// tilelang.dtypes = {self.kernel.dtype_signature}",
            f"// tilelang.verify = {self.kernel.verify_enabled}",
            f"// tilelang.advanced = {self.kernel.advanced_enabled}",
        ]
        for binding in self.kernel.tile_bindings:
            valid_shape = ""
            if binding.valid_shape is not None:
                valid_shape = f" valid_shape={self._format_shape_tuple(binding.valid_shape)}"
            lines.append(
                "// tilelang.specialize "
                f"{binding.name} shape={binding.shape} memory_space={binding.memory_space} "
                f"config={binding.config}{valid_shape}"
            )
        lines.append(f'module attributes {{pto.target_arch = "{self.kernel.target}"}} {{')
        lines.append(
            "  func.func "
            f"{_format_symbol_name(self.kernel.symbol_name)}({parameter_list}) "
            "attributes { pto.tilelang.instance } {"
        )
        lines.extend(self._constant_lines)
        lines.extend(entry_lines)
        lines.extend(body_lines)
        lines.append("  }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _collect_used_tile_buffers(
        self,
        statements: tuple[SemanticStmt, ...],
    ) -> set[str]:
        used: set[str] = set()
        for stmt in statements:
            self._collect_used_tile_buffers_from_stmt(stmt, used)
        return used

    def _collect_used_tile_buffers_from_stmt(
        self,
        stmt: SemanticStmt,
        used: set[str],
    ) -> None:
        if isinstance(stmt, SemanticAssignStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            return
        if isinstance(stmt, SemanticExprStmt):
            self._collect_used_tile_buffers_from_expr(stmt.expr, used)
            return
        if isinstance(stmt, SemanticDmaLoadStmt):
            self._record_tile_buffer_use(stmt.dst, used)
            self._collect_used_tile_buffers_from_expr(stmt.src, used)
            return
        if isinstance(stmt, SemanticDmaStoreStmt):
            self._record_tile_buffer_use(stmt.src, used)
            self._collect_used_tile_buffers_from_expr(stmt.dst, used)
            return
        if isinstance(stmt, SemanticVectorStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            self._collect_used_tile_buffers_from_expr(stmt.mask, used)
            return
        if isinstance(stmt, SemanticPredicateStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            self._collect_used_tile_buffers_from_expr(stmt.dist, used)
            return
        if isinstance(stmt, SemanticAlignStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            if stmt.offset is not None:
                self._collect_used_tile_buffers_from_expr(stmt.offset, used)
            return
        if isinstance(stmt, SemanticVecscopeStmt):
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticStrictVecscopeStmt):
            for capture in stmt.captures:
                self._record_tile_buffer_use(capture, used)
                self._collect_used_tile_buffers_from_expr(capture, used)
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticForStmt):
            self._collect_used_tile_buffers_from_expr(stmt.lower_bound, used)
            self._collect_used_tile_buffers_from_expr(stmt.upper_bound, used)
            self._collect_used_tile_buffers_from_expr(stmt.step, used)
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticIfStmt):
            self._collect_used_tile_buffers_from_expr(stmt.condition, used)
            for nested in stmt.then_body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            for nested in stmt.else_body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticReturnStmt) and stmt.value is not None:
            self._collect_used_tile_buffers_from_expr(stmt.value, used)

    def _collect_used_tile_buffers_from_expr(
        self,
        expr: SemanticExpr,
        used: set[str],
    ) -> None:
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace == "pto" and expr.name in {"vlds", "vldas", "vldus"} and expr.args:
                self._record_tile_buffer_use(expr.args[0], used)
            for arg in expr.args:
                self._collect_used_tile_buffers_from_expr(arg, used)
            return
        if isinstance(expr, SemanticBinaryExpr):
            self._collect_used_tile_buffers_from_expr(expr.lhs, used)
            self._collect_used_tile_buffers_from_expr(expr.rhs, used)
            return
        if isinstance(expr, SemanticTupleExpr):
            for element in expr.elements:
                self._collect_used_tile_buffers_from_expr(element, used)
            return
        if isinstance(expr, SemanticTensorSliceExpr):
            self._collect_used_tile_buffers_from_expr(expr.base, used)
            for slice_expr in expr.slices:
                if slice_expr.start is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.start, used)
                if slice_expr.stop is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.stop, used)
                if slice_expr.step is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.step, used)
            return
        if isinstance(expr, SemanticAttributeAccess):
            if expr.attr not in {"shape", "valid_shape", "strides", "element_type"}:
                self._collect_used_tile_buffers_from_expr(expr.base, used)
            return
        if isinstance(expr, SemanticSubscriptAccess):
            self._collect_used_tile_buffers_from_expr(expr.base, used)
            self._collect_used_tile_buffers_from_expr(expr.index, used)

    def _record_tile_buffer_use(
        self,
        expr: SemanticExpr,
        used: set[str],
    ) -> None:
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticTileType):
            used.add(expr.binding.name)

    def _render_block(
        self,
        statements: tuple[SemanticStmt, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_stmt(stmt, env, indent=indent))
        return lines

    def _render_stmt(
        self,
        stmt: SemanticStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if isinstance(stmt, SemanticAssignStmt):
            return self._render_assign(stmt, env, indent=indent)
        if isinstance(stmt, SemanticExprStmt):
            lines: list[str] = []
            self._lower_expr(stmt.expr, env, indent=indent, into=lines)
            return lines
        if isinstance(stmt, SemanticDmaLoadStmt):
            return self._render_dma_load(stmt, env, indent=indent)
        if isinstance(stmt, SemanticDmaStoreStmt):
            return self._render_dma_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticVectorStoreStmt):
            return self._render_vector_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticVectorPairStoreStmt):
            return self._render_vector_pair_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticPredicateStoreStmt):
            return self._render_predicate_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticAlignStoreStmt):
            return self._render_align_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticScalarStoreStmt):
            return self._render_scalar_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticSetFlagStmt):
            return [
                self._indent(indent)
                + f'pto.set_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticWaitFlagStmt):
            return [
                self._indent(indent)
                + f'pto.wait_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticPipeBarrierStmt):
            return [self._indent(indent) + f"pto.barrier #pto.pipe<{stmt.pipe}>"]
        if isinstance(stmt, SemanticGetBufStmt):
            return self._render_buffer_sync_stmt("get_buf", stmt.pipe, stmt.buf_id, stmt.mode, env, indent=indent)
        if isinstance(stmt, SemanticRlsBufStmt):
            return self._render_buffer_sync_stmt("rls_buf", stmt.pipe, stmt.buf_id, stmt.mode, env, indent=indent)
        if isinstance(stmt, SemanticMemBarStmt):
            return [self._indent(indent) + f'pto.mem_bar "{stmt.barrier_type}"']
        if isinstance(stmt, SemanticSetCrossCoreStmt):
            return self._render_i64_pair_stmt("set_cross_core", stmt.core_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticSetIntraBlockStmt):
            return self._render_i64_pair_stmt("set_intra_block", stmt.block_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticSetIntraCoreStmt):
            return self._render_i32_stmt("set_intra_core", stmt.config, env, indent=indent)
        if isinstance(stmt, SemanticWaitFlagDevStmt):
            return self._render_i64_pair_stmt("wait_flag_dev", stmt.core_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticWaitIntraCoreStmt):
            return self._render_i64_pair_stmt("wait_intra_core", stmt.block_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticDmaConfigStmt):
            return self._render_dma_config(stmt, env, indent=indent)
        if isinstance(stmt, SemanticLowLevelCopyStmt):
            return self._render_low_level_copy(stmt, env, indent=indent)
        if isinstance(stmt, SemanticReturnStmt):
            lines: list[str] = []
            if stmt.value is None:
                return [self._indent(indent) + "return"]
            value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
            lines.append(self._indent(indent) + f"return {value.name} : {self._render_type(value.type)}")
            return lines
        if isinstance(stmt, SemanticVecscopeStmt):
            return self._render_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticStrictVecscopeStmt):
            return self._render_strict_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticForStmt):
            return self._render_for(stmt, env, indent=indent)
        if isinstance(stmt, SemanticIfStmt):
            return self._render_if(stmt, env, indent=indent)
        raise ValueError(f"unsupported semantic statement {type(stmt).__name__}")

    def _render_dma_config(
        self,
        stmt: SemanticDmaConfigStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        first = self._lower_to_i64(stmt.first, env, indent=indent, into=lines)
        second = self._lower_to_i64(stmt.second, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {first.name}, {second.name} : i64, i64"
        )
        return lines

    def _render_buffer_sync_stmt(
        self,
        name: str,
        pipe: str,
        buf_id: SemanticExpr,
        mode: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_buf_id = self._lower_to_i64(buf_id, env, indent=indent, into=lines)
        rendered_mode = self._lower_to_i64(mode, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f'pto.{name} "{pipe}", {rendered_buf_id.name}, {rendered_mode.name} : i64, i64'
        )
        return lines

    def _render_i64_pair_stmt(
        self,
        name: str,
        first: SemanticExpr,
        second: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_first = self._lower_to_i64(first, env, indent=indent, into=lines)
        rendered_second = self._lower_to_i64(second, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{name} {rendered_first.name}, {rendered_second.name} : i64, i64"
        )
        return lines

    def _render_i32_stmt(
        self,
        name: str,
        value: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_value = self._lower_to_i32(value, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{name} {rendered_value.name} : i32"
        )
        return lines

    def _render_low_level_copy(
        self,
        stmt: SemanticLowLevelCopyStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        source = self._lower_expr(stmt.source, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)

        rendered_operands = []
        rendered_types = []
        for index, operand in enumerate(stmt.operands):
            if stmt.name == "copy_gm_to_ubuf" and index == 5:
                lowered = self._lower_to_i1(operand, env, indent=indent, into=lines)
            else:
                lowered = self._lower_to_i64(operand, env, indent=indent, into=lines)
            rendered_operands.append(lowered.name)
            rendered_types.append(self._render_type(lowered.type))

        operand_text = ", ".join([source.name, destination.name, *rendered_operands])
        type_text = ", ".join(
            [self._render_type(source.type), self._render_type(destination.type), *rendered_types]
        )
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {operand_text} : {type_text}"
        )
        return lines

    def _render_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if len(stmt.targets) != 1:
            if isinstance(stmt.value, SemanticTupleExpr) or (
                isinstance(stmt.value, SemanticAttributeAccess)
                and isinstance(stmt.value.type, SemanticShapeType)
            ):
                return self._render_tuple_expr_assign(stmt, env, indent=indent)
            return self._render_multi_result_assign(stmt, env, indent=indent)
        target = stmt.targets[0]
        if isinstance(target.type, SemanticMetaType):
            env[target.name] = _RenderedValue(name=target.ssa_name, type=target.type)
            return []
        lines: list[str] = []
        lowered = self._lower_expr(
            stmt.value,
            env,
            indent=indent,
            desired_name=target.ssa_name,
            into=lines,
        )
        env[target.name] = lowered
        return lines

    def _render_tuple_expr_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if isinstance(stmt.value, SemanticTupleExpr):
            elements = stmt.value.elements
        elif isinstance(stmt.value, SemanticAttributeAccess) and isinstance(stmt.value.type, SemanticShapeType):
            elements = tuple(
                SemanticSubscriptAccess(
                    base=stmt.value,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
                for axis in range(stmt.value.type.rank)
            )
        else:
            raise NotImplementedError(
                "tuple expression assignment expects a SemanticTupleExpr or shape-like attribute value"
            )
        if len(stmt.targets) != len(elements):
            raise NotImplementedError("tuple expression assignment arity mismatch")

        lines: list[str] = []
        for target, element in zip(stmt.targets, elements):
            lowered = self._lower_expr(
                element,
                env,
                indent=indent,
                desired_name=target.ssa_name,
                into=lines,
            )
            env[target.name] = lowered
        return lines

    def _render_multi_result_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if not isinstance(stmt.value, SemanticCallExpr):
            raise NotImplementedError("multi-result assignment expects a call expression in TileLang DSL v1")
        if stmt.value.namespace != "pto":
            raise NotImplementedError(
                f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
            )
        if len(stmt.targets) != 2:
            raise NotImplementedError("multi-result lowering expects exactly two assignment targets")
        if not isinstance(stmt.value.type, SemanticTupleType) or len(stmt.value.type.elements) != 2:
            raise NotImplementedError("multi-result lowering expects a two-result tuple type")

        if stmt.value.name == "make_mask":
            dtype_expr, remaining_expr = stmt.value.args
            if not self._is_dtype_meta_expr(dtype_expr):
                raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")

            lines: list[str] = []
            remaining = self._lower_remaining_to_i32(remaining_expr, env, indent=indent, into=lines)
            mask_target, remaining_target = stmt.targets
            mask_type, remaining_type = stmt.value.type.elements
            suffix = self._mask_suffix(mask_type)
            lines.append(
                self._indent(indent)
                + f"{mask_target.ssa_name}, {remaining_target.ssa_name} = pto.plt_{suffix} {remaining.name} : "
                + f"i32 -> {self._render_type(mask_type)}, {self._render_type(remaining_type)}"
            )
            env[mask_target.name] = _RenderedValue(name=mask_target.ssa_name, type=mask_type)
            env[remaining_target.name] = _RenderedValue(name=remaining_target.ssa_name, type=remaining_type)
            return lines

        if stmt.value.name in {"vaddc", "vsubc"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vaddcs", "vsubcs"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            carry_in = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[3], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {carry_in.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, "
                + f"{self._render_type(carry_in.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vintlv", "vdintlv"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name} : {self._render_type(lhs.type)}, {self._render_type(rhs.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vmull":
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.vmull "
                + f"{lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vldsx2":
            lines = []
            source = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=lines)
            index_args = stmt.value.args[1:-1]
            if (
                isinstance(stmt.value.args[0].type, SemanticTileType)
                and stmt.value.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    stmt.value.args[0].type,
                    index_args,
                    env,
                    indent=indent,
                    into=lines,
                )
                rendered_indices = self._materialize_constant(0, SemanticIndexType())
            else:
                rendered_indices = self._render_index_list(index_args, env, indent=indent, into=lines)
            dist = self._render_string_literal(stmt.value.args[-1])
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.vldsx2 "
                + f"{source.name}[{rendered_indices}], {dist} : "
                + f"{self._render_type(source.type)}, index -> "
                + f"{self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vldus":
            lines = []
            source = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            index_args = stmt.value.args[1:-1]
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=lines)
            if (
                isinstance(stmt.value.args[0].type, SemanticTileType)
                and stmt.value.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    stmt.value.args[0].type,
                    index_args,
                    env,
                    indent=indent,
                    into=lines,
                )
            if self._is_memref_like_type(source.type):
                ptr_name, ptr_type = self._materialize_copy_buffer_ptr(source, indent=indent, into=lines)
                source = _RenderedValue(name=ptr_name, type=_RenderedTextualType(ptr_type))
            align = self._lower_expr(stmt.value.args[-1], env, indent=indent, into=lines)
            result_target, align_target = stmt.targets
            result_type, align_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {align_target.ssa_name} = pto.vldus "
                + f"{source.name}, {align.name} : "
                + f"{self._render_type(source.type)}, {self._render_type(align.type)} -> "
                + f"{self._render_type(result_type)}, {self._render_type(align_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[align_target.name] = _RenderedValue(name=align_target.ssa_name, type=align_type)
            return lines

        if stmt.value.name == "pstu":
            lines = []
            align_in = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            value = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            base = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            align_target, base_target = stmt.targets
            align_type, base_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{align_target.ssa_name}, {base_target.ssa_name} = pto.pstu "
                + f"{align_in.name}, {value.name}, {base.name} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(align_type)}, {self._render_type(base_type)}"
            )
            env[align_target.name] = _RenderedValue(name=align_target.ssa_name, type=align_type)
            env[base_target.name] = _RenderedValue(name=base_target.ssa_name, type=base_type)
            return lines

        raise NotImplementedError(
            f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
        )

    def _render_dma_load(
        self,
        stmt: SemanticDmaLoadStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        profile = self._resolve_dma_load_padding_profile(stmt.options)
        src = self._lower_expr(stmt.src.base, env, indent=indent, into=lines)
        dst = self._lower_expr(stmt.dst, env, indent=indent, into=lines)
        src_name, src_type = self._materialize_tensor_slice_ptr(
            stmt.src,
            src,
            env,
            indent=indent,
            into=lines,
        )
        dst_name, dst_type = self._materialize_tile_window_ptr(
            dst,
            col_offset=profile.left_padding,
            indent=indent,
            into=lines,
        )
        transfer = self._infer_dma_load_transfer(stmt.src, stmt.dst.type, src, env, indent=indent, into=lines)

        copy_lines = self._render_dma_load_copy_ops(
            src_name,
            src_type,
            dst_name,
            dst_type,
            transfer,
            indent=indent,
        )
        prefill_lines = self._render_dma_load_prefill(
            stmt.dst,
            dst,
            env,
            profile,
            indent=indent,
        )
        if profile.pad_mode_name == "PadFirstElem":
            lines.extend(copy_lines)
            lines.extend(prefill_lines)
            if profile.init_out_buffer:
                lines.extend(copy_lines)
            return lines

        lines.extend(prefill_lines)
        lines.extend(copy_lines)
        return lines

    def _render_dma_store(
        self,
        stmt: SemanticDmaStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        profile = self._resolve_dma_store_trim_profile(stmt.options)
        src = self._lower_expr(stmt.src, env, indent=indent, into=lines)
        dst = self._lower_expr(stmt.dst.base, env, indent=indent, into=lines)
        src_name, src_type = self._materialize_tile_window_ptr(
            src,
            col_offset=profile.left_padding,
            indent=indent,
            into=lines,
        )
        dst_name, dst_type = self._materialize_tensor_slice_ptr(
            stmt.dst,
            dst,
            env,
            indent=indent,
            into=lines,
        )
        transfer = self._infer_dma_store_transfer(stmt.dst, stmt.src.type, dst, env, indent=indent, into=lines)

        c0_i64 = self._materialize_constant(0, _I64_TYPE)
        c1_i64 = self._materialize_constant(1, _I64_TYPE)

        lines.extend(
            [
                self._indent(indent)
                + f"pto.set_loop_size_ubtoout {c1_i64}, {c1_i64} : i64, i64",
                self._indent(indent)
                + f"pto.set_loop1_stride_ubtoout {transfer.loop_src_stride.name}, {transfer.loop_dst_stride.name} : i64, i64",
                self._indent(indent)
                + f"pto.set_loop2_stride_ubtoout {transfer.loop_src_stride.name}, {transfer.loop_dst_stride.name} : i64, i64",
                self._indent(indent)
                + "pto.copy_ubuf_to_gm "
                + f"{src_name}, {dst_name}, {c0_i64}, {transfer.n_burst.name}, {transfer.len_burst.name}, {c0_i64}, "
                + f"{transfer.copy_dst_stride.name}, {transfer.copy_src_stride.name} : {src_type}, {dst_type}, "
                + "i64, i64, i64, i64, i64, i64",
            ]
        )
        return lines

    def _render_vector_store(
        self,
        stmt: SemanticVectorStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_indices = self._materialize_constant(0, SemanticIndexType())
        else:
            rendered_indices = self._render_index_list(stmt.indices, env, indent=indent, into=lines)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + "pto.vsts "
            + f"{value.name}, {destination.name}[{rendered_indices}], {mask.name} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _render_vector_pair_store(
        self,
        stmt: SemanticVectorPairStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        low = self._lower_expr(stmt.low, env, indent=indent, into=lines)
        high = self._lower_expr(stmt.high, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_indices = self._materialize_constant(0, SemanticIndexType())
        else:
            rendered_indices = self._render_index_list(stmt.indices, env, indent=indent, into=lines)
        dist = self._render_string_literal(stmt.dist)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + "pto.vstsx2 "
            + f"{low.name}, {high.name}, {destination.name}[{rendered_indices}], {dist}, {mask.name} : "
            + f"{self._render_type(low.type)}, {self._render_type(high.type)}, "
            + f"{self._render_type(destination.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _render_predicate_store(
        self,
        stmt: SemanticPredicateStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_offset = self._materialize_constant(0, SemanticIndexType())
        else:
            rendered_offset = self._lower_expr(stmt.indices[0], env, indent=indent, into=lines)
        dist = self._render_string_literal(stmt.dist)
        lines.append(
            self._indent(indent)
            + f"pto.psts {value.name}, {destination.name}[{rendered_offset.name}], {dist} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(rendered_offset.type)}"
        )
        return lines

    def _render_align_store(
        self,
        stmt: SemanticAlignStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
        if stmt.op_name == "vstar":
            lines.append(
                self._indent(indent)
                + f"pto.vstar {value.name}, {destination.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(destination.type)}"
            )
            return lines
        if stmt.offset is None:
            raise NotImplementedError("vstas lowering expects an explicit offset operand")
        offset = self._lower_expr(stmt.offset, env, indent=indent, into=lines)
        offset = self._coerce_rendered_value(offset, _I32_TYPE, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.vstas {value.name}, {destination.name}, {offset.name} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(offset.type)}"
        )
        return lines

    def _render_scalar_store(
        self,
        stmt: SemanticScalarStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        offset = self._lower_expr(stmt.offset, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.store_scalar {value.name}, {destination.name}[{offset.name}] : "
            + f"{self._render_type(destination.type)}, {self._render_type(value.type)}"
        )
        return lines

    def _render_index_list(
        self,
        indices: tuple[SemanticExpr, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> str:
        rendered = [
            self._lower_expr(index, env, indent=indent, into=into).name for index in indices
        ]
        return ", ".join(rendered)

    def _render_rank2_subview_result_type(
        self,
        *,
        element_dtype: str,
        memory_space: str,
    ) -> _RenderedTextualType:
        return _RenderedTextualType(
            f"memref<?x?x{element_dtype}, strided<[?, ?], offset: ?>, "
            f"{self._render_memref_memory_space(memory_space)}>"
        )

    def _materialize_rank2_tile_subview(
        self,
        base: _RenderedValue,
        tile_type: SemanticTileType,
        indices: tuple[SemanticExpr, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        row_index, col_index = indices
        row_value = self._lower_expr(row_index, env, indent=indent, into=into)
        col_value = self._lower_expr(col_index, env, indent=indent, into=into)
        one = self._materialize_constant(1, SemanticIndexType())
        total_cols = self._materialize_constant(tile_type.shape[1], SemanticIndexType())
        remaining_cols = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{remaining_cols} = arith.subi {total_cols}, {col_value.name} : index"
        )
        subview_type = self._render_rank2_subview_result_type(
            element_dtype=tile_type.element_dtype.name,
            memory_space=tile_type.memory_space or "ub",
        )
        subview_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{subview_name} = memref.subview {base.name}[{row_value.name}, {col_value.name}] "
            + f"[{one}, {remaining_cols}] [{one}, {one}] : "
            + f"{self._render_type(base.type)} to {self._render_type(subview_type)}"
        )
        return _RenderedValue(name=subview_name, type=subview_type)

    def _tensor_slice_extents(self, expr: SemanticTensorSliceExpr) -> tuple[int, int]:
        if expr.type.rank != 2 or len(expr.type.extents) != 2:
            raise NotImplementedError("TileLang DSL v1 DMA lowering currently only supports rank-2 TensorView slices")
        return expr.type.extents

    def _resolve_dma_load_padding_profile(self, options: object) -> _DmaLoadPaddingProfile:
        pad_mode_name = self._static_pad_mode_name(getattr(options, "pad_mode", None)) or "PadNull"
        left_padding = self._static_expr_value(getattr(options, "left_padding", None), default=0)
        right_padding = self._static_expr_value(getattr(options, "right_padding", None), default=0)
        init_out_buffer = self._static_expr_value(getattr(options, "init_out_buffer", None), default=False)
        if not isinstance(left_padding, int) or left_padding < 0:
            raise NotImplementedError(
                "pto.dma_load lowering currently expects `left_padding` to be a static non-negative index"
            )
        if not isinstance(right_padding, int) or right_padding < 0:
            raise NotImplementedError(
                "pto.dma_load lowering currently expects `right_padding` to be a static non-negative index"
            )
        if not isinstance(init_out_buffer, bool):
            raise NotImplementedError(
                "pto.dma_load lowering currently expects `init_out_buffer` to be a compile-time bool"
            )
        if pad_mode_name not in {"PadNull", "PadFirstElem", "PadValue"}:
            raise NotImplementedError(
                f"pto.dma_load lowering does not recognize pad_mode `{pad_mode_name}` in TileLang DSL v1"
            )
        if pad_mode_name == "PadNull" and init_out_buffer:
            raise NotImplementedError(
                "pto.dma_load lowering does not support `init_out_buffer=True` with `pad_mode=PadMode.PadNull`; "
                "the stable frontend-only path has no explicit fill value for that combination"
            )
        return _DmaLoadPaddingProfile(
            pad_mode_name=pad_mode_name,
            left_padding=left_padding,
            right_padding=right_padding,
            init_out_buffer=init_out_buffer,
            pad_value=getattr(options, "pad_value", None),
        )

    def _resolve_dma_store_trim_profile(self, options: object) -> _DmaStoreTrimProfile:
        pad_mode_name = self._static_pad_mode_name(getattr(options, "pad_mode", None)) or "PadNull"
        left_padding = self._static_expr_value(getattr(options, "left_padding", None), default=0)
        right_padding = self._static_expr_value(getattr(options, "right_padding", None), default=0)
        if pad_mode_name != "PadNull":
            raise NotImplementedError(
                "pto.dma_store lowering only supports `pad_mode=PadMode.PadNull`; "
                "non-PadNull store padding would require GM-side fill in the stable frontend-only path"
            )
        if self._static_expr_value(getattr(options, "pad_value", None)) is not None:
            raise NotImplementedError(
                "pto.dma_store lowering does not support `pad_value`; GM-side fill is unsupported"
            )
        if not isinstance(left_padding, int) or left_padding < 0:
            raise NotImplementedError(
                "pto.dma_store lowering currently expects `left_padding` to be a static non-negative index"
            )
        if not isinstance(right_padding, int) or right_padding < 0:
            raise NotImplementedError(
                "pto.dma_store lowering currently expects `right_padding` to be a static non-negative index"
            )
        return _DmaStoreTrimProfile(
            left_padding=left_padding,
            right_padding=right_padding,
        )

    def _require_default_dma_lowering_profile(self, options: object, op_name: str) -> None:
        if not self._is_default_dma_lowering_profile(options):
            raise NotImplementedError(
                f"{op_name} lowering for padding/trim/init options is not implemented yet in TileLang DSL v1; "
                "this stable frontend-only DMA path only lowers the default no-padding profile today"
            )

    def _is_default_dma_lowering_profile(self, options: object) -> bool:
        return (
            self._static_pad_mode_name(getattr(options, "pad_mode", None)) in {None, "PadNull"}
            and self._static_expr_value(getattr(options, "pad_value", None)) is None
            and self._static_expr_value(getattr(options, "left_padding", None), default=0) == 0
            and self._static_expr_value(getattr(options, "right_padding", None), default=0) == 0
            and self._static_expr_value(getattr(options, "init_out_buffer", None), default=False) is False
        )

    def _static_pad_mode_name(self, expr: SemanticExpr | None) -> str | None:
        value = self._static_expr_value(expr)
        return None if value is None else getattr(value, "name", None)

    def _static_expr_value(self, expr: SemanticExpr | None, *, default: object = None) -> object:
        if expr is None:
            return default
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticSymbolExpr):
            return expr.value
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        return None

    def _infer_dma_load_transfer(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tile_type: SemanticTileType,
        tensor_base: _RenderedValue,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _DmaTransferConfig:
        element_bytes = self._dtype_byte_width(slice_expr.type.element_dtype)
        row_count = self._materialize_dma_axis_extent(slice_expr, 0, env, indent=indent, into=into)
        col_count = self._materialize_dma_axis_extent(slice_expr, 1, env, indent=indent, into=into)
        gm_row_stride = self._materialize_tensor_row_stride_bytes(
            slice_expr,
            tensor_base,
            element_bytes,
            indent=indent,
            into=into,
        )
        row_step = self._materialize_dma_row_step(slice_expr, env, indent=indent, into=into)
        copy_src_stride = self._emit_binary_value(
            "mul",
            gm_row_stride,
            row_step,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        copy_dst_stride = self._materialize_tile_row_stride_bytes(
            tile_type,
            element_bytes,
            indent=indent,
            into=into,
        )
        len_burst = self._materialize_dma_len_burst(
            col_count,
            element_bytes,
            indent=indent,
            into=into,
        )
        loop_src_stride = self._emit_binary_value(
            "mul",
            row_count,
            copy_src_stride,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        loop_dst_stride = self._emit_binary_value(
            "mul",
            row_count,
            copy_dst_stride,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        return _DmaTransferConfig(
            n_burst=row_count,
            len_burst=len_burst,
            copy_src_stride=copy_src_stride,
            copy_dst_stride=copy_dst_stride,
            loop_src_stride=loop_src_stride,
            loop_dst_stride=loop_dst_stride,
        )

    def _infer_dma_store_transfer(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tile_type: SemanticTileType,
        tensor_base: _RenderedValue,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _DmaTransferConfig:
        element_bytes = self._dtype_byte_width(slice_expr.type.element_dtype)
        row_count = self._materialize_dma_axis_extent(slice_expr, 0, env, indent=indent, into=into)
        col_count = self._materialize_dma_axis_extent(slice_expr, 1, env, indent=indent, into=into)
        copy_src_stride = self._materialize_tile_row_stride_bytes(
            tile_type,
            element_bytes,
            indent=indent,
            into=into,
        )
        gm_row_stride = self._materialize_tensor_row_stride_bytes(
            slice_expr,
            tensor_base,
            element_bytes,
            indent=indent,
            into=into,
        )
        row_step = self._materialize_dma_row_step(slice_expr, env, indent=indent, into=into)
        copy_dst_stride = self._emit_binary_value(
            "mul",
            gm_row_stride,
            row_step,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        len_burst = self._materialize_dma_len_burst(
            col_count,
            element_bytes,
            indent=indent,
            into=into,
        )
        loop_src_stride = self._emit_binary_value(
            "mul",
            row_count,
            copy_src_stride,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        loop_dst_stride = self._emit_binary_value(
            "mul",
            row_count,
            copy_dst_stride,
            _I64_TYPE,
            indent=indent,
            into=into,
        )
        return _DmaTransferConfig(
            n_burst=row_count,
            len_burst=len_burst,
            copy_src_stride=copy_src_stride,
            copy_dst_stride=copy_dst_stride,
            loop_src_stride=loop_src_stride,
            loop_dst_stride=loop_dst_stride,
        )

    def _render_dma_load_copy_ops(
        self,
        src_name: str,
        src_type: str,
        dst_name: str,
        dst_type: str,
        transfer: _DmaTransferConfig,
        *,
        indent: int,
    ) -> list[str]:
        c0_i64 = self._materialize_constant(0, _I64_TYPE)
        c1_i64 = self._materialize_constant(1, _I64_TYPE)
        false_bit = self._materialize_constant(False, _I1_TYPE)
        return [
            self._indent(indent)
            + f"pto.set_loop2_stride_outtoub {transfer.loop_src_stride.name}, {transfer.loop_dst_stride.name} : i64, i64",
            self._indent(indent)
            + f"pto.set_loop1_stride_outtoub {transfer.loop_src_stride.name}, {transfer.loop_dst_stride.name} : i64, i64",
            self._indent(indent)
            + f"pto.set_loop_size_outtoub {c1_i64}, {c1_i64} : i64, i64",
            self._indent(indent)
            + "pto.copy_gm_to_ubuf "
            + f"{src_name}, {dst_name}, {c0_i64}, {transfer.n_burst.name}, {transfer.len_burst.name}, {c0_i64}, {c0_i64}, "
            + f"{false_bit}, {c0_i64}, {transfer.copy_src_stride.name}, {transfer.copy_dst_stride.name} : "
            + f"{src_type}, {dst_type}, "
            + "i64, i64, i64, i64, i64, i1, i64, i64, i64",
        ]

    def _render_dma_load_prefill(
        self,
        tile_expr: SemanticExpr,
        tile_value: _RenderedValue,
        env: dict[str, _RenderedValue],
        profile: _DmaLoadPaddingProfile,
        *,
        indent: int,
    ) -> list[str]:
        fill_bands = profile.left_padding > 0 or profile.right_padding > 0
        if profile.pad_mode_name == "PadNull" and not profile.init_out_buffer:
            return []
        if profile.pad_mode_name in {"PadValue", "PadFirstElem"} and not (profile.init_out_buffer or fill_bands):
            return []

        lines: list[str] = []
        tile_memref = self._materialize_tile_memref(tile_value, indent=indent, into=lines)
        rows_upper = self._materialize_tile_window_extent(
            tile_expr,
            tile_value,
            axis=0,
            indent=indent,
            into=lines,
        )
        cols_upper = self._materialize_tile_window_extent(
            tile_expr,
            tile_value,
            axis=1,
            indent=indent,
            into=lines,
        )
        fill_vec = self._materialize_dma_load_prefill_vector(
            tile_memref,
            tile_value.type.element_dtype,
            env,
            profile,
            indent=indent,
            into=lines,
        )

        windows: list[tuple[_RenderedValue, _RenderedValue]] = []
        c0_index = _RenderedValue(
            name=self._materialize_constant(0, SemanticIndexType()),
            type=SemanticIndexType(),
        )
        if profile.init_out_buffer:
            windows.append((c0_index, cols_upper))
        else:
            if profile.left_padding > 0:
                windows.append(
                    (
                        c0_index,
                        _RenderedValue(
                            name=self._materialize_constant(profile.left_padding, SemanticIndexType()),
                            type=SemanticIndexType(),
                        ),
                    )
                )
            if profile.right_padding > 0:
                right_width = _RenderedValue(
                    name=self._materialize_constant(profile.right_padding, SemanticIndexType()),
                    type=SemanticIndexType(),
                )
                right_start = self._emit_binary_value(
                    "sub",
                    cols_upper,
                    right_width,
                    SemanticIndexType(),
                    indent=indent,
                    into=lines,
                )
                windows.append((right_start, cols_upper))

        if not windows:
            return []
        lines.extend(
            self._render_tile_fill_windows(
                tile_memref,
                tile_value.type.element_dtype,
                fill_vec,
                rows_upper,
                windows,
                indent=indent,
            )
        )
        return lines

    def _render_tile_fill_windows(
        self,
        tile_memref: _RenderedValue,
        element_dtype: ScalarType,
        fill_vec: _RenderedValue,
        rows_upper: _RenderedValue,
        windows: list[tuple[_RenderedValue, _RenderedValue]],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        c0 = self._materialize_constant(0, SemanticIndexType())
        c1 = self._materialize_constant(1, SemanticIndexType())
        vector_step = self._materialize_constant(get_lanes(element_dtype), SemanticIndexType())
        mask_type = SemanticMaskType(granularity=self._mask_granularity_for_dtype(element_dtype))
        lines.append(self._indent(indent) + "pto.vecscope {")
        for start, stop in windows:
            row_iv = self._new_temp()
            lines.append(
                self._indent(indent + 2)
                + f"scf.for {row_iv} = {c0} to {rows_upper.name} step {c1} {{"
            )
            col_iv = self._new_temp()
            lines.append(
                self._indent(indent + 4)
                + f"scf.for {col_iv} = {start.name} to {stop.name} step {vector_step} {{"
            )
            remaining = self._emit_binary_value(
                "sub",
                stop,
                _RenderedValue(name=col_iv, type=SemanticIndexType()),
                SemanticIndexType(),
                indent=indent + 6,
                into=lines,
            )
            remaining_i32 = self._coerce_rendered_value(
                remaining,
                _I32_TYPE,
                indent=indent + 6,
                into=lines,
            )
            mask_name = self._new_temp()
            next_name = self._new_temp()
            lines.append(
                self._indent(indent + 6)
                + f"{mask_name}, {next_name} = pto.plt_{mask_type.granularity} {remaining_i32.name} : "
                + f"i32 -> {self._render_type(mask_type)}, i32"
            )
            lines.append(
                self._indent(indent + 6)
                + f"pto.vsts {fill_vec.name}, {tile_memref.name}[{row_iv}, {col_iv}], {mask_name} : "
                + f"{self._render_type(fill_vec.type)}, {self._render_type(tile_memref.type)}, {self._render_type(mask_type)}"
            )
            lines.append(self._indent(indent + 4) + "}")
            lines.append(self._indent(indent + 2) + "}")
        lines.append(self._indent(indent) + "}")
        return lines

    def _materialize_dma_load_prefill_vector(
        self,
        tile_memref: _RenderedValue,
        element_dtype: ScalarType,
        env: dict[str, _RenderedValue],
        profile: _DmaLoadPaddingProfile,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        vec_type = SemanticVRegType(element_dtype=element_dtype, lanes=get_lanes(element_dtype))
        result_name = self._new_temp()
        if profile.pad_mode_name == "PadValue":
            scalar = self._materialize_dma_pad_value_scalar(
                profile.pad_value,
                element_dtype,
                env,
                indent=indent,
                into=into,
            )
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vbr {scalar.name} : {self._render_type(scalar.type)} -> {self._render_type(vec_type)}"
            )
            return _RenderedValue(name=result_name, type=vec_type)
        if profile.pad_mode_name == "PadFirstElem":
            c0 = self._materialize_constant(0, SemanticIndexType())
            first_col = self._materialize_constant(profile.left_padding, SemanticIndexType())
            into.append(
                self._indent(indent)
                + f'{result_name} = pto.vlds {tile_memref.name}[{c0}, {first_col}] {{dist = "{self._broadcast_dist_for_dtype(element_dtype)}"}} : '
                + f"{self._render_type(tile_memref.type)} -> {self._render_type(vec_type)}"
            )
            return _RenderedValue(name=result_name, type=vec_type)
        raise NotImplementedError(
            f"pto.dma_load lowering does not produce a prefill vector for pad_mode `{profile.pad_mode_name}`"
        )

    def _materialize_dma_pad_value_scalar(
        self,
        expr: SemanticExpr | None,
        element_dtype: ScalarType,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        scalar_type = SemanticScalarType(dtype=element_dtype)
        static_value = self._static_expr_value(expr)
        if isinstance(static_value, (int, float)):
            return _RenderedValue(
                name=self._materialize_constant(static_value, scalar_type),
                type=scalar_type,
            )
        if expr is None:
            raise NotImplementedError("pto.dma_load PadValue lowering requires a concrete `pad_value` expression")
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype == element_dtype:
            return value
        raise NotImplementedError(
            "pto.dma_load PadValue lowering currently expects `pad_value` to be a compile-time numeric literal "
            "or a scalar value whose dtype matches the destination Tile element dtype"
        )

    def _materialize_tile_window_extent(
        self,
        tile_expr: SemanticExpr,
        tile_value: _RenderedValue,
        *,
        axis: int,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if (
            isinstance(tile_expr, SemanticBindingRef)
            and isinstance(tile_expr.type, SemanticTileType)
            and tile_expr.type.valid_shape is not None
            and tile_expr.type.valid_shape[axis] is None
        ):
            return self._materialize_tile_valid_dim(
                tile_expr.binding,
                axis,
                indent=indent,
                into=into,
            )
        if not isinstance(tile_value.type, SemanticTileType):
            raise NotImplementedError("DMA load prefill expects a Tile destination")
        valid_shape = tile_value.type.valid_shape or tile_value.type.shape
        if valid_shape is None:
            raise NotImplementedError("DMA load prefill expects a statically known Tile shape or valid_shape")
        extent = valid_shape[axis]
        if extent is None:
            raise NotImplementedError("DMA load prefill does not support dynamic Tile valid_shape on non-binding values")
        return _RenderedValue(
            name=self._materialize_constant(extent, SemanticIndexType()),
            type=SemanticIndexType(),
        )

    def _mask_granularity_for_dtype(self, dtype: ScalarType) -> str:
        int_bits = integer_bitwidth(dtype)
        if dtype.name == "f32" or int_bits == 32:
            return "b32"
        if dtype.name in {"f16", "bf16"} or int_bits == 16:
            return "b16"
        if int_bits == 8:
            return "b8"
        raise NotImplementedError(f"dtype `{dtype.name}` is not supported by DMA load prefill lowering")

    def _broadcast_dist_for_dtype(self, dtype: ScalarType) -> str:
        int_bits = integer_bitwidth(dtype)
        if dtype.name == "f32" or int_bits == 32:
            return "BRC_B32"
        if dtype.name in {"f16", "bf16"} or int_bits == 16:
            return "BRC_B16"
        if int_bits == 8:
            return "BRC_B8"
        raise NotImplementedError(f"dtype `{dtype.name}` is not supported by DMA load broadcast lowering")

    def _materialize_tile_window_ptr(
        self,
        tile_value: _RenderedValue,
        *,
        col_offset: int,
        indent: int,
        into: list[str],
    ) -> tuple[str, str]:
        base_ptr_name, base_ptr_type = self._materialize_copy_buffer_ptr(
            tile_value,
            indent=indent,
            into=into,
        )
        if col_offset == 0:
            return base_ptr_name, base_ptr_type
        byte_ptr_type = "!pto.ptr<i8, ub>"
        byte_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{byte_ptr_name} = pto.castptr {base_ptr_name} : {base_ptr_type} -> {byte_ptr_type}"
        )
        offset_bytes = self._materialize_constant(
            col_offset * self._dtype_byte_width(tile_value.type.element_dtype),
            SemanticIndexType(),
        )
        offset_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{offset_ptr_name} = pto.addptr {byte_ptr_name}, {offset_bytes} : {byte_ptr_type} -> {byte_ptr_type}"
        )
        typed_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{typed_ptr_name} = pto.castptr {offset_ptr_name} : {byte_ptr_type} -> {base_ptr_type}"
        )
        return typed_ptr_name, base_ptr_type

    def _materialize_tensor_slice_ptr(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _RenderedValue,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> tuple[str, str]:
        base_ptr_name, base_ptr_type = self._materialize_copy_buffer_ptr(
            tensor_base,
            indent=indent,
            into=into,
        )
        if self._is_zero_index_expr(slice_expr.slices[0].start) and self._is_zero_index_expr(slice_expr.slices[1].start):
            return base_ptr_name, base_ptr_type

        byte_ptr_type = "!pto.ptr<i8, gm>"
        byte_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{byte_ptr_name} = pto.castptr {base_ptr_name} : {base_ptr_type} -> {byte_ptr_type}"
        )
        offset = self._materialize_tensor_slice_offset_bytes(
            slice_expr,
            tensor_base,
            env,
            indent=indent,
            into=into,
        )
        offset_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{offset_ptr_name} = pto.addptr {byte_ptr_name}, {offset.name} : "
            + f"{byte_ptr_type} -> {byte_ptr_type}"
        )
        typed_ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{typed_ptr_name} = pto.castptr {offset_ptr_name} : {byte_ptr_type} -> {base_ptr_type}"
        )
        return typed_ptr_name, base_ptr_type

    def _materialize_tensor_slice_offset_bytes(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _RenderedValue,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        offset_elems = _RenderedValue(
            name=self._materialize_constant(0, SemanticIndexType()),
            type=SemanticIndexType(),
        )
        for axis_index, slice_axis in enumerate(slice_expr.slices):
            axis_start = self._lower_expr(slice_axis.start, env, indent=indent, into=into)
            axis_stride_elems = self._materialize_tensor_axis_stride_elems(
                tensor_base,
                axis=slice_expr.type.physical_axes[axis_index],
                indent=indent,
                into=into,
            )
            axis_offset_elems = self._emit_binary_value(
                "mul",
                axis_start,
                axis_stride_elems,
                SemanticIndexType(),
                indent=indent,
                into=into,
            )
            offset_elems = self._emit_binary_value(
                "add",
                offset_elems,
                axis_offset_elems,
                SemanticIndexType(),
                indent=indent,
                into=into,
            )
        return self._emit_binary_value(
            "mul",
            offset_elems,
            _RenderedValue(
                name=self._materialize_constant(
                    self._dtype_byte_width(slice_expr.type.element_dtype),
                    SemanticIndexType(),
                ),
                type=SemanticIndexType(),
            ),
            SemanticIndexType(),
            indent=indent,
            into=into,
        )

    def _is_zero_index_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticLiteralExpr):
            return isinstance(expr.value, int) and expr.value == 0
        if isinstance(expr, SemanticBindingRef):
            return isinstance(expr.binding.value, int) and expr.binding.value == 0
        return False

    def _materialize_tensor_dim(
        self,
        tensor_base: _RenderedValue,
        *,
        axis: int,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        dim_index = self._new_temp()
        axis_value = self._materialize_constant(axis, SemanticIndexType())
        if isinstance(tensor_base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            into.append(
                self._indent(indent)
                + f"{dim_index} = pto.get_tensor_view_dim {tensor_base.name}, {axis_value} : "
                + f"{self._render_type(tensor_base.type)} -> index"
            )
        else:
            into.append(
                self._indent(indent)
                + f"{dim_index} = memref.dim {tensor_base.name}, {axis_value} : {self._render_type(tensor_base.type)}"
            )
        return _RenderedValue(name=dim_index, type=SemanticIndexType())

    def _materialize_dma_axis_extent(
        self,
        slice_expr: SemanticTensorSliceExpr,
        axis: int,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        axis_slice = slice_expr.slices[axis]
        if axis_slice.extent is not None:
            return _RenderedValue(
                name=self._materialize_constant(axis_slice.extent, _I64_TYPE),
                type=_I64_TYPE,
            )

        distance_expr = SemanticBinaryExpr(
            lhs=axis_slice.stop,
            op="sub",
            rhs=axis_slice.start,
            type=SemanticIndexType(),
        )
        extent_expr: SemanticExpr = distance_expr
        step_value = self._static_expr_value(axis_slice.step)
        if not isinstance(step_value, int):
            raise NotImplementedError("DMA lowering currently expects a static slice step")
        if step_value != 1:
            extent_expr = SemanticBinaryExpr(
                lhs=SemanticBinaryExpr(
                    lhs=distance_expr,
                    op="add",
                    rhs=SemanticLiteralExpr(value=step_value - 1, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                ),
                op="floordiv",
                rhs=SemanticLiteralExpr(value=step_value, type=SemanticIndexType()),
                type=SemanticIndexType(),
            )
        return self._lower_to_i64(extent_expr, env, indent=indent, into=into)

    def _materialize_dma_row_step(
        self,
        slice_expr: SemanticTensorSliceExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        return self._lower_to_i64(slice_expr.slices[0].step, env, indent=indent, into=into)

    def _materialize_tensor_axis_stride_elems(
        self,
        tensor_base: _RenderedValue,
        axis: int,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        stride = _RenderedValue(
            name=self._materialize_constant(1, SemanticIndexType()),
            type=SemanticIndexType(),
        )
        for dim_axis in range(axis + 1, tensor_base.type.rank):
            dim_value = self._materialize_tensor_dim(
                tensor_base,
                axis=dim_axis,
                indent=indent,
                into=into,
            )
            stride = self._emit_binary_value(
                "mul",
                stride,
                dim_value,
                SemanticIndexType(),
                indent=indent,
                into=into,
            )
        return stride

    def _materialize_tensor_row_stride_bytes(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _RenderedValue,
        element_bytes: int,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        stride_elems = self._materialize_tensor_axis_stride_elems(
            tensor_base,
            axis=slice_expr.type.physical_axes[0],
            indent=indent,
            into=into,
        )
        dim_bytes = self._emit_binary_value(
            "mul",
            stride_elems,
            _RenderedValue(
                name=self._materialize_constant(element_bytes, SemanticIndexType()),
                type=SemanticIndexType(),
            ),
            SemanticIndexType(),
            indent=indent,
            into=into,
        )
        return self._coerce_rendered_to_i64(dim_bytes, indent=indent, into=into)

    def _materialize_tile_row_stride_bytes(
        self,
        tile_type: SemanticTileType,
        element_bytes: int,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if tile_type.shape is None or len(tile_type.shape) != 2:
            raise NotImplementedError("DMA lowering requires a statically specialized rank-2 Tile shape")
        row_bytes = tile_type.shape[1] * element_bytes
        return _RenderedValue(
            name=self._materialize_constant(row_bytes, _I64_TYPE),
            type=_I64_TYPE,
        )

    def _materialize_dma_len_burst(
        self,
        col_count: _RenderedValue,
        element_bytes: int,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        return self._emit_binary_value(
            "mul",
            col_count,
            _RenderedValue(
                name=self._materialize_constant(element_bytes, _I64_TYPE),
                type=_I64_TYPE,
            ),
            _I64_TYPE,
            indent=indent,
            into=into,
        )

    def _dma_transfer_extents(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tile_type: SemanticTileType,
    ) -> tuple[int, int]:
        row_count, col_count = self._tensor_slice_extents(slice_expr)
        if row_count is not None and col_count is not None:
            return row_count, col_count
        if tile_type.shape is None or len(tile_type.shape) != 2:
            raise NotImplementedError("DMA lowering requires a statically specialized rank-2 Tile shape")
        return tile_type.shape

    def _emit_binary_value(
        self,
        op: str,
        lhs: _RenderedValue,
        rhs: _RenderedValue,
        result_type: SemanticType,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        result_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {self._render_binary_op(op, result_type)} "
            f"{lhs.name}, {rhs.name} : {self._render_type(result_type)}"
        )
        return _RenderedValue(name=result_name, type=result_type)

    def _render_strict_vecscope(
        self,
        stmt: SemanticStrictVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        capture_values = []
        block_argument_values = []
        for expr, binding in zip(stmt.captures, stmt.block_arguments):
            capture = self._lower_expr(expr, env, indent=indent, into=lines)
            capture, block_arg = self._materialize_strict_vecscope_capture(
                capture,
                binding,
                indent=indent,
                into=lines,
            )
            capture_values.append(capture)
            block_argument_values.append(block_arg)
        capture_names = ", ".join(value.name for value in capture_values)
        block_args = ", ".join(
            f"{binding.ssa_name}: {self._render_type(value.type)}"
            for binding, value in zip(stmt.block_arguments, block_argument_values)
        )
        function_type = ", ".join(
            self._render_type(value.type) for value in block_argument_values
        )

        scope_env = {
            binding.name: _RenderedValue(name=binding.ssa_name, type=value.type)
            for binding, value in zip(stmt.block_arguments, block_argument_values)
        }

        lines.append(self._indent(indent) + f"pto.strict_vecscope({capture_names}) {{")
        lines.append(self._indent(indent) + f"^bb0({block_args}):")
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + f"}} : ({function_type}) -> ()")
        return lines

    def _render_vecscope(
        self,
        stmt: SemanticVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        scope_env = dict(env)
        lines = [self._indent(indent) + "pto.vecscope {"]
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + "}")
        return lines

    def _render_for(
        self,
        stmt: SemanticForStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        lower_bound = self._lower_expr(stmt.lower_bound, env, indent=indent, into=lines)
        upper_bound = self._lower_expr(stmt.upper_bound, env, indent=indent, into=lines)
        step = self._lower_expr(stmt.step, env, indent=indent, into=lines)

        body_env = dict(env)
        body_env[stmt.induction_variable.name] = _RenderedValue(
            name=stmt.induction_variable.ssa_name,
            type=stmt.induction_variable.type,
        )

        if not stmt.loop_carried:
            lines.append(
                self._indent(indent)
                + f"scf.for {stmt.induction_variable.ssa_name} = {lower_bound.name} "
                f"to {upper_bound.name} step {step.name} {{"
            )
            lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        carried_bindings = tuple(stmt.loop_carried)
        if len(carried_bindings) == 1:
            carried_binding = carried_bindings[0]
            initial_value = self._coerce_rendered_value(
                env[carried_binding.name],
                carried_binding.type,
                indent=indent,
                into=lines,
            )
            iter_arg_name = f"%{carried_binding.name}_iter_{self._loop_counter}"
            self._loop_counter += 1
            body_env[carried_binding.name] = _RenderedValue(
                name=iter_arg_name,
                type=carried_binding.type,
            )

            lines.append(
                self._indent(indent)
                + f"{carried_binding.ssa_name}:1 = scf.for {stmt.induction_variable.ssa_name} = "
                f"{lower_bound.name} to {upper_bound.name} step {step.name} "
                f"iter_args({iter_arg_name} = {initial_value.name}) -> "
                f"({self._render_type(carried_binding.type)}) {{"
            )
            lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
            yielded_value = self._coerce_rendered_value(
                body_env[carried_binding.name],
                carried_binding.type,
                indent=indent + 2,
                into=lines,
            )
            lines.append(
                self._indent(indent + 2)
                + f"scf.yield {yielded_value.name} : {self._render_type(yielded_value.type)}"
            )
            lines.append(self._indent(indent) + "}")
            env[carried_binding.name] = _RenderedValue(
                name=carried_binding.ssa_name,
                type=carried_binding.type,
            )
            return lines

        loop_id = self._loop_counter
        self._loop_counter += 1

        initial_values: list[_RenderedValue] = []
        iter_arg_names: list[str] = []
        for index, binding in enumerate(carried_bindings):
            initial_values.append(
                self._coerce_rendered_value(
                    env[binding.name],
                    binding.type,
                    indent=indent,
                    into=lines,
                )
            )
            iter_arg_names.append(f"%{binding.name}_iter_{loop_id}_{index}")
            body_env[binding.name] = _RenderedValue(
                name=iter_arg_names[-1],
                type=binding.type,
            )

        result_names = ", ".join(binding.ssa_name for binding in carried_bindings)
        iter_args = ", ".join(
            f"{iter_name} = {initial.name}"
            for iter_name, initial in zip(iter_arg_names, initial_values)
        )
        result_types = ", ".join(self._render_type(binding.type) for binding in carried_bindings)

        lines.append(
            self._indent(indent)
            + f"{result_names} = scf.for {stmt.induction_variable.ssa_name} = "
            f"{lower_bound.name} to {upper_bound.name} step {step.name} "
            f"iter_args({iter_args}) -> ({result_types}) {{"
        )
        lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
        yielded_values = [
            self._coerce_rendered_value(
                body_env[binding.name],
                binding.type,
                indent=indent + 2,
                into=lines,
            )
            for binding in carried_bindings
        ]
        yielded_names = ", ".join(value.name for value in yielded_values)
        yielded_types = ", ".join(self._render_type(value.type) for value in yielded_values)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {yielded_names} : {yielded_types}"
        )
        lines.append(self._indent(indent) + "}")
        for binding in carried_bindings:
            env[binding.name] = _RenderedValue(
                name=binding.ssa_name,
                type=binding.type,
            )
        return lines

    def _render_if(
        self,
        stmt: SemanticIfStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        cond_lines: list[str] = []
        condition = self._lower_condition(stmt.condition, env, indent=indent, into=cond_lines)
        then_env = dict(env)
        else_env = dict(env)

        if not stmt.results:
            lines = list(cond_lines)
            lines.append(self._indent(indent) + f"scf.if {condition.name} {{")
            lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
            if stmt.else_body:
                lines.append(self._indent(indent) + "} else {")
                lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        if len(stmt.results) != 1:
            raise NotImplementedError(
                "TileLang DSL v1 lowering currently supports at most one merged if/else binding"
            )

        result = stmt.results[0]
        lines = list(cond_lines)
        lines.append(
            self._indent(indent)
            + f"{result.result_binding.ssa_name} = scf.if {condition.name} -> "
            + f"({self._render_type(result.result_binding.type)}) {{"
        )
        lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
        then_value = then_env.get(result.result_binding.name, then_env.get(result.then_binding.name))
        if then_value is None:
            then_value = _RenderedValue(result.then_binding.ssa_name, result.then_binding.type)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {then_value.name} : {self._render_type(then_value.type)}"
        )
        lines.append(self._indent(indent) + "} else {")
        lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
        else_value = else_env.get(result.result_binding.name, else_env.get(result.else_binding.name))
        if else_value is None:
            else_value = _RenderedValue(result.else_binding.ssa_name, result.else_binding.type)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {else_value.name} : {self._render_type(else_value.type)}"
        )
        lines.append(self._indent(indent) + "}")
        env[result.result_binding.name] = _RenderedValue(
            name=result.result_binding.ssa_name,
            type=result.result_binding.type,
        )
        return lines

    def _lower_condition(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value

        zero_type: SemanticType
        predicate: str
        if isinstance(value.type, SemanticIndexType):
            zero_type = SemanticIndexType()
            predicate = "arith.cmpi ne"
        elif isinstance(value.type, SemanticScalarType):
            zero_type = value.type
            if value.type.dtype.name in {"f16", "bf16", "f32"}:
                predicate = "arith.cmpf une"
            else:
                predicate = "arith.cmpi ne"
        else:
            raise NotImplementedError(f"unsupported if condition type {value.type!r}")

        zero = self._materialize_constant(0, zero_type)
        result_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {predicate}, {value.name}, {zero} : {self._render_type(value.type)}"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _lower_expr(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None = None,
        into: list[str] | None = None,
    ) -> _RenderedValue:
        if isinstance(expr, SemanticBindingRef):
            return env.get(expr.binding.name, _RenderedValue(expr.binding.ssa_name, expr.type))
        if isinstance(expr, SemanticLiteralExpr):
            if desired_name is not None and into is not None:
                into.append(
                    self._indent(indent)
                    + f"{desired_name} = arith.constant {self._format_constant(expr.value, expr.type)} : "
                    f"{self._render_type(expr.type)}"
                )
                return _RenderedValue(name=desired_name, type=expr.type)
            return _RenderedValue(
                name=self._materialize_constant(expr.value, expr.type),
                type=expr.type,
            )
        if isinstance(expr, SemanticSubscriptAccess):
            return self._lower_subscript_access(
                expr,
                env,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if isinstance(expr, SemanticBinaryExpr):
            if into is None:
                into = []
            if expr.op in {"and", "or"}:
                return self._lower_bool_expr(
                    expr.op,
                    expr.lhs,
                    expr.rhs,
                    env,
                    indent=indent,
                    desired_name=desired_name,
                    into=into,
                )
            lhs = self._lower_expr(expr.lhs, env, indent=indent, into=into)
            rhs = self._lower_expr(expr.rhs, env, indent=indent, into=into)
            if expr.op in {"eq", "ne", "gt", "lt", "ge", "le"}:
                return self._lower_compare_expr(
                    expr.op,
                    lhs,
                    rhs,
                    indent=indent,
                    desired_name=desired_name,
                    into=into,
                )
            result_name = desired_name or self._new_temp()
            into.append(
                self._indent(indent)
                + f"{result_name} = {self._render_binary_op(expr.op, expr.type)} "
                f"{lhs.name}, {rhs.name} : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)
        if isinstance(expr, SemanticCallExpr):
            return self._lower_call_expr(expr, env, indent=indent, desired_name=desired_name, into=into)
        if isinstance(expr, SemanticAttributeAccess):
            raise NotImplementedError("bare shape attribute values are not materialized directly")
        if isinstance(expr, SemanticTensorSliceExpr):
            raise NotImplementedError("TensorView slices are only lowered through DMA statements in TileLang DSL v1")
        if isinstance(expr, SemanticSymbolExpr):
            raise NotImplementedError("symbol expressions are only lowered through specialized TileLang DSL ops")
        raise NotImplementedError(f"unsupported semantic expression {type(expr).__name__}")

    def _lower_call_expr(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if expr.namespace is None:
            if into is None:
                into = []
            rendered_args = [
                self._lower_expr(arg, env, indent=indent, into=into)
                for arg in expr.args
            ]
            rendered_arg_names = ", ".join(arg.name for arg in rendered_args)
            rendered_arg_types = ", ".join(self._render_type(arg.type) for arg in rendered_args)
            if not rendered_arg_types:
                rendered_arg_types = ""
            if expr.type is None:
                into.append(
                    self._indent(indent)
                    + f"func.call {_format_symbol_name(expr.name)}({rendered_arg_names}) : "
                    + f"({rendered_arg_types}) -> ()"
                )
                return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))
            result_name = desired_name or self._new_temp()
            into.append(
                self._indent(indent)
                + f"{result_name} = func.call {_format_symbol_name(expr.name)}({rendered_arg_names}) : "
                + f"({rendered_arg_types}) -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.namespace != "pto":
            raise NotImplementedError(f"unsupported call namespace {expr.namespace!r}")
        if isinstance(expr.type, SemanticTupleType):
            raise NotImplementedError("multi-result call values must be assigned directly in TileLang DSL v1")
        if into is None:
            into = []
        result_name = desired_name or self._new_temp()

        if expr.name == "make_mask":
            dtype_expr, pattern_expr = expr.args
            if not self._is_dtype_meta_expr(dtype_expr):
                raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")
            if not isinstance(pattern_expr, SemanticSymbolExpr) or not isinstance(pattern_expr.value, MaskPattern):
                raise NotImplementedError("make_mask pattern lowering expects a MaskPattern symbol")
            suffix = expr.type.granularity
            into.append(
                self._indent(indent)
                + f'{result_name} = pto.pset_{suffix} "{pattern_expr.value.value}" : {self._render_type(expr.type)}'
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "init_align":
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.init_align : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vlds":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=into)
            if (
                isinstance(expr.args[0].type, SemanticTileType)
                and expr.args[0].type.rank == 2
                and len(expr.args[1:]) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    expr.args[0].type,
                    expr.args[1:],
                    env,
                    indent=indent,
                    into=into,
                )
                rendered_indices = self._materialize_constant(0, SemanticIndexType())
            else:
                rendered_indices = self._render_index_list(expr.args[1:], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vlds {source.name}[{rendered_indices}] : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vldas":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            index_args = expr.args[1:]
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=into)
            if (
                isinstance(expr.args[0].type, SemanticTileType)
                and expr.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    expr.args[0].type,
                    index_args,
                    env,
                    indent=indent,
                    into=into,
                )
            if self._is_memref_like_type(source.type):
                ptr_name, ptr_type = self._materialize_copy_buffer_ptr(source, indent=indent, into=into)
                source = _RenderedValue(name=ptr_name, type=_RenderedTextualType(ptr_type))
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vldas {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "load_scalar":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.load_scalar {source.name}[{offset.name}] : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vstus":
            align_in = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            offset = self._coerce_rendered_value(offset, _I32_TYPE, indent=indent, into=into)
            value = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            base = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vstus {align_in.name}, {offset.name}, {value.name}, {base.name} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(offset.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vstur":
            align_in = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            value = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            base = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vstur {align_in.name}, {value.name}, {base.name}, {mode} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "get_block_idx",
            "get_subblock_idx",
            "get_block_num",
            "get_subblock_num",
        }:
            into.append(self._indent(indent) + f"{result_name} = pto.{expr.name}")
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vbr":
            scalar = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vbr {scalar.name} : "
                + f"{self._render_type(scalar.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vdup":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            position = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vdup {value.name}, {mask.name} {{position = {position}}} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vci":
            index = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            order = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vci {index.name} {{order = {order}}} : "
                + f"{self._render_type(index.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "tensor_view_as_ptr":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.tensor_view_addr {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "tile_as_ptr":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.tile_buf_addr {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "castptr":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if isinstance(expr.type, SemanticPtrType) and isinstance(value.type, SemanticIndexType):
                value = self._coerce_rendered_to_i64(value, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.castptr {value.name} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
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
        }:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            return self._coerce_rendered_value(value, expr.type, indent=indent, into=into)

        if expr.name == "addptr":
            pointer = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.addptr {pointer.name}, {offset.name} : "
                + f"{self._render_type(pointer.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"ppack", "punpack"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {part} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "pnot":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.pnot {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "psel":
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.psel {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmp":
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmp {lhs.name}, {rhs.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmps":
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmps {vector.name}, {scalar.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(vector.type)}, {self._render_type(scalar.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vsel":
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vsel {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vselr", "vselrv2"}:
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {src0.name}, {src1.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vintlvv2", "vdintlvv2"}:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {part} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcvt":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            target_dtype = self._render_dtype_symbol(expr.args[1], context="pto.vcvt to_type")
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcvt {value.name}, {target_dtype}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vmrgsort4":
            vec0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            vec1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            vec2 = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            vec3 = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[4], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vmrgsort4 {vec0.name}, {vec1.name}, {vec2.name}, {vec3.name}, {mask.name} : "
                + f"{self._render_type(vec0.type)}, {self._render_type(vec1.type)}, {self._render_type(vec2.type)}, "
                + f"{self._render_type(vec3.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "vabs",
            "vrelu",
            "vexp",
            "vln",
            "vsqrt",
            "vrec",
            "vnot",
            "vcadd",
            "vcmax",
            "vbcnt",
            "vneg",
            "vcls",
            "vcmin",
            "vrsqrt",
            "vmov",
            "vsunpack",
            "vzunpack",
            "vusqz",
            "vsqz",
            "vexpdiff",
            "vtrc",
            "vbitsort",
            "vcgadd",
            "vcgmax",
            "vcgmin",
            "vcpadd",
            "vsort32",
        }:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "vadd",
            "vsub",
            "vmul",
            "vdiv",
            "vmax",
            "vmin",
            "vand",
            "vor",
            "vxor",
            "vaddrelu",
            "vaddreluconv",
            "vsubrelu",
            "vmulconv",
            "vshl",
            "vshr",
            "vprelu",
            "vpack",
            "vperm",
            "vmrgsort",
        }:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vshift", "vslide"}:
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            immediate = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {vector.name}, {immediate.name}, {mask.name} : "
                + f"{self._render_type(vector.type)}, {self._render_type(immediate.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vadds", "vsubs", "vmuls", "vdivs", "vmaxs", "vmins", "vlrelu", "vshls", "vshrs", "vands", "vors", "vxors"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {scalar.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(scalar.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vaxpy", "vmula"}:
            vec0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            vec1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            vec2 = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {vec0.name}, {vec1.name}, {vec2.name}, {mask.name} : "
                + f"{self._render_type(vec0.type)}, {self._render_type(vec1.type)}, {self._render_type(vec2.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        raise NotImplementedError(f"unsupported pto call `{expr.name}` in lowering")

    def _lower_compare_expr(
        self,
        op: str,
        lhs: _RenderedValue,
        rhs: _RenderedValue,
        *,
        indent: int,
        desired_name: str | None,
        into: list[str],
    ) -> _RenderedValue:
        result_name = desired_name or self._new_temp()
        if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
            index_predicates = {
                "eq": "eq",
                "ne": "ne",
                "gt": "sgt",
                "lt": "slt",
                "ge": "sge",
                "le": "sle",
            }
            predicate = index_predicates[op]
        elif isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
            if lhs.type.dtype.name in {"f16", "bf16", "f32"}:
                float_predicates = {
                    "eq": "oeq",
                    "ne": "une",
                    "gt": "ogt",
                    "lt": "olt",
                    "ge": "oge",
                    "le": "ole",
                }
                predicate = float_predicates[op]
                cmp_name = "arith.cmpf"
            else:
                int_predicates = {
                    "eq": "eq",
                    "ne": "ne",
                    "gt": "sgt",
                    "lt": "slt",
                    "ge": "sge",
                    "le": "sle",
                }
                predicate = int_predicates[op]
                cmp_name = "arith.cmpi"
            into.append(
                self._indent(indent)
                + f"{result_name} = {cmp_name} {predicate}, {lhs.name}, {rhs.name} : "
                f"{self._render_type(lhs.type)}"
            )
            return _RenderedValue(name=result_name, type=_I1_TYPE)
        else:
            raise NotImplementedError(
                f"comparison lowering requires matching scalar types or index operands, got {lhs.type!r} and {rhs.type!r}"
            )

        into.append(
            self._indent(indent)
            + f"{result_name} = arith.cmpi {predicate}, {lhs.name}, {rhs.name} : {self._render_type(lhs.type)}"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _lower_bool_expr(
        self,
        op: str,
        lhs_expr: SemanticExpr,
        rhs_expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str],
    ) -> _RenderedValue:
        lhs = self._lower_condition(lhs_expr, env, indent=indent, into=into)
        rhs = self._lower_condition(rhs_expr, env, indent=indent, into=into)
        result_name = desired_name or self._new_temp()
        arith_op = "arith.andi" if op == "and" else "arith.ori"
        into.append(
            self._indent(indent)
            + f"{result_name} = {arith_op} {lhs.name}, {rhs.name} : i1"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _render_string_literal(self, expr: SemanticExpr) -> str:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, str):
            escaped = expr.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.binding.value, str):
            escaped = expr.binding.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        raise NotImplementedError("expected a string literal for TileLang DSL advanced-family lowering")

    def _render_dtype_symbol(self, expr: SemanticExpr, *, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and isinstance(expr.value, ScalarType):
            return expr.value.name
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "dtype"
            and isinstance(expr.binding.value, ScalarType)
        ):
            return expr.binding.value.name
        raise NotImplementedError(f"{context} expects a dtype symbol in TileLang DSL v1 lowering")

    def _lower_to_i1(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value
        raise NotImplementedError("expected an i1 operand during TileLang DSL v1 lowering")

    def _lower_to_i64(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        return self._coerce_rendered_to_i64(value, indent=indent, into=into)

    def _lower_to_i32(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i32":
            return value
        if isinstance(value.type, SemanticIndexType):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = arith.index_cast {value.name} : index to i32"
            )
            return _RenderedValue(name=cast_name, type=_I32_TYPE)
        raise NotImplementedError("expected an i32 or index operand during TileLang DSL v1 lowering")

    def _coerce_rendered_to_i64(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i64":
            return value
        if isinstance(value.type, SemanticIndexType):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = arith.index_castui {value.name} : index to i64"
            )
            return _RenderedValue(name=cast_name, type=_I64_TYPE)
        raise NotImplementedError("expected an i64 or index operand during TileLang DSL v1 lowering")

    def _lower_remaining_to_i32(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i32":
            return value
        if isinstance(value.type, SemanticIndexType):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = arith.index_cast {value.name} : index to i32"
            )
            return _RenderedValue(name=cast_name, type=_I32_TYPE)
        raise NotImplementedError("tail make_mask lowering expects an i32 or index remaining operand")

    def _materialize_copy_buffer_ptr(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> tuple[str, str]:
        ptr_type = self._render_copy_buffer_type(value.type)
        cache_key = (value.name, ptr_type)
        existing = self._castptr_cache.get(cache_key)
        if existing is not None:
            return existing, ptr_type

        if isinstance(value.type, SemanticTileType):
            value = self._materialize_tile_memref(value, indent=indent, into=into)

        if self._is_memref_like_type(value.type):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = pto.castptr {value.name} : {self._render_type(value.type)} -> {ptr_type}"
            )
            self._castptr_cache[cache_key] = cast_name
            return cast_name, ptr_type

        return value.name, ptr_type

    def _coerce_rendered_value(
        self,
        value: _RenderedValue,
        target_type: SemanticType,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        def _scalar_int_bits(dtype: ScalarType) -> int | None:
            if dtype.name == "i1":
                return 1
            return integer_bitwidth(dtype)

        def _scalar_int_sign(dtype: ScalarType) -> str:
            sign = integer_signedness(dtype)
            return "signless" if sign is None else sign

        if type(value.type) is type(target_type) and value.type == target_type:
            return value
        if isinstance(value.type, SemanticIndexType) and isinstance(target_type, SemanticScalarType):
            target_int_bits = _scalar_int_bits(target_type.dtype)
            target_sign = _scalar_int_sign(target_type.dtype)
            if target_int_bits == 32:
                op = "arith.index_castui" if target_sign == "unsigned" else "arith.index_cast"
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : index to {target_type.dtype.name}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if target_int_bits == 64:
                op = "arith.index_castui" if target_sign in {"signless", "unsigned"} else "arith.index_cast"
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : index to {target_type.dtype.name}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if target_type.dtype.name in {"f16", "bf16", "f32"}:
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = arith.uitofp {value.name} : index to {target_type.dtype.name}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
        if isinstance(value.type, SemanticScalarType) and isinstance(target_type, SemanticScalarType):
            src = value.type.dtype.name
            dst = target_type.dtype.name
            if src == dst:
                return value
            cast_name = self._new_temp()
            src_bits = _scalar_int_bits(value.type.dtype)
            dst_bits = _scalar_int_bits(target_type.dtype)
            if src_bits is not None and dst_bits is not None:
                if src_bits == dst_bits:
                    op = "arith.bitcast"
                elif src_bits < dst_bits:
                    op = "arith.extui" if _scalar_int_sign(value.type.dtype) == "unsigned" else "arith.extsi"
                else:
                    op = "arith.trunci"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if src_bits is not None and dst in {"f16", "bf16", "f32"}:
                op = "arith.uitofp" if _scalar_int_sign(value.type.dtype) == "unsigned" else "arith.sitofp"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if src in {"f16", "bf16", "f32"} and dst_bits is not None:
                op = "arith.fptoui" if _scalar_int_sign(target_type.dtype) == "unsigned" else "arith.fptosi"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if src in {"f16", "bf16", "f32"} and dst in {"f16", "bf16", "f32"}:
                op = "arith.extf" if src in {"f16", "bf16"} and dst == "f32" else "arith.truncf"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
        raise NotImplementedError(
            f"unsupported value coercion from {value.type!r} to {target_type!r} in TileLang DSL v1 lowering"
        )

    def _materialize_strict_vecscope_capture(
        self,
        capture: _RenderedValue,
        binding: SemanticBinding,
        *,
        indent: int,
        into: list[str],
    ) -> tuple[_RenderedValue, _RenderedValue]:
        if not self._is_memref_like_type(capture.type):
            return capture, _RenderedValue(name=binding.ssa_name, type=binding.type)

        ptr_name, ptr_type = self._materialize_copy_buffer_ptr(
            capture,
            indent=indent,
            into=into,
        )
        rendered_ptr_type = _RenderedTextualType(ptr_type)
        return (
            _RenderedValue(name=ptr_name, type=rendered_ptr_type),
            _RenderedValue(name=binding.ssa_name, type=rendered_ptr_type),
        )

    def _mask_suffix(self, ty: SemanticType) -> str:
        if not isinstance(ty, SemanticMaskType):
            raise NotImplementedError("tail make_mask lowering expects a mask result type")
        return ty.granularity

    def _is_dtype_meta_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticSymbolExpr):
            return isinstance(expr.value, ScalarType) and expr.type.kind == "dtype"
        if isinstance(expr, SemanticBindingRef):
            return (
                isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            )
        return False

    def _lower_subscript_access(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if (
            into is not None
            and isinstance(expr.base, SemanticAttributeAccess)
            and expr.base.attr == "valid_shape"
            and isinstance(expr.base.base, SemanticBindingRef)
            and isinstance(expr.base.base.type, SemanticTileType)
            and isinstance(expr.index, SemanticLiteralExpr)
            and isinstance(expr.index.value, int)
        ):
            return self._materialize_tile_valid_dim(
                expr.base.base.binding,
                expr.index.value,
                indent=indent,
                into=into,
                desired_name=desired_name,
            )
        if (
            into is not None
            and isinstance(expr.base, SemanticAttributeAccess)
            and expr.base.attr in {"shape", "valid_shape", "strides"}
            and isinstance(expr.base.base, SemanticBindingRef)
            and isinstance(
                expr.base.base.type,
                (SemanticTensorViewType, SemanticPartitionTensorViewType),
            )
            and isinstance(expr.index, SemanticLiteralExpr)
            and isinstance(expr.index.value, int)
        ):
            tensor_value = env.get(
                expr.base.base.binding.name,
                _RenderedValue(expr.base.base.binding.ssa_name, expr.base.base.type),
            )
            result_name = desired_name or self._new_temp()
            axis_value = self._materialize_constant(expr.index.value, SemanticIndexType())
            op_name = (
                "pto.get_tensor_view_stride"
                if expr.base.attr == "strides"
                else "pto.get_tensor_view_dim"
            )
            into.append(
                self._indent(indent)
                + f"{result_name} = {op_name} {tensor_value.name}, {axis_value} : "
                + f"{self._render_type(tensor_value.type)} -> index"
            )
            return _RenderedValue(name=result_name, type=SemanticIndexType())
        value = self._extract_shape_subscript_value(expr, env)
        if isinstance(value, _RenderedValue):
            return value
        if desired_name is not None and into is not None:
            into.append(
                self._indent(indent)
                + f"{desired_name} = arith.constant {self._format_constant(value, expr.type)} : "
                f"{self._render_type(expr.type)}"
            )
            return _RenderedValue(name=desired_name, type=expr.type)
        return _RenderedValue(
            name=self._materialize_constant(value, expr.type),
            type=expr.type,
        )

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _tensor_stride_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__stride_{tensor_name}_{axis}"

    def _materialize_tile_memref(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        existing = self._tile_memref_cache.get(value.name)
        if existing is not None:
            return existing
        if not isinstance(value.type, SemanticTileType):
            return value
        memref_type = _RenderedTextualType(
            self._render_memref_type(
                element_dtype=value.type.element_dtype.name,
                shape=value.type.shape if value.type.shape is not None else ("?",) * value.type.rank,
                memory_space=value.type.memory_space or "ub",
            )
        )
        memref_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{memref_name} = pto.tile_buf_addr {value.name} : "
            + f"{self._render_type(value.type)} -> {self._render_type(memref_type)}"
        )
        rendered = _RenderedValue(name=memref_name, type=memref_type)
        self._tile_memref_cache[value.name] = rendered
        return rendered

    def _materialize_tile_valid_dim(
        self,
        binding: object,
        axis: int,
        *,
        indent: int,
        into: list[str],
        desired_name: str | None = None,
    ) -> _RenderedValue:
        cache_key = (binding.name, axis)
        existing = self._tile_valid_dim_cache.get(cache_key)
        if existing is not None:
            return existing
        source = _RenderedValue(name=binding.ssa_name, type=binding.type)
        op_name = "pto.tile_valid_rows" if axis == 0 else "pto.tile_valid_cols"
        result_name = desired_name or self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {op_name} {source.name} : "
            + f"{self._render_type(source.type)} -> index"
        )
        rendered = _RenderedValue(name=result_name, type=SemanticIndexType())
        self._tile_valid_dim_cache[cache_key] = rendered
        return rendered

    def _extract_shape_subscript_value(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
    ) -> int | _RenderedValue:
        if not isinstance(expr.base, SemanticAttributeAccess):
            raise NotImplementedError("only shape/stride indexing is supported in TileLang DSL v1 lowering")
        if expr.base.attr not in {"shape", "valid_shape", "strides"}:
            raise NotImplementedError(
                "only `.shape[...]`, `.valid_shape[...]`, and `.strides[...]` indexing are supported in TileLang DSL v1 lowering"
            )
        if not isinstance(expr.index, SemanticLiteralExpr) or not isinstance(expr.index.value, int):
            raise NotImplementedError("shape/stride indices must be integer literals in TileLang DSL v1 lowering")
        if not isinstance(expr.base.base, SemanticBindingRef):
            raise NotImplementedError("shape/stride indexing expects a bound TensorView or Tile value")

        base_binding = expr.base.base.binding
        base_value = env.get(base_binding.name, _RenderedValue(base_binding.ssa_name, base_binding.type))
        base_type = base_value.type
        index = expr.index.value

        if isinstance(base_type, SemanticTileType):
            if expr.base.attr == "shape":
                if base_type.shape is None:
                    raise NotImplementedError("dynamic Tile shapes are not supported in TileLang DSL v1 lowering")
                return base_type.shape[index]
            if base_type.valid_shape is None:
                raise NotImplementedError("dynamic Tile shapes are not supported in TileLang DSL v1 lowering")
            valid_dim = base_type.valid_shape[index]
            if valid_dim is not None:
                return valid_dim
            return _RenderedValue(name=base_binding.ssa_name, type=base_type)

        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            if expr.base.attr == "strides":
                hidden_name = self._tensor_stride_binding_name(base_binding.name, index)
            else:
                hidden_name = self._tensor_shape_binding_name(base_binding.name, index)
            hidden_value = env.get(hidden_name)
            if hidden_value is None:
                raise NotImplementedError(
                    f"missing TensorView/PartitionTensorView {expr.base.attr} binding for '{base_binding.name}.{expr.base.attr}[{index}]'"
                )
            return hidden_value

        raise NotImplementedError("shape/stride indexing expects a Tile, TensorView, or PartitionTensorView operand")

    def _format_shape_tuple(self, shape: tuple[int | None, ...]) -> str:
        return "(" + ", ".join("?" if dim is None else str(dim) for dim in shape) + ")"

    def _materialize_constant(self, value: object, ty: SemanticType) -> str:
        cache_key = (self._render_type(ty), value)
        if cache_key in self._constant_cache:
            return self._constant_cache[cache_key]

        name = self._constant_name(value, ty)
        self._constant_cache[cache_key] = name
        self._constant_lines.append(
            self._indent(4)
            + f"{name} = arith.constant {self._format_constant(value, ty)} : {self._render_type(ty)}"
        )
        return name

    def _constant_name(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            stem = f"c{value}"
        elif isinstance(ty, SemanticScalarType):
            if ty.dtype.name == "i1" and isinstance(value, bool):
                stem = "true" if value else "false"
            else:
                stem = f"c{value}_{ty.dtype.name}"
        else:
            stem = "cst"
        name = f"%{stem}"
        existing = {line.split(" = ", 1)[0].strip() for line in self._constant_lines}
        if name not in existing:
            return name
        suffix = 0
        while f"{name}_{suffix}" in existing:
            suffix += 1
        return f"{name}_{suffix}"

    def _format_constant(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            return str(value)
        if isinstance(ty, SemanticScalarType):
            if ty.dtype.name == "i1" and isinstance(value, bool):
                return "1" if value else "0"
            return str(value)
        raise NotImplementedError(f"unsupported constant type {ty!r}")

    def _render_binary_op(self, op: str, ty: SemanticType) -> str:
        if isinstance(ty, (SemanticIndexType, SemanticScalarType)):
            if op == "add":
                return "arith.addi"
            if op == "sub":
                return "arith.subi"
            if op == "mul":
                return "arith.muli"
            if op == "floordiv":
                return "arith.floordivsi"
        raise NotImplementedError(f"unsupported binary op '{op}' for type {ty!r}")

    def _render_type(self, ty: SemanticType) -> str:
        if isinstance(ty, _RenderedTextualType):
            return ty.text
        if isinstance(ty, SemanticIndexType):
            return "index"
        if isinstance(ty, SemanticScalarType):
            return ty.dtype.name
        if isinstance(ty, SemanticPtrType):
            return f"!pto.ptr<{ty.element_dtype.name}, {ty.memory_space}>"
        if isinstance(ty, SemanticTensorViewType):
            return self._render_tensor_view_type(
                element_dtype=ty.element_dtype.name,
                shape=("?",) * ty.rank,
            )
        if isinstance(ty, SemanticPartitionTensorViewType):
            return self._render_partition_tensor_view_type(
                element_dtype=ty.element_dtype.name,
                shape=("?",) * ty.rank,
            )
        if isinstance(ty, SemanticTileType):
            return self._render_tile_buf_type(ty)
        if isinstance(ty, SemanticAlignType):
            return "!pto.align"
        if isinstance(ty, SemanticMaskType):
            return f"!pto.mask<{ty.granularity}>"
        if isinstance(ty, SemanticVRegType):
            return f"!pto.vreg<{ty.lanes}x{ty.element_dtype.name}>"
        raise NotImplementedError(f"unsupported semantic type {ty!r}")

    def _is_memref_like_type(self, ty: SemanticType) -> bool:
        return isinstance(ty, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)) or (
            isinstance(ty, _RenderedTextualType) and ty.text.startswith("memref<")
        )

    def _render_copy_buffer_type(self, ty: SemanticType) -> str:
        if isinstance(ty, SemanticPtrType):
            return self._render_type(ty)
        if isinstance(ty, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return f"!pto.ptr<{ty.element_dtype.name}, gm>"
        if isinstance(ty, SemanticTileType):
            memory_space = ty.memory_space or "ub"
            return f"!pto.ptr<{ty.element_dtype.name}, {memory_space}>"
        return self._render_type(ty)

    def _render_memref_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
        memory_space: str,
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"memref<{dims}x{element_dtype}, {self._render_memref_memory_space(memory_space)}>"

    def _render_tensor_view_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"!pto.tensor_view<{dims}x{element_dtype}>"

    def _render_partition_tensor_view_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"!pto.partition_tensor_view<{dims}x{element_dtype}>"

    def _render_memref_memory_space(self, memory_space: str) -> str:
        if memory_space == "gm":
            return "#pto.address_space<gm>"
        if memory_space == "ub":
            return "#pto.address_space<vec>"
        raise NotImplementedError(f"unsupported memref memory space '{memory_space}' in TileLang DSL v1 lowering")

    def _render_tile_buf_type(self, ty: SemanticTileType) -> str:
        if ty.shape is None:
            raise NotImplementedError("tile_buf lowering requires statically specialized Tile shape")
        if ty.rank not in (1, 2):
            raise NotImplementedError("tile_buf lowering only supports rank-1 or rank-2 Tile values")
        rows = ty.shape[0]
        cols = 1 if ty.rank == 1 else ty.shape[1]
        valid_shape = ty.valid_shape or ty.shape
        v_row = valid_shape[0]
        v_col = 1 if ty.rank == 1 else valid_shape[1]
        return (
            f"!pto.tile_buf<loc={self._render_tile_buf_loc(ty.memory_space or 'ub')}, "
            f"dtype={ty.element_dtype.name}, rows={rows}, cols={cols}, "
            f"v_row={self._render_tile_buf_dim(v_row)}, v_col={self._render_tile_buf_dim(v_col)}, "
            "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
        )

    def _render_tile_buf_loc(self, memory_space: str) -> str:
        if memory_space == "ub":
            return "vec"
        if memory_space == "gm":
            return "gm"
        raise NotImplementedError(f"unsupported tile_buf memory space '{memory_space}'")

    def _render_tile_buf_dim(self, dim: int | None) -> str:
        return "?" if dim is None else str(dim)

    def _dtype_byte_width(self, dtype: ScalarType) -> int:
        try:
            return bytewidth(dtype)
        except TypeError as exc:
            raise NotImplementedError(f"unsupported DMA dtype '{dtype.name}' in TileLang DSL v1 lowering") from exc

    def _indent(self, indent: int) -> str:
        return " " * indent

    def _new_temp(self) -> str:
        name = f"%tmp_{self._temp_counter}"
        self._temp_counter += 1
        return name


def lower_semantic_kernel(kernel: SemanticKernel) -> AuthoringModule:
    """Lower the semantic model to the current authoring-form VPTO builder."""

    return AuthoringModule(kernel=kernel)


__all__ = ["AuthoringModule", "lower_semantic_kernel"]

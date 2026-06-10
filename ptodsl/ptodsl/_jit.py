# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.jit`` decorator and compiled-kernel handles."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from ._diagnostics import invalid_jit_mode_error
from ._kernel_compilation import CompiledKernelHandle, KernelCompiler
from ._kernel_signature import parse_jit_kernel_signature
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
)

from mlir.ir import InsertionPoint


_MODULE_ATTRS = ("pto.target_arch", "pto.kernel_kind", "pto.mode")
_SUPPORTED_FRONTEND_OPTION_KEYS = {"ast_rewrite", "rewrite_part", "dump_rewritten_source"}
_SUPPORTED_REWRITE_PARTS = {"control_flow"}


def _normalize_mode(mode: str, *, fn=None) -> str:
    if mode not in {"auto", "explicit"}:
        source_file = None
        source_line = None
        function_name = None
        if fn is not None:
            function_name = fn.__name__
            try:
                source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
            except (OSError, TypeError):
                source_file = None
            source_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        raise invalid_jit_mode_error(
            mode,
            function_name=function_name,
            source_file=source_file,
            source_line=source_line,
        )
    return mode


def _module_attr_map(module):
    attrs = module.operation.attributes
    return {name: str(attrs[name]) for name in _MODULE_ATTRS if name in attrs}


def _normalize_frontend_options(*, ast_rewrite, frontend_options):
    if frontend_options is None:
        return True if ast_rewrite is None else bool(ast_rewrite)
    if not isinstance(frontend_options, Mapping):
        raise TypeError("@pto.jit frontend_options must be a mapping")

    unknown = set(frontend_options) - _SUPPORTED_FRONTEND_OPTION_KEYS
    if unknown:
        raise ValueError(f"@pto.jit frontend_options has unsupported keys: {sorted(unknown)!r}")

    option_ast_rewrite = frontend_options.get("ast_rewrite")
    if option_ast_rewrite is not None and not isinstance(option_ast_rewrite, bool):
        raise TypeError("@pto.jit frontend_options['ast_rewrite'] must be a bool")
    if ast_rewrite is not None and option_ast_rewrite is not None and bool(ast_rewrite) != option_ast_rewrite:
        raise ValueError("@pto.jit ast_rewrite conflicts with frontend_options['ast_rewrite']")

    enabled = option_ast_rewrite if option_ast_rewrite is not None else (True if ast_rewrite is None else bool(ast_rewrite))

    rewrite_part = frontend_options.get("rewrite_part", {"control_flow"})
    if isinstance(rewrite_part, str):
        rewrite_parts = {rewrite_part}
    else:
        try:
            rewrite_parts = set(rewrite_part)
        except TypeError as exc:
            raise TypeError("@pto.jit frontend_options['rewrite_part'] must be a string or iterable of strings") from exc
    unsupported_parts = rewrite_parts - _SUPPORTED_REWRITE_PARTS
    if unsupported_parts:
        raise ValueError(
            "@pto.jit frontend_options['rewrite_part'] currently only supports "
            f"{sorted(_SUPPORTED_REWRITE_PARTS)!r}; got unsupported parts: {sorted(unsupported_parts)!r}"
        )
    if enabled and "control_flow" not in rewrite_parts:
        raise ValueError("@pto.jit ast_rewrite=True requires rewrite_part to include 'control_flow'")

    dump_rewritten_source = frontend_options.get("dump_rewritten_source", False)
    if not isinstance(dump_rewritten_source, bool):
        raise TypeError("@pto.jit frontend_options['dump_rewritten_source'] must be a bool")
    if dump_rewritten_source:
        raise ValueError("@pto.jit frontend_options['dump_rewritten_source']=True is reserved but not implemented yet")
    return enabled


def merge_jit_modules(*kernels: KernelHandle):
    """
    Merge multiple ``@pto.jit`` kernels into one MLIR module.

    Each handle must have been compiled with the same ``target``,
    ``kernel_kind``, and ``mode`` module attributes. Function order follows
    *kernels*.
    """
    if not kernels:
        raise ValueError("merge_jit_modules() requires at least one kernel handle")

    merged = kernels[0].build()
    expected_attrs = _module_attr_map(merged)

    for kernel in kernels[1:]:
        module = kernel.build()
        actual_attrs = _module_attr_map(module)
        if actual_attrs != expected_attrs:
            raise ValueError(
                "merge_jit_modules() requires compatible module attributes; "
                f"expected {expected_attrs}, got {actual_attrs}"
            )
        with InsertionPoint(merged.body):
            for op in module.body.operations:
                op.operation.clone()

    merged.operation.verify()
    return merged


def jit(
    name=None,
    *,
    target: str = "a5",
    kernel_kind: str = "vector",
    mode: str = "auto",
    insert_sync: bool | None = None,
    ast_rewrite: bool | None = None,
    frontend_options: Mapping | None = None,
):
    """
    Decorator that wraps a Python function as a PTODSL JIT kernel template.

    Parameters
    ----------
    name:        IR function name (defaults to the Python function name).
    target:      Target architecture string, e.g. ``"a5"``.
    kernel_kind: ``"vector"`` or ``"cube"`` – sets ``pto.kernel_kind``.
    mode:        ``"auto"`` or ``"explicit"`` – sets ``pto.mode``.
    insert_sync: ``True``/``False`` to explicitly control PTOAS sync insertion
                 for launch builds. ``None`` keeps the mode-based default
                 behavior.
    ast_rewrite:
                 ``True`` enables AST rewriting of Python ``if`` /
                 ``for range(...)`` into device-side PTODSL control flow.
                 Defaults to ``True``. ``False`` is intended for frontend
                 debugging and trace-time compatibility checks.
    frontend_options:
                 Reserved structured frontend options. Currently supports
                 ``ast_rewrite`` and ``rewrite_part={"control_flow"}``.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - supports ``my_kernel.compile(**constexprs)`` specialization,
    - prints as the default-specialization MLIR text,
    - exposes ``my_kernel.mlir_module()`` / ``verify()`` / ``emit()`` on the
      default specialization for convenience.
    - emits a flat aicore launch-entry module by default.
    """
    normalized_ast_rewrite = _normalize_frontend_options(
        ast_rewrite=ast_rewrite,
        frontend_options=frontend_options,
    )

    def decorator(fn):
        fn_name = name or fn.__name__
        kernel_signature = parse_jit_kernel_signature(fn)
        normalized_mode = _normalize_mode(mode, fn=fn)
        source_file = None
        try:
            source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        except (OSError, TypeError):
            source_file = None
        compiler = KernelCompiler(
            fn.__name__,
            KernelModuleSpec(
                function_name=fn_name,
                target_arch=target,
                kernel_kind=kernel_kind,
                mode=normalized_mode,
                insert_sync=insert_sync,
                module_style=ModuleStyle.FLAT_AICORE,
                source_file=source_file,
                source_line=getattr(fn.__code__, "co_firstlineno", None),
            ),
            kernel_signature,
            fn,
            ast_rewrite=normalized_ast_rewrite,
        )
        return KernelHandle(fn.__name__, compiler)

    return decorator


class KernelHandle(ModuleArtifact):
    """
    Represents a JIT kernel template plus its compiled specializations.

    ``handle.compile(**constexprs)`` returns one compiled specialization.
    ``print(handle)`` emits the default-specialization MLIR module text.
    """

    def __init__(self, py_name: str, compiler: KernelCompiler):
        self._compiler = compiler
        super().__init__(py_name, module_factory=self._build_default_module)

    def compile(self, **constexpr_bindings) -> CompiledKernelHandle:
        return self._compiler.compile(**constexpr_bindings)

    def cached_specializations(self):
        return self._compiler.cached_specializations()

    def _build_default_module(self):
        return self.compile().build()


__all__ = ["CompiledKernelHandle", "jit", "KernelHandle", "merge_jit_modules"]

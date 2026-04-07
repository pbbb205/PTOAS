"""CLI helper invoked by ExpandTileOp to instantiate a tilelang DSL template.

Usage:
    python3 -m tilelang_dsl.expand_helper \
        --template-dir /path/to/templates \
        --op pto.tadd \
        --dtype f32 \
        --shape 16,64 \
        --memory-space ub

Scans --template-dir for .py files, finds a @vkernel whose `op` matches,
specializes every Tile parameter with the given shape/memory_space, and
prints the materialized MLIR module to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .kernel import VKernelDescriptor
from .types import MemorySpace, ScalarType, TileSpecialization


_DTYPE_MAP: dict[str, ScalarType] = {}


def _populate_dtype_map() -> None:
    from . import types as _t

    for name in ("f16", "bf16", "f32", "i8", "i16", "i32", "i64"):
        obj = getattr(_t, name, None)
        if isinstance(obj, ScalarType):
            _DTYPE_MAP[name] = obj


_populate_dtype_map()

_MEMSPACE_MAP = {
    "ub": MemorySpace.UB,
    "gm": MemorySpace.GM,
}


def _find_descriptors(module) -> list[VKernelDescriptor]:
    """Return all VKernelDescriptor instances found as module-level attributes."""
    result = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, VKernelDescriptor):
            result.append(obj)
    return result


def _import_py_file(path: Path):
    """Import a .py file as a module and return it."""
    spec = importlib.util.spec_from_file_location(f"_tl_template_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"expand_helper: warning: failed to import {path}: {exc}", file=sys.stderr)
        return None
    return mod


def _match_descriptor(
    descriptors: list[VKernelDescriptor],
    op_name: str,
    dtype_name: str,
) -> VKernelDescriptor | None:
    """Find the first descriptor matching (op, dtype)."""
    target_dtype = _DTYPE_MAP.get(dtype_name)
    if target_dtype is None:
        return None

    for desc in descriptors:
        if desc.op != op_name:
            continue
        # Check dtype signature: all entries must match the target dtype.
        sig = desc.dtype_signature
        if all(d == target_dtype for d in sig):
            return desc
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TileLang DSL expand helper")
    parser.add_argument("--template-dir", required=True, help="Directory of .py templates")
    parser.add_argument("--op", required=True, help="Tile op name, e.g. pto.tadd")
    parser.add_argument("--dtype", required=True, help="Element dtype, e.g. f32")
    parser.add_argument("--shape", required=True, help="Tile shape, e.g. 16,64")
    parser.add_argument("--memory-space", default="ub", help="Memory space (ub or gm)")
    args = parser.parse_args(argv)

    template_dir = Path(args.template_dir)
    if not template_dir.is_dir():
        print(f"expand_helper: error: {template_dir} is not a directory", file=sys.stderr)
        return 1

    shape = tuple(int(d) for d in args.shape.split(","))
    mem_space = _MEMSPACE_MAP.get(args.memory_space)
    if mem_space is None:
        print(f"expand_helper: error: unknown memory-space '{args.memory_space}'", file=sys.stderr)
        return 1

    # Scan all .py files for descriptors.
    all_descriptors: list[VKernelDescriptor] = []
    for py_path in sorted(template_dir.glob("*.py")):
        mod = _import_py_file(py_path)
        if mod is None:
            continue
        all_descriptors.extend(_find_descriptors(mod))

    if not all_descriptors:
        print(f"expand_helper: error: no @vkernel descriptors found in {template_dir}", file=sys.stderr)
        return 1

    # Match.
    desc = _match_descriptor(all_descriptors, args.op, args.dtype)
    if desc is None:
        print(
            f"expand_helper: error: no template matches op={args.op} dtype={args.dtype}",
            file=sys.stderr,
        )
        return 1

    # Specialize all Tile parameters with the same shape/memory_space.
    tile_specs = {}
    for param in desc.tile_parameters:
        tile_specs[param.name] = TileSpecialization(
            shape=shape,
            memory_space=mem_space,
        )

    specialized = desc.specialize(**tile_specs)

    # Emit MLIR to stdout.
    try:
        mlir_text = specialized.mlir_text()
    except Exception as exc:
        print(f"expand_helper: error: materialization failed: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(mlir_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

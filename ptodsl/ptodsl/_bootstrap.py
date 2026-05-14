# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MLIR path bootstrap and context factory.

Discovers local LLVM MLIR Python bindings plus PTO Python dialect artifacts so
that ``ptodsl`` can import ``mlir`` / ``mlir.dialects.pto`` directly from a
developer workspace without requiring the caller to pre-seed ``PYTHONPATH``.
"""

import os
import sys
from pathlib import Path


def _candidate_python_roots() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    workspace_root = repo_root.parent
    env_roots = []
    for env_name in ("MLIR_PYTHON_ROOT", "PTO_PYTHON_ROOT"):
        raw = os.environ.get(env_name)
        if raw:
            env_roots.append(Path(raw))

    return [
        *env_roots,
        repo_root / "build" / "python",
        repo_root / "install",
        workspace_root / "llvm-project" / "build-shared" / "tools" / "mlir" / "python_packages" / "mlir_core",
    ]


def _bootstrap_python_paths() -> None:
    added = set()
    for root in _candidate_python_roots():
        if not root or not root.is_dir():
            continue
        if not (root / "mlir").exists():
            continue
        root_text = str(root)
        if root_text in added or root_text in sys.path:
            continue
        sys.path.insert(0, root_text)
        added.add(root_text)


_bootstrap_python_paths()

from mlir.dialects import pto as _pto_dialect  # noqa: E402
from mlir.ir import Context, Location           # noqa: E402


def make_context() -> Context:
    """Create a fresh MLIR Context with the PTO dialect loaded."""
    ctx = Context()
    _pto_dialect.register_dialect(ctx, load=True)
    return ctx


__all__ = ["make_context"]

#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context
from mlir.dialects import pto


def main() -> None:
    with Context() as ctx:
        pto.register_dialect(ctx)

        default_attr = pto.PrecisionModeAttr.get(pto.PrecisionMode.DEFAULT, ctx)
        high_attr = pto.PrecisionModeAttr.get(pto.PrecisionMode.HIGH_PRECISION, ctx)

        if default_attr.value != pto.PrecisionMode.DEFAULT.value:
            raise AssertionError("DEFAULT PrecisionModeAttr value mismatch")
        if high_attr.value != pto.PrecisionMode.HIGH_PRECISION.value:
            raise AssertionError("HIGH_PRECISION PrecisionModeAttr value mismatch")
        if "precision_mode" not in str(high_attr) or "HIGH_PRECISION" not in str(high_attr):
            raise AssertionError(f"unexpected PrecisionModeAttr print form: {high_attr}")

    print("precision_mode_bindings: PASS")


if __name__ == "__main__":
    main()

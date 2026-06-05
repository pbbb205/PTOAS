# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module
from mlir.dialects import pto


def build_sections_test():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.parse(
                r"""
module {
  func.func @test_vector_section_region(%out: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.000000e+00 : f32
    %two = arith.constant 2.000000e+00 : f32

    pto.section.vector {
      pto.store_scalar %one, %out[%c0] : !pto.ptr<f32>, f32
      pto.store_scalar %two, %out[%c1] : !pto.ptr<f32>, f32
    }

    return
  }
}
""",
                ctx,
            )
            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build_sections_test())

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains:
//   - pto.tile_buf_addr   → extract memref address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// This pass resolves them against the concrete tile_buf values at the
// call site.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

/// Compute the row-major strided memref type for a tile_buf.
static MemRefType computeBridgeMemrefType(pto::TileBufType tbTy,
                                          MLIRContext *ctx) {
  ArrayRef<int64_t> shape = tbTy.getShape();
  ArrayRef<int64_t> validShape = tbTy.getValidShape();

  SmallVector<int64_t, 2> memrefDims;
  for (unsigned d = 0; d < shape.size(); ++d) {
    if (d < validShape.size() && validShape[d] != ShapedType::kDynamic)
      memrefDims.push_back(validShape[d]);
    else
      memrefDims.push_back(ShapedType::kDynamic);
  }

  SmallVector<int64_t, 2> strides(shape.size(), 1);
  for (int s = static_cast<int>(shape.size()) - 2; s >= 0; --s)
    strides[s] = strides[s + 1] * shape[s + 1];

  auto stridedLayout = StridedLayoutAttr::get(ctx, /*offset=*/0, strides);
  return MemRefType::get(memrefDims, tbTy.getElementType(), stridedLayout,
                         tbTy.getMemorySpace());
}

/// Try to find the dynamic valid_row index from the tile_buf's defining op
/// chain (e.g. pto.bind_tile carries optional valid_row/valid_col operands).
static Value findDynamicValidRow(Value tileBuf) {
  Value cur = tileBuf;
  while (cur) {
    if (auto bindOp = cur.getDefiningOp<pto::BindTileOp>()) {
      if (bindOp.getValidRow())
        return bindOp.getValidRow();
      // bind_tile may chain — trace further through its source.
      cur = bindOp.getSource();
      continue;
    }
    break;
  }
  return nullptr;
}

/// Try to find the dynamic valid_col index from the tile_buf's defining op.
static Value findDynamicValidCol(Value tileBuf) {
  Value cur = tileBuf;
  while (cur) {
    if (auto bindOp = cur.getDefiningOp<pto::BindTileOp>()) {
      if (bindOp.getValidCol())
        return bindOp.getValidCol();
      cur = bindOp.getSource();
      continue;
    }
    break;
  }
  return nullptr;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    SmallVector<pto::TileBufAddrOp, 8> addrOps;
    SmallVector<pto::TileValidRowsOp, 8> rowsOps;
    SmallVector<pto::TileValidColsOp, 8> colsOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op))
        addrOps.push_back(addr);
      else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op))
        rowsOps.push_back(rows);
      else if (auto cols = dyn_cast<pto::TileValidColsOp>(op))
        colsOps.push_back(cols);
    });

    // Fold pto.tile_buf_addr → pto.simd.tile_to_memref.
    for (auto addrOp : addrOps) {
      builder.setInsertionPoint(addrOp);
      auto tbTy = dyn_cast<pto::TileBufType>(addrOp.getSrc().getType());
      if (!tbTy) {
        addrOp.emitError("tile_buf_addr source is not tile_buf");
        return signalPassFailure();
      }

      MemRefType bridgeMemref = computeBridgeMemrefType(tbTy, ctx);
      auto bridge = builder.create<pto::SimdTileToMemrefOp>(
          addrOp.getLoc(), bridgeMemref, addrOp.getSrc());

      Value result = bridge.getDst();
      if (result.getType() != addrOp.getDst().getType()) {
        result = builder.create<memref::CastOp>(
            addrOp.getLoc(), addrOp.getDst().getType(), result);
      }

      addrOp.getDst().replaceAllUsesWith(result);
      addrOp.erase();
    }

    // Fold pto.tile_valid_rows → arith.constant or dynamic index.
    for (auto rowsOp : rowsOps) {
      builder.setInsertionPoint(rowsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().empty()) {
        rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vRow = tbTy.getValidShape()[0];
      Value replacement;
      if (vRow != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
      } else {
        replacement = findDynamicValidRow(rowsOp.getSrc());
        if (!replacement) {
          rowsOp.emitError(
              "tile_valid_rows: dynamic v_row but cannot find runtime value "
              "(expected pto.bind_tile with valid_row operand)");
          return signalPassFailure();
        }
      }
      rowsOp.getResult().replaceAllUsesWith(replacement);
      rowsOp.erase();
    }

    // Fold pto.tile_valid_cols → arith.constant or dynamic index.
    for (auto colsOp : colsOps) {
      builder.setInsertionPoint(colsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().size() < 2) {
        colsOp.emitError("tile_valid_cols: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vCol = tbTy.getValidShape()[1];
      Value replacement;
      if (vCol != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
      } else {
        replacement = findDynamicValidCol(colsOp.getSrc());
        if (!replacement) {
          colsOp.emitError(
              "tile_valid_cols: dynamic v_col but cannot find runtime value "
              "(expected pto.bind_tile with valid_col operand)");
          return signalPassFailure();
        }
      }
      colsOp.getResult().replaceAllUsesWith(replacement);
      colsOp.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

} // namespace pto
} // namespace mlir

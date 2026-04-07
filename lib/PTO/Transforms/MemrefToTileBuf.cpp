// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- MemrefToTileBuf.cpp ------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// After PTOViewToMemref + PlanMemory + InsertSync, the IR uses memref types
// with pto.bind_tile ops carrying tile metadata.  This pass recovers tile_buf
// types from those anchors so that the subsequent Tile→Vector lowering
// (Expand TileOp) can operate on tile_buf semantics.
//
// The pass does NOT redo memory planning or synchronisation; it only re-wraps
// planned memref values into tile_buf through unrealized_conversion_cast.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace pto {
  namespace func = ::mlir::func;

  #define GEN_PASS_DEF_MEMREFTOTILEBUF
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

// ============================================================================
// Helper: reconstruct TileBufType from a BindTileOp
// ============================================================================
// BindTileOp carries:
//   - source memref → shape, elementType, memorySpace
//   - config attr   → bLayout, sLayout, fractal, pad
//   - valid_row/col → validShape (static or dynamic)
static pto::TileBufType reconstructTileBufType(pto::BindTileOp bindOp) {
  auto memrefTy = cast<MemRefType>(bindOp.getSource().getType());
  MLIRContext *ctx = bindOp.getContext();

  ArrayRef<int64_t> shape = memrefTy.getShape();
  Type elemTy = memrefTy.getElementType();
  Attribute memSpace = memrefTy.getMemorySpace();
  pto::TileBufConfigAttr config = bindOp.getConfig();

  // Recover valid shape: if BindTileOp provides valid_row/valid_col, check
  // whether they are static constants.  Otherwise mark as dynamic.
  SmallVector<int64_t, 2> validShape;
  if (shape.size() == 2) {
    auto resolveValidDim = [](Value v, int64_t staticDim) -> int64_t {
      if (!v)
        return staticDim; // no dynamic override → use static shape
      if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
        return cOp.value();
      if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
        return cInt.value();
      return ShapedType::kDynamic;
    };
    validShape.push_back(resolveValidDim(bindOp.getValidRow(), shape[0]));
    validShape.push_back(resolveValidDim(bindOp.getValidCol(), shape[1]));
  } else {
    // Fallback: validShape = shape
    validShape.assign(shape.begin(), shape.end());
  }

  return pto::TileBufType::get(ctx, shape, elemTy, memSpace, validShape,
                               config);
}

// ============================================================================
// Helper: check whether an op is a tile-level op (needs tile_buf operands)
// ============================================================================
static bool isTileOp(Operation *op) {
  return isa<pto::OpPipeInterface>(op);
}

// ============================================================================
// The Pass
// ============================================================================
struct MemrefToTileBufPass
    : public mlir::pto::impl::MemrefToTileBufBase<MemrefToTileBufPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (failed(processFunction(func, ctx)))
        return signalPassFailure();
    }
  }

private:
  LogicalResult processFunction(func::FuncOp func, MLIRContext *ctx);
};

LogicalResult MemrefToTileBufPass::processFunction(func::FuncOp func,
                                                    MLIRContext *ctx) {
  OpBuilder builder(ctx);

  // Phase 1: For each BindTileOp, reconstruct tile_buf type and create a
  // cast from the BindTileOp result (memref) to tile_buf.
  //
  // We build a mapping: BindTileOp result (memref Value) → tile_buf Value
  // so that Phase 2 can replace tile op operands.
  llvm::DenseMap<Value, Value> memrefToTileBuf;

  SmallVector<pto::BindTileOp, 16> bindOps;
  func.walk([&](pto::BindTileOp op) { bindOps.push_back(op); });

  for (auto bindOp : bindOps) {
    pto::TileBufType tileBufTy = reconstructTileBufType(bindOp);

    // Insert unrealized_conversion_cast right after the BindTileOp.
    builder.setInsertionPointAfter(bindOp);
    auto cast = builder.create<UnrealizedConversionCastOp>(
        bindOp.getLoc(), tileBufTy, bindOp.getResult());

    memrefToTileBuf[bindOp.getResult()] = cast.getResult(0);
  }

  // Phase 2: For each tile op, replace memref operands that have a
  // corresponding tile_buf value.
  func.walk([&](Operation *op) {
    if (!isTileOp(op))
      return;
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = memrefToTileBuf.find(operand.get());
      if (it != memrefToTileBuf.end()) {
        operand.set(it->second);
      }
    }
  });

  // Phase 3: For function arguments that feed directly into BindTileOp,
  // convert the argument type to tile_buf and propagate.
  //
  // Pattern:  func @f(%arg: memref<...>) { %b = bind_tile %arg ... }
  //        →  func @f(%arg: tile_buf<...>) { ... }
  //
  // We track which args were converted so we can update the function type.
  Block &entry = func.front();
  auto fnTy = func.getFunctionType();
  SmallVector<Type> newInputTypes(fnTy.getInputs().begin(),
                                  fnTy.getInputs().end());
  bool sigChanged = false;

  for (auto bindOp : bindOps) {
    Value source = bindOp.getSource();
    auto blockArg = dyn_cast<BlockArgument>(source);
    if (!blockArg || blockArg.getOwner() != &entry)
      continue;

    // This argument was originally tile_buf before PTOViewToMemref.
    unsigned idx = blockArg.getArgNumber();
    pto::TileBufType tileBufTy = reconstructTileBufType(bindOp);

    // Replace all tile op uses of the cast with the block arg directly.
    auto castIt = memrefToTileBuf.find(bindOp.getResult());
    if (castIt == memrefToTileBuf.end())
      continue;
    Value tileBufVal = castIt->second;

    // Save the original memref type before mutating.
    Type origMemrefTy = blockArg.getType();

    // Change the block argument type to tile_buf.
    blockArg.setType(tileBufTy);
    newInputTypes[idx] = tileBufTy;
    sigChanged = true;

    // Insert a tile_buf → memref cast so that existing memref users of the
    // block arg continue to work (PlanMemory / InsertSync results).
    builder.setInsertionPointToStart(&entry);
    auto backCast = builder.create<UnrealizedConversionCastOp>(
        func.getLoc(), origMemrefTy, blockArg);

    // Replace all non-tile-op uses of the original block arg with the
    // back-cast memref.  (Tile ops will use the tile_buf directly.)
    //
    // We must be careful: the BindTileOp itself uses the block arg as source.
    // After this rewrite, BindTileOp.source should use the back-cast.
    blockArg.replaceAllUsesWith(backCast.getResult(0));
    // But the back-cast's own operand must remain the block arg.
    backCast.getInputsMutable().assign(ValueRange{blockArg});

    // Now replace tile op uses: they should use the tile_buf block arg
    // directly instead of going through the unrealized_conversion_cast
    // chain.
    tileBufVal.replaceAllUsesWith(blockArg);
    // Erase the now-dead forward cast (memref → tile_buf).
    if (auto castOp =
            tileBufVal.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp->use_empty())
        castOp->erase();
    }
  }

  // Update function signature if any arguments changed.
  if (sigChanged) {
    func.setFunctionType(
        FunctionType::get(ctx, newInputTypes, fnTy.getResults()));
  }

  // Phase 4: Clean up BindTileOps whose results are only used by the
  // (now-erased) forward casts.  If a BindTileOp still has memref users
  // (e.g. memref.subview), keep it.
  for (auto bindOp : bindOps) {
    if (bindOp->use_empty())
      bindOp->erase();
  }

  return success();
}

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createMemrefToTileBufPass() {
  return std::make_unique<MemrefToTileBufPass>();
}

} // namespace pto
} // namespace mlir

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- ExpandTileOp.cpp ---------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Expand tile-level ops (pto.tadd, pto.tsub, ...) by invoking the TileLang
// Python DSL to instantiate template libraries.
//
// The generated template functions use tile_buf parameters. After this pass,
// the Inline pass inlines the template body, and FoldTileBufIntrinsics
// resolves tile_buf_addr / tile_valid_rows / tile_valid_cols.
//
// Workflow per tile op:
//   1. Extract SpecKey from ALL operands' tile_buf types.
//   2. Invoke Python DSL helper to generate a specialized MLIR function
//      (with tile_buf parameters).
//   3. Parse the generated MLIR and clone the function into the module.
//   4. Replace the original tile op with func.call, passing tile_buf
//      operands directly (no type bridging needed).
//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <unistd.h>

extern "C" {
extern char **environ;
}

using namespace mlir;

namespace mlir {
namespace pto {
  namespace func = ::mlir::func;

  #define GEN_PASS_DEF_EXPANDTILEOP
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

// ============================================================================
// OperandTypeInfo: captures the tile_buf type info for one operand.
// ============================================================================
struct OperandTypeInfo {
  std::string dtype;
  SmallVector<int64_t, 2> shape;
  int32_t blayout = 0;
  int32_t slayout = 0;
  int32_t fractal = 0;
  int32_t pad = 0;

  bool operator==(const OperandTypeInfo &rhs) const {
    return dtype == rhs.dtype && shape == rhs.shape &&
           blayout == rhs.blayout && slayout == rhs.slayout &&
           fractal == rhs.fractal && pad == rhs.pad;
  }
};

// ============================================================================
// SpecKey: identifies a specialized template instance using ALL operands.
// ============================================================================
struct SpecKey {
  std::string opName;
  SmallVector<OperandTypeInfo, 4> operands;

  bool operator==(const SpecKey &rhs) const {
    return opName == rhs.opName && operands == rhs.operands;
  }
};

struct SpecKeyInfo : public llvm::DenseMapInfo<SpecKey> {
  static inline SpecKey getEmptyKey() { return {"", {}}; }
  static inline SpecKey getTombstoneKey() { return {"__tombstone__", {}}; }
  static unsigned getHashValue(const SpecKey &key) {
    unsigned h = llvm::hash_value(key.opName);
    for (const auto &op : key.operands) {
      h = llvm::hash_combine(h, op.dtype, op.blayout, op.slayout,
                              op.fractal, op.pad);
      for (int64_t d : op.shape)
        h = llvm::hash_combine(h, d);
    }
    return h;
  }
  static bool isEqual(const SpecKey &lhs, const SpecKey &rhs) {
    return lhs == rhs;
  }
};

// ============================================================================
// Helpers
// ============================================================================
static std::string getDtypeString(Type elemTy) {
  if (elemTy.isF32()) return "f32";
  if (elemTy.isF16()) return "f16";
  if (elemTy.isBF16()) return "bf16";
  if (elemTy.isSignlessInteger(32)) return "i32";
  if (elemTy.isSignlessInteger(16)) return "i16";
  if (elemTy.isSignlessInteger(8)) return "i8";
  return "";
}

static StringRef getTileOpName(Operation *op) {
  return op->getName().stripDialect();
}

static std::string getMemorySpaceString(pto::TileBufType tbTy) {
  auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
  if (!msAttr) return "ub";
  if (msAttr.getAddressSpace() == pto::AddressSpace::GM) return "gm";
  return "ub";
}

static std::optional<OperandTypeInfo>
buildOperandTypeInfo(pto::TileBufType tbTy) {
  OperandTypeInfo info;
  info.dtype = getDtypeString(tbTy.getElementType());
  if (info.dtype.empty())
    return std::nullopt;
  info.shape.assign(tbTy.getShape().begin(), tbTy.getShape().end());
  if (auto config = tbTy.getConfigAttr()) {
    info.blayout = static_cast<int32_t>(config.getBLayout().getValue());
    info.slayout = static_cast<int32_t>(config.getSLayout().getValue());
    info.fractal = config.getSFractalSize()
                       ? static_cast<int32_t>(config.getSFractalSize().getInt())
                       : 0;
    info.pad = static_cast<int32_t>(config.getPad().getValue());
  }
  return info;
}

static std::optional<SpecKey> buildSpecKey(Operation *op) {
  SpecKey key;
  key.opName = getTileOpName(op).str();

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto tbTy = dyn_cast<pto::TileBufType>(op->getOperand(i).getType());
    if (!tbTy)
      return std::nullopt;
    auto info = buildOperandTypeInfo(tbTy);
    if (!info)
      return std::nullopt;
    key.operands.push_back(*info);
  }
  if (key.operands.empty())
    return std::nullopt;

  return key;
}

// ============================================================================
// ExpandState: runtime state for a single pass invocation.
// ============================================================================
struct ExpandState {
  std::vector<OwningOpRef<ModuleOp>> parsedModules;
  llvm::DenseMap<SpecKey, func::FuncOp, SpecKeyInfo> specCache;

  std::string tilelangPath;
  std::string tilelangPkgPath;
  std::string pythonExe;

  func::FuncOp invokeTilelangDSL(const SpecKey &key, Operation *tileOp,
                                  ModuleOp mod, MLIRContext *ctx);

  LogicalResult expandTileOpsInFunction(func::FuncOp func, ModuleOp mod,
                                        MLIRContext *ctx);
};

// ============================================================================
// The Pass
// ============================================================================
struct ExpandTileOpPass
    : public mlir::pto::impl::ExpandTileOpBase<ExpandTileOpPass> {
  using ExpandTileOpBase::ExpandTileOpBase;

  void runOnOperation() override;
};

// ============================================================================
// Invoke Python DSL helper to generate a specialized template function.
// ============================================================================
func::FuncOp ExpandState::invokeTilelangDSL(const SpecKey &key,
                                              Operation *tileOp,
                                              ModuleOp mod, MLIRContext *ctx) {
  // Check cache first.
  auto cacheIt = specCache.find(key);
  if (cacheIt != specCache.end())
    return cacheIt->second;

  // 1. Locate the Python executable.
  auto pythonPath = llvm::sys::findProgramByName(pythonExe);
  if (!pythonPath) {
    llvm::errs() << "ExpandTileOp: cannot find '" << pythonExe << "'\n";
    return nullptr;
  }

  // 2. Build shape string from the first operand (e.g. "16,64").
  //    TODO: extend expand_helper to accept per-operand shapes if needed.
  const auto &firstOp = key.operands[0];
  std::string shapeStr;
  for (unsigned i = 0; i < firstOp.shape.size(); ++i) {
    if (i > 0) shapeStr += ",";
    shapeStr += std::to_string(firstOp.shape[i]);
  }

  // Get memory space from the first tile_buf operand.
  auto firstTbTy = dyn_cast<pto::TileBufType>(tileOp->getOperand(0).getType());
  std::string memSpace = firstTbTy ? getMemorySpaceString(firstTbTy) : "ub";

  // 3. Create temp file for stdout redirect.
  SmallString<128> tmpPath;
  int tmpFD;
  if (auto ec = llvm::sys::fs::createTemporaryFile("tilelang_expand", "mlir",
                                                     tmpFD, tmpPath)) {
    llvm::errs() << "ExpandTileOp: cannot create temp file: "
                 << ec.message() << "\n";
    return nullptr;
  }
  ::close(tmpFD);

  // 4. Build command args.
  std::string opName = "pto." + key.opName;
  SmallVector<StringRef> args = {
      *pythonPath, "-m", "tilelang_dsl.expand_helper",
      "--template-dir", tilelangPath,
      "--op",           opName,
      "--dtype",        firstOp.dtype,
      "--shape",        shapeStr,
      "--memory-space", memSpace,
  };

  // 5. Set up environment with PYTHONPATH.
  std::optional<StringRef> redirects[] = {std::nullopt, StringRef(tmpPath),
                                          std::nullopt};

  SmallVector<StringRef> envp;
  std::string pythonPathEnv;
  std::vector<std::string> envStorage;
  bool hasPythonPath = !tilelangPkgPath.empty();
  if (hasPythonPath) {
    const char *existingPath = ::getenv("PYTHONPATH");
    pythonPathEnv = "PYTHONPATH=" + tilelangPkgPath;
    if (existingPath && existingPath[0] != '\0') {
      pythonPathEnv += ":";
      pythonPathEnv += existingPath;
    }
    for (char **e = environ; *e; ++e) {
      StringRef entry(*e);
      if (entry.starts_with("PYTHONPATH="))
        continue;
      envStorage.push_back(std::string(entry));
    }
    envStorage.push_back(pythonPathEnv);
    for (auto &s : envStorage)
      envp.push_back(s);
  }

  // 6. Execute.
  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(
      *pythonPath, args,
      hasPythonPath ? std::optional<ArrayRef<StringRef>>(envp) : std::nullopt,
      redirects, /*secondsToWait=*/30, /*memoryLimit=*/0, &errMsg);

  if (rc != 0) {
    llvm::errs() << "ExpandTileOp: tilelang DSL helper failed (rc=" << rc
                 << "): " << errMsg << "\n";
    llvm::sys::fs::remove(tmpPath);
    return nullptr;
  }

  // 7. Read the generated MLIR.
  auto bufOrErr = llvm::MemoryBuffer::getFile(tmpPath);
  llvm::sys::fs::remove(tmpPath);
  if (!bufOrErr) {
    llvm::errs() << "ExpandTileOp: cannot read DSL output\n";
    return nullptr;
  }
  StringRef mlirText = (*bufOrErr)->getBuffer();
  if (mlirText.empty()) {
    llvm::errs() << "ExpandTileOp: empty DSL output\n";
    return nullptr;
  }

  // 8. Parse the MLIR text.
  auto parsedMod = parseSourceString<ModuleOp>(mlirText, ctx);
  if (!parsedMod) {
    llvm::errs() << "ExpandTileOp: failed to parse DSL output\n";
    return nullptr;
  }

  // 9. Find func.func in the parsed module and clone into target module.
  func::FuncOp srcFn;
  for (auto fn : parsedMod->getOps<func::FuncOp>()) {
    srcFn = fn;
    break;
  }
  if (!srcFn) {
    llvm::errs() << "ExpandTileOp: no func.func in DSL output\n";
    return nullptr;
  }

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(mod.getBody());
  IRMapping mapping;
  auto cloned = cast<func::FuncOp>(builder.clone(*srcFn, mapping));

  // Build a unique name from all operand types.
  std::string uniqueName = "__pto_tilelang_" + key.opName;
  for (const auto &op : key.operands) {
    uniqueName += "_" + op.dtype;
    for (int64_t d : op.shape)
      uniqueName += "_" + std::to_string(d);
  }
  cloned.setName(uniqueName);
  cloned.setVisibility(SymbolTable::Visibility::Private);
  // The pto.tilelang.instance attribute should already be set by the
  // TileLang DSL frontend in the generated MLIR. Verify it exists.
  if (!cloned->hasAttr("pto.tilelang.instance")) {
    llvm::errs() << "ExpandTileOp: warning: DSL output function @"
                 << cloned.getSymName()
                 << " missing pto.tilelang.instance attribute\n";
  }

  // Keep the parsed module alive.
  parsedModules.push_back(std::move(parsedMod));

  specCache[key] = cloned;
  return cloned;
}

// ============================================================================
// Expand tile ops in a single function.
// ============================================================================
LogicalResult ExpandState::expandTileOpsInFunction(func::FuncOp func,
                                                   ModuleOp mod,
                                                   MLIRContext *ctx) {
  OpBuilder builder(ctx);

  // Collect tile ops first (avoid modifying while iterating).
  SmallVector<Operation *, 16> tileOps;
  func.walk([&](Operation *op) {
    if (isa<pto::OpPipeInterface>(op))
      tileOps.push_back(op);
  });

  for (auto *op : tileOps) {
    auto specKeyOpt = buildSpecKey(op);
    if (!specKeyOpt) {
      op->emitWarning("ExpandTileOp: cannot build specialization key, skipping");
      continue;
    }

    // Invoke tilelang DSL (with caching).
    func::FuncOp dslFn = invokeTilelangDSL(*specKeyOpt, op, mod, ctx);
    if (!dslFn) {
      StringRef opName = getTileOpName(op);
      op->emitWarning("ExpandTileOp: no tilelang template for " + opName +
                       ", skipping");
      continue;
    }

    // Replace tile op with func.call, passing tile_buf operands directly.
    builder.setInsertionPoint(op);
    SmallVector<Value> operands(op->getOperands());
    builder.create<func::CallOp>(op->getLoc(), dslFn, operands);
    op->erase();
  }

  return success();
}

// ============================================================================
// Main entry point.
// ============================================================================
void ExpandTileOpPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = &getContext();

  if (tilelangPath.empty()) {
    return;
  }

  ExpandState state;
  state.tilelangPath = std::string(tilelangPath);
  state.tilelangPkgPath = std::string(tilelangPkgPath);
  state.pythonExe = std::string(pythonExe);

  for (auto func : mod.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    if (failed(state.expandTileOpsInFunction(func, mod, ctx)))
      return signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createExpandTileOpPass() {
  return std::make_unique<ExpandTileOpPass>();
}

std::unique_ptr<Pass>
createExpandTileOpPass(const ExpandTileOpOptions &options) {
  return std::make_unique<ExpandTileOpPass>(options);
}

} // namespace pto
} // namespace mlir

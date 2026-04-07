#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINLINELIBCALL
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrInstVariantId =
    "pto.oplib.instance.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrInstOp = "pto.oplib.instance.op";
static constexpr llvm::StringLiteral kOpLibAttrInstDType =
    "pto.oplib.instance.dtype";
static constexpr llvm::StringLiteral kErrInstanceBodyMissing =
    "E_OPLIB_INSTANCE_BODY_MISSING";

static bool isInstanceFunc(func::FuncOp fn) {
  return fn->hasAttr(kOpLibAttrInstVariantId);
}

static bool isTilelangFunc(func::FuncOp fn) {
  return fn->hasAttr("pto.tilelang.instance");
}

static bool isInlineableLibFunc(func::FuncOp fn) {
  return isInstanceFunc(fn) || isTilelangFunc(fn);
}

static Value maybeUnwrapCastToExpected(Value operand, Type expectedType) {
  if (operand.getType() == expectedType)
    return operand;

  auto cast = operand.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return operand;

  if (cast.getOperand(0).getType() == expectedType)
    return cast.getOperand(0);
  return operand;
}

static Operation *cloneOpForInlineWithFix(OpBuilder &builder, Operation &op,
                                          IRMapping &mapping) {
  if (auto alloc = dyn_cast<pto::AllocTileOp>(&op)) {
    auto mapOperand = [&](Value operand, Type expectedType) -> Value {
      if (!operand)
        return Value();
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped)
        mapped = operand;
      return maybeUnwrapCastToExpected(mapped, expectedType);
    };

    Value mappedAddr = mapOperand(
        alloc.getAddr(), alloc.getAddr() ? alloc.getAddr().getType() : Type());
    Value mappedValidRow = mapOperand(
        alloc.getValidRow(),
        alloc.getValidRow() ? alloc.getValidRow().getType() : Type());
    Value mappedValidCol = mapOperand(
        alloc.getValidCol(),
        alloc.getValidCol() ? alloc.getValidCol().getType() : Type());

    auto cloned = builder.create<pto::AllocTileOp>(
        alloc.getLoc(), alloc.getType(), mappedAddr, mappedValidRow,
        mappedValidCol);
    cloned->setAttrs(alloc->getAttrs());
    return cloned.getOperation();
  }

  return builder.clone(op, mapping);
}

static void eraseDeadBridgeCasts(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<UnrealizedConversionCastOp, 8> deadUnrealized;
    func.walk([&](UnrealizedConversionCastOp cast) {
      if (cast->use_empty())
        deadUnrealized.push_back(cast);
    });

    SmallVector<memref::CastOp, 8> deadMemrefCasts;
    func.walk([&](memref::CastOp cast) {
      if (cast->use_empty())
        deadMemrefCasts.push_back(cast);
    });

    if (deadUnrealized.empty() && deadMemrefCasts.empty())
      break;

    for (UnrealizedConversionCastOp cast : llvm::reverse(deadUnrealized))
      cast.erase();
    for (memref::CastOp cast : llvm::reverse(deadMemrefCasts))
      cast.erase();
    changed = true;
  }
}

static LogicalResult inlineCall(func::CallOp call, func::FuncOp callee) {
  if (call.getNumResults() != 0)
    return call.emitOpError("OP-Lib inline expects call without results");
  if (callee.isExternal())
    return call.emitOpError("callee must have a body before inlining");

  Block &entry = callee.getBody().front();
  if (entry.getNumArguments() != call.getNumOperands())
    return call.emitOpError("callee argument count mismatch during inlining");

  OpBuilder builder(call);
  IRMapping mapping;
  for (auto [arg, operand] :
       llvm::zip(entry.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  for (Operation &op : entry.without_terminator()) {
    Operation *newOp = cloneOpForInlineWithFix(builder, op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }

  call.erase();
  return success();
}

struct PTOInlineLibCallPass
    : public pto::impl::PTOInlineLibCallBase<PTOInlineLibCallPass> {
  using pto::impl::PTOInlineLibCallBase<
      PTOInlineLibCallPass>::PTOInlineLibCallBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int inlinedCalls = 0;
    int touchedFuncs = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (isInlineableLibFunc(func))
        continue;
      if (func.empty())
        continue;

      SmallVector<func::CallOp, 16> calls;
      func.walk([&](func::CallOp call) { calls.push_back(call); });

      bool changedThisFunc = false;
      for (func::CallOp oldCall : calls) {
        if (!oldCall || !oldCall->getBlock())
          continue;

        auto calleeAttr = oldCall.getCalleeAttr();
        if (!calleeAttr)
          continue;

        func::FuncOp callee =
            module.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
        if (!callee || !isInlineableLibFunc(callee))
          continue;

        if (callee.isExternal()) {
          oldCall.emitError() << kErrInstanceBodyMissing
                              << ": OP-Lib instance body is missing for @"
                              << callee.getSymName();
          if (auto variant =
                  callee->getAttrOfType<StringAttr>(kOpLibAttrInstVariantId)) {
            oldCall.emitRemark() << "variant_id=" << variant.getValue();
          }
          if (auto op = callee->getAttrOfType<StringAttr>(kOpLibAttrInstOp)) {
            oldCall.emitRemark() << "op=" << op.getValue();
          }
          if (auto dtype =
                  callee->getAttrOfType<StringAttr>(kOpLibAttrInstDType)) {
            oldCall.emitRemark() << "dtype=" << dtype.getValue();
          }
          signalPassFailure();
          return;
        }

        func::CallOp call = oldCall;
        SmallVector<Value, 4> concreteOperands;
        concreteOperands.reserve(call.getNumOperands());
        for (auto [operand, expectedTy] : llvm::zip(
                 call.getOperands(), callee.getFunctionType().getInputs())) {
          concreteOperands.push_back(
              maybeUnwrapCastToExpected(operand, expectedTy));
        }

        OpBuilder builder(call);
        auto newCall = builder.create<func::CallOp>(call.getLoc(), callee,
                                                    concreteOperands);
        call.erase();

        if (failed(inlineCall(newCall, callee))) {
          signalPassFailure();
          return;
        }

        ++inlinedCalls;
        changedThisFunc = true;
        if (debug) {
          llvm::errs() << "[op-fusion] inline-libcall: inlined @"
                       << callee.getSymName() << " into @" << func.getSymName()
                       << "\n";
        }
      }

      if (changedThisFunc) {
        eraseDeadBridgeCasts(func);
        ++touchedFuncs;
      }
    }

    if (debug) {
      llvm::errs() << "[op-fusion] inline-libcall touched " << touchedFuncs
                   << " function(s), inlined " << inlinedCalls << " call(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOInlineLibCallPass(const PTOInlineLibCallOptions &options) {
  return std::make_unique<PTOInlineLibCallPass>(options);
}

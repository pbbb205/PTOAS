// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_H
#define PTOAS_H

#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <string>

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace mlir::pto {

extern llvm::cl::opt<bool> emitMlirIR;
extern llvm::cl::opt<std::string> ptoTargetArch;
extern llvm::cl::opt<std::string> ptoBackend;
extern llvm::cl::opt<bool> emitVPTO;
extern llvm::cl::opt<bool> ptoPrintSeamIR;
extern llvm::cl::opt<std::string> ptoSeamIRFile;

enum class PTOBackend {
  EmitC,
  VPTO,
};

enum class PTOASCompileResultKind {
  Text,
  VPTOObject,
  MixedObject,
};

struct PTOASCompileResult {
  void reset() {
    textOutput.clear();
    vptoStubSource.clear();
    vptoCubeModule.reset();
    vptoVectorModule.reset();
    kind = PTOASCompileResultKind::Text;
  }

  PTOASCompileResultKind kind = PTOASCompileResultKind::Text;
  std::string textOutput;
  std::string vptoStubSource;
  EmittedLLVMModule vptoCubeModule;
  EmittedLLVMModule vptoVectorModule;
};

int compilePTOASModule(OwningOpRef<ModuleOp> &module, llvm::StringRef arch,
                       PTOBackend backend, int argc, char **argv,
                       PTOASCompileResult &result,
                       bool emitVPTOHostStub = true);
void registerPTOASDialects(DialectRegistry &registry);
void registerPTOASPassesAndCLOptions();
void loadPTOASDialects(MLIRContext &context);

} // namespace mlir::pto

#endif

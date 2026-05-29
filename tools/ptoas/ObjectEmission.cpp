// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ObjectEmission.h"

#include "PTO/Transforms/VPTOLLVMEmitter.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>

namespace {

using llvm::StringRef;

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath =
                                     std::nullopt);

static bool writeTextFile(StringRef path, StringRef content,
                          llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  os << content;
  os.flush();
  return true;
}

static void stripUnsupportedBishengAttrs(llvm::Module &module) {
  for (llvm::Function &function : module) {
    // LLVM 19 prints memory effect attributes in textual form like
    // `memory(none)`. beta.1 Bisheng cannot parse that syntax, so remove only
    // the unsupported memory-effect attribute before serializing the module.
    function.setAttributes(
        function.getAttributes().removeFnAttribute(module.getContext(),
                                                   llvm::Attribute::Memory));
  }
}

static bool writeLLVMModuleFile(llvm::Module &module, StringRef path,
                                llvm::raw_ostream &diagOS) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "Error: failed to open " << path << " for write: "
           << ec.message() << "\n";
    return false;
  }
  stripUnsupportedBishengAttrs(module);
  module.print(os, nullptr);
  os.flush();
  return true;
}

static std::string sanitizeModuleId(llvm::StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  if (out.empty())
    out = "ptoas_fatobj";
  return out;
}

static std::optional<std::string> getAscendHomePath() {
  const char *env = std::getenv("ASCEND_HOME_PATH");
  if (!env || !*env)
    return std::nullopt;
  return std::string(env);
}

static std::optional<std::string> getEnvPath(llvm::StringRef name) {
  const char *env = std::getenv(name.str().c_str());
  if (!env || !*env)
    return std::nullopt;
  return std::string(env);
}

static std::string joinPath(llvm::StringRef lhs, llvm::StringRef rhs) {
  llvm::SmallString<256> joined(lhs);
  llvm::sys::path::append(joined, rhs);
  return std::string(joined.str());
}

static std::optional<std::string> parseCANNVersionInfo(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  llvm::StringRef content = buffer.get()->getBuffer();
  llvm::SmallVector<llvm::StringRef, 16> lines;
  content.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    llvm::StringRef keys[] = {"Version=", "version="};
    for (llvm::StringRef key : keys) {
      if (!line.starts_with(key))
        continue;
      llvm::StringRef value = line.drop_front(key.size()).trim();
      if (value.consume_front("\""))
        value.consume_back("\"");
      if (!value.empty())
        return value.str();
    }
  }
  return std::nullopt;
}

static std::optional<std::string>
discoverCANNVersion(llvm::StringRef ascendHome) {
  for (llvm::StringRef relPath :
       {"compiler/version.info",
        "x86_64-linux/ascend_toolkit_install.info",
        "x86_64-linux/ascend_all_cann_install.info",
        "aarch64-linux/ascend_toolkit_install.info",
        "aarch64-linux/ascend_all_cann_install.info",
        "ascend_toolkit_install.info", "ascend_all_cann_install.info",
        "opp/version.info"}) {
    if (auto version = parseCANNVersionInfo(joinPath(ascendHome, relPath)))
      return version;
  }
  return std::nullopt;
}

static bool usesBeta1VPTOPublicABI(llvm::StringRef cannVersion) {
  return cannVersion == "9.0.0-beta.1";
}

static llvm::StringRef getVPTOVectorPublicABISuffix(
    const mlir::pto::ObjectEmissionToolchain &toolchain) {
  if (!toolchain.vptoVectorPublicABISuffix.empty())
    return toolchain.vptoVectorPublicABISuffix;
  return ".vector";
}

static llvm::StringRef getVPTOCubePublicABISuffix(
    const mlir::pto::ObjectEmissionToolchain &toolchain) {
  if (!toolchain.vptoCubePublicABISuffix.empty())
    return toolchain.vptoCubePublicABISuffix;
  return ".cube";
}

static std::optional<std::string> locateProgram(llvm::StringRef envPath,
                                                llvm::StringRef fallbackName) {
  if (!envPath.empty() && llvm::sys::fs::exists(envPath))
    return envPath.str();
  if (auto found = llvm::sys::findProgramByName(fallbackName))
    return *found;
  return std::nullopt;
}

static bool hasPTOISAHeader(llvm::StringRef includeDir) {
  return llvm::sys::fs::exists(joinPath(includeDir, "pto/pto-inst.hpp"));
}

static void addExistingIncludeDir(llvm::SmallVectorImpl<std::string> &dirs,
                                  llvm::StringRef path) {
  if (path.empty() || !llvm::sys::fs::is_directory(path))
    return;
  if (llvm::is_contained(dirs, path))
    return;
  dirs.push_back(path.str());
}

static void addPTOISAIncludeDirs(llvm::SmallVectorImpl<std::string> &dirs,
                                 llvm::StringRef ptoIsaPath) {
  if (ptoIsaPath.empty() || !llvm::sys::fs::is_directory(ptoIsaPath))
    return;
  std::string includeDir = joinPath(ptoIsaPath, "include");
  if (hasPTOISAHeader(includeDir))
    addExistingIncludeDir(dirs, includeDir);
  std::string commonDir = joinPath(ptoIsaPath, "tests/common");
  addExistingIncludeDir(dirs, commonDir);
  if (hasPTOISAHeader(ptoIsaPath))
    addExistingIncludeDir(dirs, ptoIsaPath);
}

static llvm::SmallVector<std::string, 8>
discoverCppIncludeDirs(llvm::StringRef ascendHome,
                       llvm::raw_ostream &diagOS,
                       std::string &ptoIsaPath) {
  llvm::SmallVector<std::string, 8> includeDirs;
  if (auto env = getEnvPath("PTO_ISA_PATH"))
    ptoIsaPath = *env;
  else if (auto env = getEnvPath("PTO_ISA_ROOT"))
    ptoIsaPath = *env;

  addPTOISAIncludeDirs(includeDirs, ptoIsaPath);
  addExistingIncludeDir(includeDirs, joinPath(ascendHome, "include"));
  std::string driverPath =
      getEnvPath("ASCEND_DRIVER_PATH").value_or("/usr/local/Ascend/driver");
  addExistingIncludeDir(includeDirs, joinPath(driverPath, "kernel/inc"));

  if (ptoIsaPath.empty()) {
    diagOS << "Warning: PTO_ISA_PATH/PTO_ISA_ROOT is not set; "
              "C++ device object emission may fail to include pto/pto-inst.hpp.\n";
  } else if (includeDirs.empty()) {
    diagOS << "Warning: no PTO-ISA include directory containing "
              "pto/pto-inst.hpp was found under "
           << ptoIsaPath << ".\n";
  }
  return includeDirs;
}

class VPTOFatobjToolchain;

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS);
static bool compileHostStubToObject(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const VPTOFatobjToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS);
static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS);

static llvm::StringRef
getTargetCPU(mlir::pto::ObjectEmissionDeviceTarget target) {
  switch (target) {
  case mlir::pto::ObjectEmissionDeviceTarget::Vector:
    return "dav-c310-vec";
  case mlir::pto::ObjectEmissionDeviceTarget::Cube:
    return "dav-c310-cube";
  }
  llvm_unreachable("unknown object emission device target");
}

class VPTOFatobjToolchain {
public:
  explicit VPTOFatobjToolchain(
      const mlir::pto::ObjectEmissionToolchain &toolchain)
      : ascendHomePath(toolchain.ascendHomePath),
        bishengPath(toolchain.bishengPath),
        bishengCc1Path(toolchain.bishengCc1Path),
        cceLdPath(toolchain.cceLdPath),
        ldLldPath(toolchain.ldLldPath),
        resourceDirPath(toolchain.resourceDirPath),
        resourceIncludeDirPath(toolchain.resourceIncludeDirPath),
        cceStubDirPath(toolchain.cceStubDirPath),
        bishengCompilerBinDirPath(toolchain.bishengCompilerBinDirPath),
        ptoIsaPath(toolchain.ptoIsaPath),
        cannVersion(toolchain.cannVersion),
        vptoVectorPublicABISuffix(toolchain.vptoVectorPublicABISuffix),
        vptoCubePublicABISuffix(toolchain.vptoCubePublicABISuffix) {
    cppIncludeDirs.assign(toolchain.cppIncludeDirs.begin(),
                          toolchain.cppIncludeDirs.end());
  }

  static std::optional<VPTOFatobjToolchain>
  create(llvm::raw_ostream &diagOS) {
    std::optional<std::string> ascendHome = getAscendHomePath();
    if (!ascendHome) {
      diagOS << "Error: ASCEND_HOME_PATH is required for VPTO fatobj emission.\n";
      return std::nullopt;
    }

    VPTOFatobjToolchain toolchain(*ascendHome, diagOS);
    if (!toolchain.validate(diagOS))
      return std::nullopt;
    return toolchain;
  }

  const std::string &ascendHome() const { return ascendHomePath; }
  const std::string &bisheng() const { return bishengPath; }
  const std::string &bishengCc1() const { return bishengCc1Path; }
  const std::string &cceLd() const { return cceLdPath; }
  const std::string &ldLld() const { return ldLldPath; }
  const std::string &resourceDir() const { return resourceDirPath; }
  const std::string &resourceIncludeDir() const {
    return resourceIncludeDirPath;
  }
  const std::string &cceStubDir() const { return cceStubDirPath; }
  const std::string &bishengCompilerBinDir() const {
    return bishengCompilerBinDirPath;
  }

  mlir::pto::ObjectEmissionToolchain toPublicToolchain() const {
    mlir::pto::ObjectEmissionToolchain toolchain;
    toolchain.ascendHomePath = ascendHomePath;
    toolchain.bishengPath = bishengPath;
    toolchain.bishengCc1Path = bishengCc1Path;
    toolchain.cceLdPath = cceLdPath;
    toolchain.ldLldPath = ldLldPath;
    toolchain.resourceDirPath = resourceDirPath;
    toolchain.resourceIncludeDirPath = resourceIncludeDirPath;
    toolchain.cceStubDirPath = cceStubDirPath;
    toolchain.bishengCompilerBinDirPath = bishengCompilerBinDirPath;
    toolchain.ptoIsaPath = ptoIsaPath;
    toolchain.cannVersion = cannVersion;
    toolchain.vptoVectorPublicABISuffix = vptoVectorPublicABISuffix;
    toolchain.vptoCubePublicABISuffix = vptoCubePublicABISuffix;
    toolchain.cppIncludeDirs.assign(cppIncludeDirs.begin(),
                                    cppIncludeDirs.end());
    return toolchain;
  }

private:
  explicit VPTOFatobjToolchain(llvm::StringRef ascendHome,
                               llvm::raw_ostream &diagOS)
      : ascendHomePath(ascendHome.str()),
        bishengPath(joinPath(ascendHomePath, "bin/bisheng")),
        bishengCc1Path(
            joinPath(ascendHomePath, "tools/bisheng_compiler/bin/bisheng")),
        cceLdPath(joinPath(ascendHomePath, "bin/cce-ld")),
        ldLldPath(
            locateProgram(joinPath(ascendHomePath, "bin/ld.lld"), "ld.lld")
                .value_or(std::string())),
        resourceDirPath(joinPath(
            ascendHomePath, "tools/bisheng_compiler/lib/clang/15.0.5")),
        resourceIncludeDirPath(joinPath(resourceDirPath, "include")),
        cceStubDirPath(joinPath(resourceIncludeDirPath, "cce_stub")),
        bishengCompilerBinDirPath(
            joinPath(ascendHomePath, "tools/bisheng_compiler/bin")) {
    cannVersion =
        discoverCANNVersion(ascendHomePath).value_or("9.0.0-beta.1");
    if (usesBeta1VPTOPublicABI(cannVersion)) {
      vptoVectorPublicABISuffix = "_mix_aiv";
      vptoCubePublicABISuffix = "_mix_aic";
    } else {
      vptoVectorPublicABISuffix = ".vector";
      vptoCubePublicABISuffix = ".cube";
    }
    cppIncludeDirs = discoverCppIncludeDirs(ascendHomePath, diagOS,
                                            ptoIsaPath);
  }

  bool validate(llvm::raw_ostream &diagOS) const {
    if (!llvm::sys::fs::exists(bishengPath)) {
      diagOS << "Error: unable to locate bisheng: " << bishengPath << "\n";
      return false;
    }
    if (!llvm::sys::fs::exists(bishengCc1Path)) {
      diagOS << "Error: unable to locate bisheng cc1 frontend: "
             << bishengCc1Path << "\n";
      return false;
    }
    if (!llvm::sys::fs::exists(cceLdPath)) {
      diagOS << "Error: unable to locate cce-ld: " << cceLdPath << "\n";
      return false;
    }
    if (ldLldPath.empty() || !llvm::sys::fs::exists(ldLldPath)) {
      diagOS << "Error: unable to locate ld.lld.\n";
      return false;
    }
    return true;
  }

  std::string ascendHomePath;
  std::string bishengPath;
  std::string bishengCc1Path;
  std::string cceLdPath;
  std::string ldLldPath;
  std::string resourceDirPath;
  std::string resourceIncludeDirPath;
  std::string cceStubDirPath;
  std::string bishengCompilerBinDirPath;
  std::string ptoIsaPath;
  std::string cannVersion;
  std::string vptoVectorPublicABISuffix;
  std::string vptoCubePublicABISuffix;
  llvm::SmallVector<std::string, 8> cppIncludeDirs;
};

class VPTOFatobjArtifacts {
public:
  explicit VPTOFatobjArtifacts(mlir::pto::TempFileRegistry &tempFiles)
      : tempFiles(tempFiles) {}

  bool emitStubSource(StringRef stubSource, llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-host-stub", ".cpp", stubPath, diagOS)))
      return false;
    if (!writeTextFile(stubPath, stubSource, diagOS))
      return false;
    return true;
  }

  bool initCommandLogs(llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-stderr", ".log", stderrPath, diagOS)))
      return false;
    return true;
  }

  bool emitCubeObject(llvm::Module *module,
                      const VPTOFatobjToolchain &toolchain,
                      llvm::raw_ostream &diagOS) {
    if (!module)
      return true;
    if (failed(tempFiles.create("ptoas-device", ".ll", cubeLLPath, diagOS)))
      return false;
    if (failed(tempFiles.create("ptoas-device", ".o", cubeObjPath, diagOS)))
      return false;
    return succeeded(mlir::pto::emitVPTOCubeDeviceObject(
        *module, cubeLLPath, cubeObjPath, toolchain.toPublicToolchain(),
        stderrPath, diagOS));
  }

  bool emitVectorObject(llvm::Module *module,
                        const VPTOFatobjToolchain &toolchain,
                        llvm::raw_ostream &diagOS) {
    if (!module)
      return true;
    if (failed(tempFiles.create("ptoas-device", ".ll", vectorLLPath, diagOS)))
      return false;
    if (failed(tempFiles.create("ptoas-device", ".o", vectorObjPath, diagOS)))
      return false;
    return succeeded(mlir::pto::emitVPTOVectorDeviceObject(
        *module, vectorLLPath, vectorObjPath, toolchain.toPublicToolchain(),
        stderrPath, diagOS));
  }

  bool mergeDeviceObjects(const VPTOFatobjToolchain &toolchain,
                          llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, 2> deviceObjPaths;
    if (!cubeObjPath.empty())
      deviceObjPaths.push_back(cubeObjPath);
    if (!vectorObjPath.empty())
      deviceObjPaths.push_back(vectorObjPath);
    if (deviceObjPaths.empty()) {
      diagOS << "Error: VPTO fatobj emission requires at least one device module.\n";
      return false;
    }
    if (failed(tempFiles.create("ptoas-device-merged", ".o",
                                mergedDeviceObjPath, diagOS)))
      return false;
    return ::mergeDeviceObjects(deviceObjPaths, mergedDeviceObjPath,
                                toolchain.ldLld(), stderrPath, diagOS);
  }

  bool compileHostStub(const VPTOFatobjToolchain &toolchain,
                       llvm::StringRef moduleId,
                       llvm::StringRef targetCPU,
                       llvm::raw_ostream &diagOS) {
    if (failed(tempFiles.create("ptoas-host-stub", ".o", hostStubObjPath,
                                diagOS)))
      return false;
    return compileHostStubToObject(stubPath, hostStubObjPath, moduleId,
                                   targetCPU, toolchain, mergedDeviceObjPath,
                                   stderrPath, diagOS);
  }

  bool compileHostStubToFatobj(const VPTOFatobjToolchain &toolchain,
                               llvm::StringRef moduleId,
                               llvm::StringRef targetCPU,
                               llvm::StringRef outputPath,
                               llvm::raw_ostream &diagOS) {
    return compileHostStubToObject(stubPath, outputPath, moduleId, targetCPU,
                                   toolchain, mergedDeviceObjPath, stderrPath,
                                   diagOS);
  }

  bool repackFatObj(const VPTOFatobjToolchain &toolchain,
                    llvm::StringRef moduleId, llvm::StringRef targetCPU,
                    llvm::StringRef outPath, llvm::raw_ostream &diagOS) {
    llvm::SmallVector<std::string, 16> args = {
        toolchain.cceLd(),
        toolchain.ldLld(),
        "-x",
        "-cce-lite-bin-module-id",
        moduleId.str(),
        std::string("-cce-aicore-arch=") + targetCPU.str(),
        // Keep device externals relocatable until the final
        // --cce-fatobj-link link step.
        "-dc",
        "-r",
        "-o",
        outPath.str(),
        "-cce-stub-dir",
        toolchain.cceStubDir(),
        "-cce-install-dir",
        toolchain.bishengCompilerBinDir(),
        "-cce-inputs-number",
        "1",
        hostStubObjPath,
    };
    return runCommandWithStderr(toolchain.cceLd(), args, stderrPath, diagOS,
                                "fatobj repack");
  }

private:
  mlir::pto::TempFileRegistry &tempFiles;
  std::string cubeLLPath;
  std::string cubeObjPath;
  std::string vectorLLPath;
  std::string vectorObjPath;
  std::string mergedDeviceObjPath;
  std::string stderrPath;
  std::string stubPath;
  std::string hostStubObjPath;
};

static bool runCommandWithStderr(llvm::StringRef program,
                                 llvm::ArrayRef<std::string> ownedArgs,
                                 llvm::StringRef stderrPath,
                                 llvm::raw_ostream &diagOS,
                                 llvm::StringRef what,
                                 std::optional<llvm::StringRef> stdinPath) {
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.reserve(ownedArgs.size());
  for (const std::string &arg : ownedArgs)
    args.push_back(arg);
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      stdinPath, std::nullopt, stderrPath};

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(program, args, std::nullopt, redirects, 0,
                                     0, &execErr, &execFailed);
  if (!execFailed && rc == 0)
    return true;

  diagOS << "Error: " << what << " failed\n";
  diagOS << "Command:";
  for (llvm::StringRef arg : args)
    diagOS << " " << arg;
  diagOS << "\n";
  if (!execErr.empty())
    diagOS << execErr << "\n";
  if (auto buffer = llvm::MemoryBuffer::getFile(stderrPath))
    diagOS << buffer.get()->getBuffer() << "\n";
  return false;
}

static bool compileDeviceLLVMToObject(llvm::StringRef llPath,
                                      llvm::StringRef outObjPath,
                                      llvm::StringRef targetCPU,
                                      llvm::StringRef bishengPath,
                                      llvm::StringRef stderrPath,
                                      llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 16> args = {
      bishengPath.str(),
      "--target=hiipu64-hisilicon-cce",
      std::string("-march=") + targetCPU.str(),
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "--cce-aicore-only",
      "-O2",
      "-dc",
      "-mllvm",
      "-cce-dyn-kernel-stack-size=true",
      "-c",
      "-x",
      "ir",
      "-",
      "-o",
      outObjPath.str(),
  };
  return runCommandWithStderr(bishengPath, args, stderrPath, diagOS,
                              "device LLVM compilation", llPath);
}

static bool compileCppDeviceSourceToObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    llvm::StringRef targetCPU, const mlir::pto::ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengPath,
      "-xcce",
      "-fenable-matrix",
      "--cce-aicore-enable-tl",
      "--cce-aicore-only",
      "-fPIC",
      "-Xhost-start",
      "-Xhost-end",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      std::string("--cce-aicore-arch=") + targetCPU.str(),
      "-DREGISTER_BASE",
      "-std=c++17",
      "-dc",
  };
  for (const std::string &includeDir : toolchain.cppIncludeDirs)
    args.push_back("-I" + includeDir);
  args.push_back("-c");
  args.push_back(cppPath.str());
  args.push_back("-o");
  args.push_back(outObjPath.str());

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "C++ device compilation");
}

static bool compileCppDeviceSourceToFatobj(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    const mlir::pto::ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengPath,
      "-xcce",
      "-fenable-matrix",
      "--cce-aicore-enable-tl",
      "-fPIC",
      "-Xhost-start",
      "-Xhost-end",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "--cce-aicore-arch=dav-c310",
      "-DREGISTER_BASE",
      "-std=c++17",
      "-O2",
      "-dc",
      "-c",
  };
  for (const std::string &includeDir : toolchain.cppIncludeDirs)
    args.push_back("-I" + includeDir);
  args.push_back(cppPath.str());
  args.push_back("-o");
  args.push_back(outObjPath.str());

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "C++ fatobj compilation");
}

static bool compileHostStubToObject(llvm::StringRef stubPath,
                                    llvm::StringRef outObjPath,
                                    llvm::StringRef moduleId,
                                    llvm::StringRef targetCPU,
                                    const VPTOFatobjToolchain &toolchain,
                                    llvm::StringRef deviceObjPath,
                                    llvm::StringRef stderrPath,
                                    llvm::raw_ostream &diagOS) {
  std::string coverageDir = ".";
  std::string debugDir = ".";
  std::string hostTriple = llvm::sys::getProcessTriple();

  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengCc1(),
      "-cc1",
      "-triple",
      hostTriple,
      "-target-cpu",
      llvm::sys::getHostCPUName().str(),
      "-fcce-aicpu-legacy-launch",
      "-fcce-is-host",
      "-cce-enable-mix",
      "-mllvm",
      "-enable-mix=true",
      "-cce-launch-with-flagv2-impl",
      "-fcce-aicore-arch",
      targetCPU.str(),
      "-fcce-fatobj-compile",
      "-emit-obj",
      "--mrelax-relocations",
      "-disable-free",
      "-clear-ast-before-backend",
      "-disable-llvm-verifier",
      "-discard-value-names",
      "-main-file-name",
      "stub.cpp",
      "-mrelocation-model",
      "pic",
      "-pic-level",
      "2",
      "-fhalf-no-semantic-interposition",
      "-mframe-pointer=none",
      "-fmath-errno",
      "-ffp-contract=on",
      "-fno-rounding-math",
      "-mconstructor-aliases",
      "-funwind-tables=2",
      "-fallow-half-arguments-and-returns",
      "-mllvm",
      "-treat-scalable-fixed-error-as-warning",
      std::string("-fcoverage-compilation-dir=") + coverageDir,
      "-resource-dir",
      toolchain.resourceDir(),
      "-internal-isystem",
      toolchain.resourceIncludeDir(),
      "-include",
      "__clang_cce_runtime_wrapper.h",
      "-D",
      "_FORTIFY_SOURCE=2",
      "-D",
      "REGISTER_BASE",
      "-O2",
      "-Wno-macro-redefined",
      "-Wno-ignored-attributes",
      "-std=c++17",
      "-fdeprecated-macro",
      std::string("-fdebug-compilation-dir=") + debugDir,
      "-ferror-limit",
      "19",
      "-stack-protector",
      "2",
      "-fno-signed-char",
      "-fgnuc-version=4.2.1",
      "-fcxx-exceptions",
      "-fexceptions",
      "-vectorize-loops",
      "-vectorize-slp",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-record-overflow=true",
      "-mllvm",
      "-cce-aicore-addr-transform",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "-fcce-include-aibinary",
      deviceObjPath.str(),
      "-fcce-device-module-id",
      moduleId.str(),
      "-faddrsig",
      "-D__GCC_HAVE_DWARF2_CFI_ASM=1",
      "-o",
      outObjPath.str(),
      "-x",
      "cce",
      stubPath.str(),
  };
  return runCommandWithStderr(toolchain.bishengCc1(), args, stderrPath, diagOS,
                              "host stub compilation");
}

static bool mergeDeviceObjects(llvm::ArrayRef<std::string> deviceObjPaths,
                               llvm::StringRef outObjPath,
                               llvm::StringRef ldLldPath,
                               llvm::StringRef stderrPath,
                               llvm::raw_ostream &diagOS) {
  if (deviceObjPaths.empty())
    return false;

  llvm::SmallVector<std::string, 16> args = {
      ldLldPath.str(),
      "-m",
      "aicorelinux",
      "-Ttext",
      "0",
  };
  for (const std::string &path : deviceObjPaths)
    args.push_back(path);
  args.push_back("-o");
  args.push_back(outObjPath.str());
  args.push_back("-r");
  args.push_back("--allow-multiple-definition");
  return runCommandWithStderr(ldLldPath, args, stderrPath, diagOS,
                              "device object merge");
}

static bool linkFatobjFiles(llvm::ArrayRef<std::string> fatobjPaths,
                            llvm::StringRef outObjPath,
                            const mlir::pto::ObjectEmissionToolchain &toolchain,
                            llvm::StringRef stderrPath,
                            llvm::raw_ostream &diagOS) {
  if (fatobjPaths.empty())
    return false;

  llvm::SmallVector<std::string, 32> args = {
      toolchain.bishengPath,
      "--cce-fatobj-link",
      "--cce-aicore-arch=dav-c310",
      "-r",
      "-o",
      outObjPath.str(),
  };
  for (const std::string &path : fatobjPaths)
    args.push_back(path);

  return runCommandWithStderr(toolchain.bishengPath, args, stderrPath, diagOS,
                              "fatobj link");
}

} // namespace

mlir::pto::TempFileRegistry::~TempFileRegistry() { cleanup(); }

void mlir::pto::TempFileRegistry::cleanup() {
  for (const std::string &path : paths)
    llvm::sys::fs::remove(path);
  paths.clear();
}

mlir::LogicalResult
mlir::pto::TempFileRegistry::create(llvm::StringRef prefix,
                                    llvm::StringRef suffix, std::string &path,
                                    llvm::raw_ostream &diagOS) {
  llvm::SmallString<128> tempPath;
  int fd = -1;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, fd, tempPath);
  if (ec) {
    diagOS << "Error: failed to create temporary file for " << prefix << suffix
           << ": " << ec.message() << "\n";
    return failure();
  }
  llvm::sys::Process::SafelyCloseFileDescriptor(fd);
  path = tempPath.str().str();
  paths.push_back(path);
  return success();
}

mlir::LogicalResult mlir::pto::discoverObjectEmissionToolchain(
    ObjectEmissionToolchain &publicToolchain, llvm::raw_ostream &diagOS) {
  std::optional<VPTOFatobjToolchain> toolchain =
      VPTOFatobjToolchain::create(diagOS);
  if (!toolchain)
    return failure();
  publicToolchain = toolchain->toPublicToolchain();
  return success();
}

mlir::LogicalResult mlir::pto::writeLLVMModule(llvm::Module &module,
                                               llvm::StringRef path,
                                               llvm::raw_ostream &diagOS) {
  return writeLLVMModuleFile(module, path, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::writeCppSource(llvm::StringRef cppSource,
                                              llvm::StringRef path,
                                              llvm::raw_ostream &diagOS) {
  return writeTextFile(path, cppSource, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::writeHostStubSource(
    llvm::StringRef stubSource, llvm::StringRef path,
    llvm::raw_ostream &diagOS) {
  return writeTextFile(path, stubSource, diagOS) ? success() : failure();
}

mlir::LogicalResult mlir::pto::compileCppToDeviceObject(
    llvm::StringRef cppPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  return compileCppDeviceSourceToObject(cppPath, outObjPath,
                                        getTargetCPU(target), toolchain,
                                        stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::compileLLVMToDeviceObject(
    llvm::StringRef llPath, llvm::StringRef outObjPath,
    ObjectEmissionDeviceTarget target, const ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  return compileDeviceLLVMToObject(llPath, outObjPath, getTargetCPU(target),
                                   toolchain.bishengPath, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitCppVectorDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS)))
    return failure();
  return compileCppToDeviceObject(cppPath, outObjPath,
                                  ObjectEmissionDeviceTarget::Vector,
                                  toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitCppCubeDeviceObject(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS)))
    return failure();
  return compileCppToDeviceObject(cppPath, outObjPath,
                                  ObjectEmissionDeviceTarget::Cube,
                                  toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitCppFatobj(
    llvm::StringRef cppSource, llvm::StringRef cppPath,
    llvm::StringRef outObjPath, const ObjectEmissionToolchain &toolchain,
    llvm::StringRef stderrPath, llvm::raw_ostream &diagOS) {
  if (failed(writeCppSource(cppSource, cppPath, diagOS)))
    return failure();
  return compileCppDeviceSourceToFatobj(cppPath, outObjPath, toolchain,
                                        stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitFatobjCCE(
    llvm::StringRef cppSource, llvm::StringRef outputPath,
    const ObjectEmissionToolchain &toolchain, TempFileRegistry &tempFiles,
    llvm::raw_ostream &diagOS) {
  std::string cppPath;
  std::string stderrPath;
  if (failed(tempFiles.create("ptoas-emitc", ".cpp", cppPath, diagOS)) ||
      failed(tempFiles.create("ptoas-emitc-fatobj", ".log", stderrPath,
                              diagOS)))
    return failure();
  return emitCppFatobj(cppSource, cppPath, outputPath, toolchain, stderrPath,
                       diagOS);
}

static bool isVPTOKernelABISymbol(llvm::StringRef name) {
  return name.ends_with("_mix_aiv") || name.ends_with("_mix_aic");
}

static bool isLegacyVPTOPublicABISymbol(llvm::StringRef name) {
  return name.ends_with(".vector") || name.ends_with(".cube");
}

static mlir::LogicalResult renameLLVMFunction(llvm::Module &module,
                                              llvm::StringRef sourceName,
                                              llvm::StringRef abiName,
                                              llvm::raw_ostream &diagOS) {
  if (sourceName == abiName)
    return mlir::success();
  llvm::Function *function = module.getFunction(sourceName);
  if (!function)
    return mlir::success();
  if (llvm::Function *existing = module.getFunction(abiName);
      existing && existing != function) {
    diagOS << "Error: cannot rename LLVM symbol '" << sourceName << "' to '"
           << abiName << "': target symbol already exists.\n";
    return mlir::failure();
  }
  function->setName(abiName);
  return mlir::success();
}

static mlir::LogicalResult applyVPTOLLVMABINames(llvm::Module &module,
                                                 llvm::StringRef suffix,
                                                 llvm::raw_ostream &diagOS) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration() || !function.hasExternalLinkage())
      continue;
    llvm::StringRef name = function.getName();
    if (name.empty() || isVPTOKernelABISymbol(name) ||
        isLegacyVPTOPublicABISymbol(name))
      continue;
    if (failed(renameLLVMFunction(module, name, (name + suffix).str(), diagOS)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::emitVPTOVectorDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const ObjectEmissionToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  if (failed(applyVPTOLLVMABINames(module, getVPTOVectorPublicABISuffix(toolchain),
                                   diagOS)))
    return failure();
  if (failed(writeLLVMModule(module, llPath, diagOS)))
    return failure();
  return compileLLVMToDeviceObject(llPath, outObjPath,
                                   ObjectEmissionDeviceTarget::Vector,
                                   toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitVPTOCubeDeviceObject(
    llvm::Module &module, llvm::StringRef llPath, llvm::StringRef outObjPath,
    const ObjectEmissionToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  if (failed(applyVPTOLLVMABINames(module, getVPTOCubePublicABISuffix(toolchain),
                                   diagOS)))
    return failure();
  if (failed(writeLLVMModule(module, llPath, diagOS)))
    return failure();
  return compileLLVMToDeviceObject(llPath, outObjPath,
                                   ObjectEmissionDeviceTarget::Cube,
                                   toolchain, stderrPath, diagOS);
}

mlir::LogicalResult mlir::pto::emitFatobjLLVM(
    llvm::Module *cubeModule, llvm::Module *vectorModule,
    llvm::StringRef stubSource, llvm::StringRef outputPath,
    llvm::StringRef moduleId, const ObjectEmissionToolchain &toolchain,
    TempFileRegistry &tempFiles, llvm::raw_ostream &diagOS) {
  if (!cubeModule && !vectorModule) {
    diagOS << "Error: VPTO fatobj emission requires at least one LLVM module.\n";
    return failure();
  }

  VPTOFatobjToolchain privateToolchain(toolchain);
  VPTOFatobjArtifacts artifacts(tempFiles);
  if (!artifacts.emitStubSource(stubSource, diagOS))
    return failure();
  if (!artifacts.initCommandLogs(diagOS))
    return failure();
  if (!artifacts.emitCubeObject(cubeModule, privateToolchain, diagOS))
    return failure();
  if (!artifacts.emitVectorObject(vectorModule, privateToolchain, diagOS))
    return failure();
  if (!artifacts.mergeDeviceObjects(privateToolchain, diagOS))
    return failure();

  constexpr llvm::StringLiteral targetCPU = "dav-c310";
  if (!artifacts.compileHostStubToFatobj(privateToolchain, moduleId, targetCPU,
                                         outputPath, diagOS))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::mergeDeviceObjects(
    llvm::ArrayRef<std::string> deviceObjPaths, llvm::StringRef outObjPath,
    const ObjectEmissionToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  return ::mergeDeviceObjects(deviceObjPaths, outObjPath, toolchain.ldLldPath,
                              stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::compileStubToFatobj(
    llvm::StringRef stubPath, llvm::StringRef deviceObjPath,
    llvm::StringRef outputPath, llvm::StringRef moduleId,
    const ObjectEmissionToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  VPTOFatobjToolchain privateToolchain(toolchain);

  constexpr llvm::StringLiteral targetCPU = "dav-c310";
  return compileHostStubToObject(stubPath, outputPath, moduleId, targetCPU,
                                 privateToolchain, deviceObjPath, stderrPath,
                                 diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::linkFatobjs(
    llvm::ArrayRef<std::string> fatobjPaths, llvm::StringRef outputPath,
    const ObjectEmissionToolchain &toolchain, llvm::StringRef stderrPath,
    llvm::raw_ostream &diagOS) {
  return linkFatobjFiles(fatobjPaths, outputPath, toolchain, stderrPath, diagOS)
             ? success()
             : failure();
}

mlir::LogicalResult mlir::pto::emitFatobjLLVMWithRuntime(
    llvm::Module *cubeModule, llvm::Module *vectorModule,
    llvm::StringRef stubSource, llvm::ToolOutputFile &outputFile,
    llvm::raw_ostream &diagOS) {
  if (!cubeModule && !vectorModule) {
    diagOS << "Error: VPTO fatobj emission requires at least one LLVM module.\n";
    return failure();
  }

  std::optional<VPTOFatobjToolchain> toolchain =
      VPTOFatobjToolchain::create(diagOS);
  if (!toolchain)
    return failure();

  TempFileRegistry tempFiles;
  VPTOFatobjArtifacts artifacts(tempFiles);
  if (!artifacts.emitStubSource(stubSource, diagOS))
    return failure();
  if (!artifacts.initCommandLogs(diagOS))
    return failure();

  if (!artifacts.emitCubeObject(cubeModule, *toolchain, diagOS))
    return failure();
  if (!artifacts.emitVectorObject(vectorModule, *toolchain, diagOS))
    return failure();

  if (!artifacts.mergeDeviceObjects(*toolchain, diagOS))
    return failure();

  std::string moduleId = sanitizeModuleId(outputFile.getFilename());
  constexpr llvm::StringLiteral hostTargetCPU = "dav-c310";
  if (!artifacts.compileHostStub(*toolchain, moduleId, hostTargetCPU, diagOS))
    return failure();

  if (!artifacts.repackFatObj(*toolchain, moduleId, hostTargetCPU,
                              outputFile.getFilename(), diagOS))
    return failure();
  outputFile.keep();
  return success();
}

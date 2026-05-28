# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

get_filename_component(_llvm_root "${LLVM_DIR}/../../.." ABSOLUTE)
set(_ptodsl_test_env
  "PYTHONPATH=${MLIR_PYTHON_PACKAGE_DIR}:${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}"
  "LD_LIBRARY_PATH=${_llvm_root}/lib:${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
  "PATH=${CMAKE_BINARY_DIR}/tools/ptoas:${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}"
)

function(add_ptodsl_python_test test_name script_path)
  add_test(
    NAME ${test_name}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/${script_path}"
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "PTODSL"
    ENVIRONMENT "${_ptodsl_test_env}"
  )
endfunction()

# Auto-discover PTODSL Python regressions by filename convention:
#   - ptodsl/tests/test_*.py
file(
  GLOB _ptodsl_test_files
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
)

list(SORT _ptodsl_test_files)

foreach(test_file IN LISTS _ptodsl_test_files)
  file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}" "${test_file}")
  get_filename_component(stem "${test_file}" NAME_WE)

  if(stem MATCHES "^test_")
    string(REGEX REPLACE "^test_" "" stem_suffix "${stem}")
    set(test_name "ptodsl_${stem_suffix}")
  else()
    set(test_name "ptodsl_${stem}")
  endif()

  add_ptodsl_python_test("${test_name}" "${rel_path}")
endforeach()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =========================================================
# TileLang DSL package wiring
# =========================================================

set(TILELANG_DSL_PACKAGE_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/python/tilelang_dsl")
set(TILELANG_DSL_BUILD_ROOT "${CMAKE_BINARY_DIR}/python")
set(TILELANG_DSL_BUILD_PACKAGE_DIR
    "${TILELANG_DSL_BUILD_ROOT}/tilelang_dsl")

add_custom_target(TileLangDSLPackage ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${TILELANG_DSL_BUILD_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${TILELANG_DSL_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${TILELANG_DSL_PACKAGE_SRC_DIR}"
          "${TILELANG_DSL_BUILD_PACKAGE_DIR}"
  COMMENT "Staging tilelang_dsl package into build/python"
  VERBATIM
)

install(
  DIRECTORY "${TILELANG_DSL_PACKAGE_SRC_DIR}"
  DESTINATION "."
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/TileOps"
  DESTINATION "share/ptoas"
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

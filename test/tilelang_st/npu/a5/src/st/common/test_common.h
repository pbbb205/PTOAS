// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace PtoTestCommon {

inline bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer,
                     size_t bufferSize) {
  struct stat sBuf;
  if (stat(filePath.c_str(), &sBuf) == -1) {
    return false;
  }
  if (!S_ISREG(sBuf.st_mode)) {
    return false;
  }

  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  std::filebuf *buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0 || size > bufferSize) {
    return false;
  }
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char *>(buffer), size);
  fileSize = size;
  return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) {
    return false;
  }

  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
  if (fd < 0) {
    return false;
  }

  ssize_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  return writeSize == static_cast<ssize_t>(size);
}

} // namespace PtoTestCommon

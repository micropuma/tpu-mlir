//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"

// 存储inference阶段的参数
// inputs和outputs是指向float的指针，主要指inference阶段输入数据位置和输出存储位置。
// handle是一个void*指针，主要用于存储模型句柄（optional）
namespace tpu_mlir {
struct InferenceParameter {
  std::vector<float *> inputs;
  std::vector<float *> outputs;
  void *handle = nullptr;
};

} // namespace tpu_mlir

/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/InferenceInterface.h.inc"

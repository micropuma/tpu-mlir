//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void LRNLowering::LoweringINT8(PatternRewriter &rewriter, top::LRNOp op,
                               bool asymmetric) const {

  LoweringF32(rewriter, op);
}

void LRNLowering::LoweringF32(PatternRewriter &rewriter, top::LRNOp op) const {
  lowering_common_f32<tpu::LRNOp>(rewriter, op, 3);
}

} // namespace bm1684
} // namespace tpu_mlir

// Copyright 2010-2022 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OR_TOOLS_MATH_OPT_VALIDATORS_INFEASIBLE_SUBSYSTEM_VALIDATOR_H_
#define OR_TOOLS_MATH_OPT_VALIDATORS_INFEASIBLE_SUBSYSTEM_VALIDATOR_H_

#include "absl/status/status.h"
#include "ortools/math_opt/core/model_summary.h"
#include "ortools/math_opt/infeasible_subsystem.pb.h"

namespace operations_research::math_opt {

absl::Status ValidateModelSubset(const ModelSubsetProto& model_subset,
                                 const ModelSummary& summary);

absl::Status ValidateComputeInfeasibleSubsystemResult(
    const ComputeInfeasibleSubsystemResultProto& result,
    const ModelSummary& summary);
// Validates the internal consistency of the fields.
absl::Status ValidateComputeInfeasibleSubsystemResultNoModel(
    const ComputeInfeasibleSubsystemResultProto& result);

}  // namespace operations_research::math_opt

#endif  // OR_TOOLS_MATH_OPT_VALIDATORS_INFEASIBLE_SUBSYSTEM_VALIDATOR_H_

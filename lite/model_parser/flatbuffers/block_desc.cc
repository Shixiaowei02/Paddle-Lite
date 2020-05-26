// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
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

#include "lite/model_parser/flatbuffers/block_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

template <>
const proto::VarDesc* BlockDesc::GetVar<const proto::VarDesc>(
    int32_t idx) {
  CHECK_LT(idx, VarsSize()) << "idx >= vars.size()";
  return desc_->vars()->Get(idx);
}

template <>
const proto::OpDesc* BlockDesc::GetOp<const proto::OpDesc>(
    int32_t idx) {
  CHECK_LT(idx, OpsSize()) << "idx >= ops.size()";
  return desc_->ops()->Get(idx);
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

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
#include "lite/model_parser/flatbuffers/op_desc.h"
#include "lite/model_parser/flatbuffers/var_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

template <>
OpDesc* BlockDesc::AddOp() {
  auto* op = new OpDesc(this);
  std::unique_ptr<proto::OpDescT> op_p(static_cast<proto::OpDescT*>(op));
  ops.push_back(std::move(op_p));
  return op;
}

template <>
VarDesc* BlockDesc::AddVar() {
  auto* var = new VarDesc(this);
  std::unique_ptr<proto::VarDescT> var_p(static_cast<proto::VarDescT*>(var));
  vars.push_back(std::move(var_p));
  return var;
}

}
}
}

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

#pragma once

#include "lite/model_parser/base/block_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class BlockDesc : public BlockDescAPI {
 public:
  explicit BlockDesc(proto::BlockDesc* desc) : desc_(desc) { CHECK(desc_); }

  int32_t Idx() const override { return desc_->idx(); }

  int32_t ParentIdx() const override { return desc_->parent_idx(); }

  size_t VarsSize() const override { return desc_->vars()->size(); }

  template <typename T>
  T const* GetVar(int32_t idx) const {
    return GetVar<T>(idx);
  }

  size_t OpsSize() const override {
    CHECK(desc_);
    CHECK(desc_->ops());
    return desc_->ops()->size();
  }

  template <typename T>
  T const* GetOp(int32_t idx) const {
    return GetOp<T>(idx);
  }

  int32_t ForwardBlockIdx() const override {
    return desc_->forward_block_idx();
  }

  BlockDesc() = delete;

 private:
  proto::BlockDesc const* desc_;  // not_own
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

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

#include <memory>
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

class BlockDesc : public BlockDescAPI {
 public:
  BlockDesc() = delete;

  explicit BlockDesc(framework::proto::BlockDesc* desc) {
    CHECK(desc_);
    desc_.reset(desc);
  }

  int32_t Idx() const override { return desc_->idx(); }

  void SetIdx(int32_t idx) override { LOG(FATAL) << "Feature not yet supported."; }

  int32_t ParentIdx() const override { return desc_->parent_idx(); }

  void SetParentIdx(int32_t idx) override { LOG(FATAL) << "Feature not yet supported."; }

  size_t VarsSize() const override { return desc_->vars()->size(); }

  void ClearVars() override { LOG(FATAL) << "Feature not yet supported."; }

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T* AddVar() {
    LOG(FATAL) << "Feature not yet supported.";
  }

  size_t OpsSize() const override { return desc_->ops()->size(); }

  void ClearOps() override { LOG(FATAL) << "Feature not yet supported."; }

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T* AddOp() {
    LOG(FATAL) << "Feature not yet supported.";
  }

  int32_t ForwardBlockIdx() const override {
    return desc_->forward_block_idx();
  }

  void SetForwardBlockIdx(int32_t idx) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

 private:
  std::unique_ptr<proto::BlockDesc> desc_;  // not_own
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

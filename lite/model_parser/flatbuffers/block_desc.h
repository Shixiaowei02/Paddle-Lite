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
#include "lite/model_parser/desc_apis.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

class ProgramDesc;

class BlockDesc : public BlockDescAPI, private proto::BlockDescT {
 public:
  BlockDesc() = delete;

  // will be deleted.
  explicit BlockDesc(paddle::lite::fbs::BlockDesc* desc) {
  }

  explicit BlockDesc(fbs::ProgramDesc* desc) {
    program_desc_  = desc;
  }

  int32_t Idx() const override {
    return idx;
  }

  void SetIdx(int32_t idx_in) override {
    idx = idx_in;
  }

  int32_t ParentIdx() const override {
    return parent_idx;
  }

  void SetParentIdx(int32_t idx_in) override {
    parent_idx = idx_in;
  }

  size_t VarsSize() const override {
    return vars.size();
  }

  void ClearVars() override {
    vars.clear();
  }

  template <typename T>
  T* GetVar(int32_t idx) {
    LOG(FATAL);
    return nullptr;
  }

  template <typename T>
  T* AddVar();

  size_t OpsSize() const override {
    return ops.size();
  }

  void ClearOps() override {
    ops.clear();
  }

  template <typename T>
  T* GetOp(int32_t idx) {
    LOG(FATAL);
    return nullptr;
  }

  template <typename T>
  T* AddOp();

  int32_t ForwardBlockIdx() const override {
    return forward_block_idx;
  }

  void SetForwardBlockIdx(int32_t idx_in) override {
    forward_block_idx = idx_in;
  }

  friend class ProgramDesc;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

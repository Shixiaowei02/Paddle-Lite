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

#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = delete;

  explicit ProgramDesc(internal::ProgramDesc *raw_desc) {
    CHECK(raw_desc);
    raw_.reset(raw_desc);
  }

  size_t BlocksSize() const override { return desc_->blocks()->size(); }

  void ClearBlocks() override { LOG(FATAL) << "Feature not yet supported."; }

  template <typename T>
  T *GetBlock(int32_t idx);

  template <typename T>
  T *AddBlock() {
    LOG(FATAL) << "Feature not yet supported.";
  }

  bool HasVersion() const override { return desc_->version() == nullptr; }

  int64_t Version() const override { return desc_->version()->version(); }

  void SetVersion(int64_t version) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

 private:
  std::vector<fbs::BlockDesc> blocks_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

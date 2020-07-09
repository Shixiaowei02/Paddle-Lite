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

#include <memory>
#include <utility>
#include "lite/model_parser/base/program_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;
  explicit ProgramDesc(std::unique_ptr<const char[]> buf)
      : buf_(std::move(buf)) {
    CHECK(buf_.get() != nullptr);
    desc_ = proto::GetProgramDesc(buf_.get());
  }

  size_t BlocksSize() const override { return desc_->blocks()->size(); }

  template <typename T>
  T const* GetBlock(int32_t idx) const {
    return GetBlock<T>(idx);
  }

  bool HasVersion() const override { return desc_->version() != nullptr; }

  int64_t Version() const override {
    CHECK(HasVersion());
    return desc_->version()->version();
  }

 private:
  proto::ProgramDesc const* desc_;
  std::unique_ptr<const char[]> buf_;

 private:
  ProgramDesc& operator=(const ProgramDesc&) = delete;
  ProgramDesc(const ProgramDesc&) = delete;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

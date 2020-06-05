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
#include "lite/model_parser/flatbuffers/block_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

class ProgramDesc : public ProgramDescAPI, public proto::ProgramDescT {
 public:
  ProgramDesc() = default;

  explicit ProgramDesc(proto::ProgramDesc* desc) {
    desc_ = desc;
    desc->UnPackTo(dynamic_cast<ProgramDescT*>(this));
  }

  size_t BlocksSize() const override {
    return blocks.size();
  }

  void ClearBlocks() override {
    blocks.clear();
  }

  template <typename T>
  T *GetBlock(int32_t idx) {
    LOG(FATAL);
    return T();
  }

  template <typename T>
  T *AddBlock();

  bool HasVersion() const override {
    return version.get();
  }

  int64_t Version() const override {
    return version->version;
  }

  void SetVersion(int64_t version_in) override {
    version->version = version_in;
  }
  private:
  proto::ProgramDesc* desc_;
  flatbuffers::FlatBufferBuilder fbb_;
};

template <>
BlockDesc* ProgramDesc::AddBlock() {
  auto* block = new BlockDesc(this);
  std::unique_ptr<proto::BlockDescT> block_p(static_cast<proto::BlockDescT*>(block));
  blocks.push_back(std::move(block_p));
  return block;
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

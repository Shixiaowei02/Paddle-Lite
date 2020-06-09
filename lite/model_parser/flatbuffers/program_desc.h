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

class ProgramDesc : public ProgramDescAPI, public proto::ProgramDescT {
 public:
  ProgramDesc() = default;

  explicit ProgramDesc(flatbuffers::DetachedBuffer&& buf) {
    buf_ = std::move(buf);
    auto* desc = proto::GetProgramDesc(buf.data());
    desc->UnPackTo(dynamic_cast<ProgramDescT*>(this));
  }

  const flatbuffers::DetachedBuffer& SyncBuffer() {
    fbb_.Reset();
    flatbuffers::Offset<proto::ProgramDesc> desc = proto::ProgramDesc::Pack(fbb_, this);
    fbb_.Finish(desc);
    buf_ = fbb_.Release();
    return buf_;
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
    return nullptr;
  }

  template <typename T>
  T *AddBlock();

  bool HasVersion() const override {
    return version.get();
  }

  int64_t Version() const override {
    if(!HasVersion()) {
      return version->version;
    } else {
      return -1;
    }
  }

  void SetVersion(int64_t version_in) override {
    if(!HasVersion()) {
      version.reset(new fbs::proto::VersionT());
    }
    version->version = version_in;
  }
private:
  flatbuffers::DetachedBuffer buf_;
  flatbuffers::FlatBufferBuilder fbb_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

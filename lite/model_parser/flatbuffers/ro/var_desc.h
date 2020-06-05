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

class VarDesc : public VarDescAPI {
 public:
  VarDesc() = default;

  explicit VarDesc(proto::VarDesc* desc) : desc_(desc) {}

  std::string Name() const override {
    return desc_->name()->str();
  }

  void SetName(std::string name) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  VarDescAPI::Type GetType() const override {
    return static_cast<VarDescAPI::Type>(desc_->type()->type());
  }

  void SetType(VarDescAPI::Type type) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  bool Persistable() const override {
    return desc_->persistable();
  }

  void SetPersistable(bool persistable) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  void SetShape(const std::vector<int64_t> &dims) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  void SetDataType(Type data_type) { LOG(FATAL) << "Feature not yet supported."; }

  std::vector<int64_t> GetShape() const override {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    const auto& dims = desc_->type()->lod_tensor()->tensor()->dims();
    std::vector<int64_t> dims_vec;
    dims_vec.reserve(dims->size());
    for (const auto& dim: *dims) {
      dims_vec.push_back(dim);
    }
    return dims_vec;
  }

 private:
  proto::VarDesc* desc_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
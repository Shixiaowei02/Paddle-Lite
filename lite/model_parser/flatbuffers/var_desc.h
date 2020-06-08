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

class BlockDesc;

class VarDesc : public VarDescAPI, private proto::VarDescT {
 public:
  // will be deleted.
  explicit VarDesc(paddle::lite::fbs::VarDesc* desc) {
  }

  std::string Name() const override {
    return name;
  }

  void SetName(std::string name_in) override {
    name = name_in;
  }

  VarDescAPI::Type GetType() const override {
    return static_cast<VarDescAPI::Type>(type->type);
  }

  void SetType(VarDescAPI::Type type_in) override {
    if (!type) {
      type = std::unique_ptr<proto::VarTypeT>(new proto::VarTypeT());
      if (type_in == VarDescAPI::Type::LOD_TENSOR) {
        type->lod_tensor = std::unique_ptr<proto::VarType_::LoDTensorDescT>(new proto::VarType_::LoDTensorDescT());
        type->lod_tensor->tensor = std::unique_ptr<proto::VarType_::TensorDescT>(new proto::VarType_::TensorDescT()); 
      }
    }
    type->type = static_cast<proto::VarType_::Type>(type_in);
  }

  bool Persistable() const override {
    return persistable;
  }

  void SetPersistable(bool persistable_in) override {
    persistable = persistable_in;
  }

  void SetShape(const std::vector<int64_t> &dims_in) override {
    type->lod_tensor->tensor->dims = dims_in;
  }

  void SetDataType(Type data_type_in) {
    type->lod_tensor->tensor->data_type = static_cast<proto::VarType_::Type>(data_type_in);
  }

  std::vector<int64_t> GetShape() const override {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    return type->lod_tensor->tensor->dims;
  }

private:
  VarDesc() = default;
  friend class BlockDesc;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

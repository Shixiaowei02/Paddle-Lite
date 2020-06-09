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

class OpDesc : public OpDescAPI, public proto::OpDescT {
public:
  // will be deleted
  explicit OpDesc(paddle::lite::fbs::OpDesc* desc) {
  }
  std::string Type() const override {
    return type;
  }

  void SetType(const std::string &type_in) override {
    type = type_in;
  }


  std::vector<std::string> Input(const std::string &param) const override {
    LOG(FATAL);
    return std::vector<std::string>();
  }

  std::vector<std::string> InputArgumentNames() const override {
    LOG(FATAL);
    return std::vector<std::string>();
  }

  void SetInput(const std::string &param,
                const std::vector<std::string> &args) override {
    auto* var = new proto::OpDesc_::VarT;
    var->parameter = param;
    var->arguments = args;
    inputs.emplace_back(var);
  }

  std::vector<std::string> Output(const std::string &param) const override {
    LOG(FATAL);
    return std::vector<std::string>();
  }

  std::vector<std::string> OutputArgumentNames() const override {
    LOG(FATAL);
    return std::vector<std::string>();
  }

  void SetOutput(const std::string &param,
                 const std::vector<std::string> &args) override {
    auto* var = new proto::OpDesc_::VarT;
    var->parameter = param;
    var->arguments = args;
    outputs.emplace_back(var);
  }

  bool HasAttr(const std::string &name) const override {
    LOG(FATAL);
    return false;
  }

  size_t AttrsSize() const {
    return attrs.size();
  }

  std::string AttrName(size_t idx) const {
    LOG(FATAL);
    return std::string();
  }

  OpDescAPI::AttrType GetAttrType(const std::string &name) const override {
    LOG(FATAL);
    return OpDescAPI::AttrType();
  }

  OpDescAPI::AttrType GetAttrType(size_t idx) const {
    LOG(FATAL);
    return OpDescAPI::AttrType();
  }

  std::vector<std::string> AttrNames() const override {
    LOG(FATAL);
    return std::vector<std::string>();
  }

  template <typename T>
  T GetAttr(const std::string &name) const {
    LOG(FATAL);
    return T();
  }

  template <typename T>
  void AddAttr(const std::string &name, const T &v);

  template <typename T>
  void SetAttr(const std::string &name, const T &v) {
     AddAttr(name, v);
  }
  
  template <typename T>
  T GetAttr(size_t idx) const {
    LOG(FATAL);
    return T();
  }
//private:
  OpDesc() = default;
  friend class BlockDesc;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

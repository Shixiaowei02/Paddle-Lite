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

#include <unique_ptr>
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

class OpDesc : public OpDescAPI {
 public:
  OpDesc() = delete;

  explicit OpDesc(framework::proto::OpDesc *desc) : desc_(desc) {
    CHECK(desc_);
    desc_.reset(desc);
  }

  std::string Type() const override { return desc_->type()->str(); }

  void SetType(const std::string &type) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string &param) const override {
    const auto& input = desc_->inputs()->LookupByKey(param);
    std::vector<std::string> input_vec;
    input_vec.reserve(input.size());
    for (const auto& in: input) {
      input_vec.push_back(in->str());
    }
    return input_vec;
  }

  std::vector<std::string> InputArgumentNames() const override {
    const auto& inputs = desc_->inputs();
    std::vector<std::string> input_names_vec;
    input_names_vec.reserve(inputs.size());
    for (const auto& in: inputs) {
      input_names_vec.push_back(in->parameter()->str());
    }
    return input_names_vec;
  }

  void SetInput(const std::string &param,
                const std::vector<std::string> &args) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  std::vector<std::string> Output(const std::string &param) const override {
    const auto& output = desc_->outputs()->LookupByKey(param);
    std::vector<std::string> output_vec;
    output_vec.reserve(output.size());
    for (const auto& out: output) {
      output_vec.push_back(out->str());
    }
    return output_vec;
  }

  std::vector<std::string> OutputArgumentNames() const override {
    const auto& outputs = desc_->outputs();
    std::vector<std::string> output_names_vec;
    output_names_vec.reserve(outputs.size());
    for (const auto& out: outputs) {
      output_names_vec.push_back(out->parameter()->str());
    }
    return output_names_vec;
  }

  void SetOutput(const std::string &param,
                 const std::vector<std::string> &args) override {
    LOG(FATAL) << "Feature not yet supported.";
  }

  bool HasAttr(const std::string &name) const override {
    return desc_->attrs()->LookupByKey(name) == nullptr;
  }

  AttrType GetAttrType(const std::string &name) const override {
    const auto& attr = desc_->attrs()->LookupByKey(name);
    CHECK(attr);
    return attr->type();
  }

  std::vector<std::string> AttrNames() const override {
    const auto& attrs = desc_->attrs();
    std::vector<std::string> attr_names_vec;
    attr_names_vec.reserve(attrs.size());
    for (const auto& attr: attrs) {
      attr_names_vec.push_back(attr->name()->str());
    }
    return attr_names_vec;
  }

  template <typename T>
  void SetAttr(const std::string &name, const T &v) {
     LOG(FATAL) << "Feature not yet supported.";
  }

  template <typename T>
  T GetAttr(const std::string &name) const;

 private:
  std::unique_ptr<proto::OpDesc> desc_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle

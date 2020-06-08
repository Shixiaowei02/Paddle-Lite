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

#include "lite/model_parser/flatbuffers/op_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

#define ADD_ATTR_IMPL(T, ty__, fbs_a_)                            \
  template <> \
  void OpDesc::AddAttr<T>(const std::string &name, const T &v) { \
    auto* attr = new proto::OpDesc_::AttrT(); \
    attr->type = proto::AttrType_##ty__; \
    attr->fbs_a_ = v; \
    std::unique_ptr<proto::OpDesc_::AttrT> attr_p(attr); \
    attrs.emplace_back(std::move(attr_p)); \
  }
  
ADD_ATTR_IMPL(int, INT, i);
ADD_ATTR_IMPL(float, FLOAT, f);
ADD_ATTR_IMPL(bool, BOOLEAN, b);
ADD_ATTR_IMPL(int64_t, LONG, l);
ADD_ATTR_IMPL(std::string, STRING, s);
ADD_ATTR_IMPL(std::vector<int>, INTS, ints);
ADD_ATTR_IMPL(std::vector<float>, FLOATS, floats);
ADD_ATTR_IMPL(std::vector<bool>, BOOLEANS, bools);
ADD_ATTR_IMPL(std::vector<int64_t>, LONGS, longs);
ADD_ATTR_IMPL(std::vector<std::string>, STRINGS, strings);

}
}
}

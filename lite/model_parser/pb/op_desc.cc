// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {
namespace pb {

google::protobuf::internal::RepeatedPtrIterator<framework::proto::OpDesc_Attr>
FindAttr(framework::proto::OpDesc *desc, const std::string &name) {
  auto &xs = *desc->mutable_attrs();
  auto it = std::find_if(
      xs.begin(), xs.end(), [&](const framework::proto::OpDesc_Attr &x) {
        return x.name() == name;
      });
  if (it == xs.end()) {
    auto *attr = xs.Add();
    attr->set_name(name);
    it = std::find_if(
        xs.begin(), xs.end(), [&](const framework::proto::OpDesc_Attr &x) {
          return x.name() == name;
        });
  }
  return it;
}

#define SET_IMPL_ONE(T, ty__, pb_f__)                            \
  template <>                                                    \
  void OpDesc::SetAttr<T>(const std::string &name, const T &v) { \
    auto it = FindAttr(desc_, name);                             \
    it->set_type(framework::proto::ty__);                        \
    it->set_##pb_f__(v);                                         \
  }
SET_IMPL_ONE(int, INT, i);
SET_IMPL_ONE(float, FLOAT, f);
SET_IMPL_ONE(bool, BOOLEAN, b);
SET_IMPL_ONE(int64_t, LONG, l);

template <>
void OpDesc::SetAttr<std::vector<int>>(const std::string &name,
                                       const std::vector<int> &v) {
  auto it = FindAttr(desc_, name);
  it->set_type(framework::proto::INTS);
  it->clear_ints();
  for (auto &i : v) {
    it->add_ints(i);
  }
}

template <>
void OpDesc::SetAttr<std::string>(const std::string &name,
                                  const std::string &v) {
  auto it = FindAttr(desc_, name);
  it->set_type(framework::proto::STRING);
  it->set_s(v.c_str());
}

template <>
void OpDesc::SetAttr<std::vector<float>>(const std::string &name,
                                         const std::vector<float> &v) {
  auto it = FindAttr(desc_, name);
  it->set_type(framework::proto::FLOATS);
  it->clear_floats();
  for (auto &i : v) {
    it->add_floats(i);
  }
}

template <>
void OpDesc::SetAttr<std::vector<std::string>>(
    const std::string &name, const std::vector<std::string> &v) {
  auto it = FindAttr(desc_, name);
  it->set_type(framework::proto::STRINGS);
  it->clear_strings();
  for (auto &i : v) {
    it->add_strings(i);
  }
}

template <>
void OpDesc::SetAttr<std::vector<int64_t>>(const std::string &name,
                                           const std::vector<int64_t> &v) {
  auto it = FindAttr(desc_, name);
  it->set_type(framework::proto::LONGS);
  it->clear_longs();
  for (auto &i : v) {
    it->add_longs(i);
  }
}
google::protobuf::internal::RepeatedPtrIterator<
    const framework::proto::OpDesc_Attr>
GetFindAttr(const framework::proto::OpDesc &desc, const std::string &name) {
  auto &xs = desc.attrs();
  auto it = std::find_if(
      xs.begin(), xs.end(), [&](const framework::proto::OpDesc_Attr &x) {
        return x.name() == name;
      });
  return it;
}

#define GET_ATTR_IMPL(T, pb_f__)                                  \
  template <>                                                     \
  typename OpAttrTypeTrait<OpAttrType::T, Standard>::RT           \
  OpDesc::GetAttr<OpAttrType::T>(const std::string &name) const { \
    auto it = GetFindAttr(*desc_, name);                          \
    return it->pb_f__();                                          \
  }

#define GET_ATTRS_IMPL(T, pb_f__)                                 \
  template <>                                                     \
  typename OpAttrTypeTrait<OpAttrType::T, Standard>::RT           \
  OpDesc::GetAttr<OpAttrType::T>(const std::string &name) const { \
    auto it = GetFindAttr(*desc_, name);                          \
    OpAttrTypeTrait<OpAttrType::T, Standard>::RT res;             \
    for (const auto &v : it->pb_f__()) {                          \
      res.push_back(v);                                           \
    }                                                             \
    return res;                                                   \
  }
GET_ATTR_IMPL(INT, i);
GET_ATTR_IMPL(BLOCK, block_idx);
GET_ATTR_IMPL(FLOAT, f);
GET_ATTR_IMPL(BOOLEAN, b);
GET_ATTR_IMPL(LONG, l);
GET_ATTR_IMPL(STRING, s);

GET_ATTRS_IMPL(INTS, ints);
GET_ATTRS_IMPL(FLOATS, floats);
GET_ATTRS_IMPL(STRINGS, strings);
GET_ATTRS_IMPL(LONGS, longs);

}  // namespace pb
}  // namespace lite
}  // namespace paddle

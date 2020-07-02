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

#include <string>
#include <vector>

namespace paddle {
namespace lite {

enum class OpAttrType {
  INT = 0,
  FLOAT = 1,
  STRING = 2,
  INTS = 3,
  FLOATS = 4,
  STRINGS = 5,
  BOOLEAN = 6,
  BOOLEANS = 7,
  BLOCK = 8,
  LONG = 9,
  BLOCKS = 10,
  LONGS = 11,
  UNK,
};

struct Standard {};
struct Flatbuffers {};

template <typename T, typename U>
class VectorView;

template <OpAttrType Type, typename U>
struct OpAttrTypeTrait;

template <typename T>
struct OpDataTypeTrait;

#define OP_TYPE_TRAIT_IMPL(T, dtype__)              \
  template <typename U>                             \
  struct OpAttrTypeTrait<OpAttrType::T, U> {        \
    typedef dtype__ DT;                             \
    typedef dtype__ RT;                             \
    static constexpr const char* ATN = #T;          \
  };                                                \
  template <>                                       \
  struct OpDataTypeTrait<dtype__> {                 \
    static constexpr OpAttrType AT = OpAttrType::T; \
    static constexpr const char* ATN = #T;          \
  };

#define OP_VEC_TYPE_TRAIT_IMPL(T, dtype__)             \
  template <>                                          \
  struct OpAttrTypeTrait<OpAttrType::T, Flatbuffers> { \
    typedef dtype__ ET;                                \
    typedef std::vector<dtype__> DT;                   \
    typedef VectorView<dtype__, Flatbuffers> RT;       \
    static constexpr const char* ATN = #T;             \
  };                                                   \
  template <>                                          \
  struct OpAttrTypeTrait<OpAttrType::T, Standard> {    \
    typedef dtype__ ET;                                \
    typedef std::vector<dtype__> DT;                   \
    typedef DT RT;                                     \
    static constexpr const char* ATN = #T;             \
  };                                                   \
  template <>                                          \
  struct OpDataTypeTrait<std::vector<dtype__>> {       \
    static constexpr OpAttrType AT = OpAttrType::T;    \
    static constexpr const char* ATN = #T;             \
  };

OP_TYPE_TRAIT_IMPL(INT, int32_t);
OP_TYPE_TRAIT_IMPL(FLOAT, float);
OP_TYPE_TRAIT_IMPL(STRING, std::string);
OP_TYPE_TRAIT_IMPL(BOOLEAN, bool);
OP_TYPE_TRAIT_IMPL(LONG, int64_t);
OP_TYPE_TRAIT_IMPL(BLOCK, int16_t);

OP_VEC_TYPE_TRAIT_IMPL(INTS, int32_t);
OP_VEC_TYPE_TRAIT_IMPL(FLOATS, float);
OP_VEC_TYPE_TRAIT_IMPL(STRINGS, std::string);
OP_VEC_TYPE_TRAIT_IMPL(LONGS, int64_t);
#undef TYPE_TRAIT_IMPL

}  // namespace lite
}  // namespace paddle

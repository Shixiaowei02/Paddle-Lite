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

#include "lite/model_parser/compatible_pb.h"
#include <string>
#include <vector>
#include "lite/model_parser/flatbuffers/block_desc.h"
#include "lite/model_parser/flatbuffers/op_desc.h"
#include "lite/model_parser/flatbuffers/program_desc.h"
#include "lite/model_parser/flatbuffers/var_desc.h"
#include "lite/model_parser/naive_buffer/block_desc.h"
#include "lite/model_parser/naive_buffer/op_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/model_parser/pb/block_desc.h"
#include "lite/model_parser/pb/op_desc.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"
#endif

namespace paddle {
namespace lite {

/// For VarDesc transfrom
#define TRANS_VAR_ANY_WITH_CPP_IMPL(T)                             \
  template <>                                                      \
  void TransformVarDescCppToAny<T>(const cpp::VarDesc &cpp_desc,   \
                                   T *any_desc) {                  \
    any_desc->SetName(cpp_desc.Name());                            \
    any_desc->SetType(cpp_desc.GetType());                         \
    any_desc->SetPersistable(cpp_desc.Persistable());              \
    if (cpp_desc.Name() != "feed" && cpp_desc.Name() != "fetch") { \
      any_desc->SetShape(cpp_desc.GetShape());                     \
      any_desc->SetDataType(cpp_desc.GetDataType());               \
    }                                                              \
  }

#ifndef LITE_ON_TINY_PUBLISH
template <>
void TransformVarDescAnyToCpp<pb::VarDesc>(const pb::VarDesc &any_desc,
                                           cpp::VarDesc *cpp_desc) {
  cpp_desc->SetName(any_desc.Name());
  cpp_desc->SetType(any_desc.GetType());
  cpp_desc->SetPersistable(any_desc.Persistable());
  if (any_desc.Name() != "feed" && any_desc.Name() != "fetch") {
    cpp_desc->SetDataType(any_desc.GetDataType());
    cpp_desc->SetShape(any_desc.GetShape());
  }
}
#endif

template <>
void TransformVarDescAnyToCpp<naive_buffer::VarDesc>(
    const naive_buffer::VarDesc &any_desc, cpp::VarDesc *cpp_desc) {
  cpp_desc->SetName(any_desc.Name());
  cpp_desc->SetType(any_desc.GetType());
  cpp_desc->SetPersistable(any_desc.Persistable());
  // todo : SetDataType function is commented out temporarily
  // because of Compatibility issues. The Compatibility issue
  // should be fixed later and the code below should be applied
  // later. @DannyIsFunny
  /*  if (any_desc.Name() != "feed" && any_desc.Name() != "fetch") {
      cpp_desc->SetDataType(any_desc.GetDataType());
      cpp_desc->SetShape(any_desc.GetShape());
    }*/
}

template <>
void TransformVarDescAnyToCpp<fbs::VarDesc>(
    const fbs::VarDesc &any_desc, cpp::VarDesc *cpp_desc) {
  cpp_desc->SetName(any_desc.Name());
  cpp_desc->SetType(any_desc.GetType());
  cpp_desc->SetPersistable(any_desc.Persistable());
}

/// For OpDesc transform
template <typename OpDescType>
void OpInputsAnyToCpp(const OpDescType &any_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : any_desc.InputArgumentNames()) {
    cpp_desc->SetInput(param, any_desc.Input(param));
  }
}

template <typename OpDescType>
void OpInputsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  for (const std::string &param : cpp_desc.InputArgumentNames()) {
    any_desc->SetInput(param, cpp_desc.Input(param));
  }
}

template <typename OpDescType>
void OpOutputsAnyToCpp(const OpDescType &any_desc, cpp::OpDesc *cpp_desc) {
  for (const std::string &param : any_desc.OutputArgumentNames()) {
    cpp_desc->SetOutput(param, any_desc.Output(param));
  }
}

template <typename OpDescType>
void OpOutputsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  for (const std::string &param : cpp_desc.OutputArgumentNames()) {
    any_desc->SetOutput(param, cpp_desc.Output(param));
  }
}

template <typename fbs::OpDesc>
void OpAttrsAnyToCpp(const fbs::OpDesc &any_desc, cpp::OpDesc *cpp_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](size_t idx, const AttrType& type) {
    const std::string& name = any_desc.AttrName(idx);
    switch (type) {
      case AttrType::INT:
        cpp_desc->SetAttr<int32_t>(name,
                                   any_desc.template GetAttr<int32_t>(idx));
        break;
      case AttrType::FLOAT:
        cpp_desc->SetAttr<float>(name, any_desc.template GetAttr<float>(idx));
        break;
      case AttrType::STRING:
        cpp_desc->SetAttr<std::string>(
            name, any_desc.template GetAttr<std::string>(idx));
        break;
      case AttrType::LONG:
        cpp_desc->SetAttr<int64_t>(idx,
                                   any_desc.template GetAttr<int64_t>(idx));
        break;
      case AttrType::INTS:
        cpp_desc->SetAttr<std::vector<int>>(
            name, any_desc.template GetAttr<std::vector<int>>(idx));
        break;
      case AttrType::FLOATS:
        cpp_desc->SetAttr<std::vector<float>>(
            name, any_desc.template GetAttr<std::vector<float>>(idx));
        break;
      case AttrType::BOOLEAN:
        cpp_desc->SetAttr<bool>(name, any_desc.template GetAttr<bool>(idx));
        break;
      case AttrType::STRINGS:
        cpp_desc->SetAttr<std::vector<std::string>>(
            name, any_desc.template GetAttr<std::vector<std::string>>(idx));
        break;
      case AttrType::LONGS:
        cpp_desc->SetAttr<std::vector<int64_t>>(
            name, any_desc.template GetAttr<std::vector<int64_t>>(idx));
        break;
      case AttrType::BLOCK: {
        auto i = any_desc.template GetAttr<int16_t>(idx);
        cpp_desc->SetAttr<int32_t>(name, i);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported attr type found " << static_cast<int>(type);
    }
  };

  for (size_t i = 0; i < any_desc.AttrsSize(); ++i) {
    const auto& type = any_desc.GetAttrType(i);
    set_attr(i, type);
  }
}



template <typename OpDescType>
void OpAttrsCppToAny(const cpp::OpDesc &cpp_desc, OpDescType *any_desc) {
  using AttrType = OpDescAPI::AttrType;
  auto set_attr = [&](const std::string &name, AttrType type) {
    switch (type) {
#define IMPL_ONE(type__, T)                                         \
  case AttrType::type__:                                            \
    any_desc->template SetAttr<T>(name, cpp_desc.GetAttr<T>(name)); \
    break;
      IMPL_ONE(INT, int32_t);
      IMPL_ONE(FLOAT, float);
      IMPL_ONE(STRING, std::string);
      IMPL_ONE(STRINGS, std::vector<std::string>);
      IMPL_ONE(FLOATS, std::vector<float>);
      IMPL_ONE(INTS, std::vector<int>);
      IMPL_ONE(BOOLEAN, bool);
      IMPL_ONE(LONG, int64_t);
      IMPL_ONE(LONGS, std::vector<int64_t>);
      default:
        LOG(FATAL) << "Unsupported attr type found: " << static_cast<int>(type);
    }
  };
#undef IMPL_ONE
  for (const auto &attr_name : cpp_desc.AttrNames()) {
    auto type = cpp_desc.GetAttrType(attr_name);
    set_attr(attr_name, type);
  }
}

#define TRANS_OP_ANY_WITH_CPP_IMPL(T)                                         \
  template <>                                                                 \
  void TransformOpDescAnyToCpp<T>(const T &any_desc, cpp::OpDesc *cpp_desc) { \
    cpp_desc->SetType(any_desc.Type());                                       \
    OpInputsAnyToCpp<T>(any_desc, cpp_desc);                                  \
    OpOutputsAnyToCpp<T>(any_desc, cpp_desc);                                 \
    OpAttrsAnyToCpp<T>(any_desc, cpp_desc);                                   \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  void TransformOpDescCppToAny<T>(const cpp::OpDesc &cpp_desc, T *any_desc) { \
    any_desc->SetType(cpp_desc.Type());                                       \
    OpInputsCppToAny<T>(cpp_desc, any_desc);                                  \
    OpOutputsCppToAny<T>(cpp_desc, any_desc);                                 \
    OpAttrsCppToAny<T>(cpp_desc, any_desc);                                   \
  }

/// For BlockDesc transform
#define TRANS_BLOCK_ANY_WITH_CPP_IMPL(T, NT, PNT)                           \
  template <>                                                               \
  void TransformBlockDescAnyToCpp<NT::T>(const NT::T &any_desc,             \
                                         cpp::BlockDesc *cpp_desc) {        \
    NT::T desc = any_desc;                                                  \
    cpp_desc->SetIdx(desc.Idx());                                           \
    cpp_desc->SetParentIdx(desc.ParentIdx());                               \
    cpp_desc->SetForwardBlockIdx(desc.ForwardBlockIdx());                   \
                                                                            \
    cpp_desc->ClearOps();                                                   \
    for (size_t i = 0; i < desc.OpsSize(); ++i) {                           \
      auto any_op_desc = NT::OpDesc(desc.GetOp<PNT::proto::OpDesc>(i));     \
      auto *cpp_op_desc = cpp_desc->AddOp<cpp::OpDesc>();                   \
      TransformOpDescAnyToCpp(any_op_desc, cpp_op_desc);                    \
    }                                                                       \
                                                                            \
    cpp_desc->ClearVars();                                                  \
    for (size_t i = 0; i < desc.VarsSize(); ++i) {                          \
      auto any_var_desc = NT::VarDesc(desc.GetVar<PNT::proto::VarDesc>(i)); \
      auto *cpp_var_desc = cpp_desc->AddVar<cpp::VarDesc>();                \
      TransformVarDescAnyToCpp(any_var_desc, cpp_var_desc);                 \
    }                                                                       \
  }                                                                         \
                                                                            \
  template <>                                                               \
  void TransformBlockDescCppToAny<NT::T>(const cpp::T &cpp_desc,            \
                                         NT::T *any_desc) {                 \
    auto desc = cpp_desc;                                                   \
    any_desc->SetIdx(desc.Idx());                                           \
    any_desc->SetParentIdx(desc.ParentIdx());                               \
    any_desc->SetForwardBlockIdx(desc.ForwardBlockIdx());                   \
                                                                            \
    any_desc->ClearOps();                                                   \
    for (size_t i = 0; i < desc.OpsSize(); ++i) {                           \
      auto *cpp_op_desc = desc.GetOp<cpp::OpDesc>(i);                       \
      auto any_op_desc = NT::OpDesc(any_desc->AddOp<PNT::proto::OpDesc>()); \
      TransformOpDescCppToAny(*cpp_op_desc, &any_op_desc);                  \
    }                                                                       \
                                                                            \
    any_desc->ClearVars();                                                  \
    for (size_t i = 0; i < desc.VarsSize(); ++i) {                          \
      auto *cpp_var_desc = desc.GetVar<cpp::VarDesc>(i);                    \
      auto any_var_desc =                                                   \
          NT::VarDesc(any_desc->AddVar<PNT::proto::VarDesc>());             \
      TransformVarDescCppToAny(*cpp_var_desc, &any_var_desc);               \
    }                                                                       \
  }

/// For ProgramDesc transform
#define TRANS_PROGRAM_ANY_WITH_CPP_IMPL(T, NT, PNT)                      \
  template <>                                                            \
  void TransformProgramDescAnyToCpp<NT::T>(const NT::T &any_desc,        \
                                           cpp::ProgramDesc *cpp_desc) { \
    NT::T desc = any_desc;                                               \
    if (desc.HasVersion()) {                                             \
      cpp_desc->SetVersion(desc.Version());                              \
    }                                                                    \
                                                                         \
    cpp_desc->ClearBlocks();                                             \
    for (size_t i = 0; i < desc.BlocksSize(); ++i) {                     \
      auto any_block_desc =                                              \
          NT::BlockDesc(desc.GetBlock<PNT::proto::BlockDesc>(i));        \
      auto *cpp_block_desc = cpp_desc->AddBlock<cpp::BlockDesc>();       \
      TransformBlockDescAnyToCpp(any_block_desc, cpp_block_desc);        \
    }                                                                    \
  }                                                                      \
                                                                         \
  template <>                                                            \
  void TransformProgramDescCppToAny<NT::T>(const cpp::T &cpp_desc,       \
                                           NT::T *any_desc) {            \
    auto desc = cpp_desc;                                                \
    if (desc.HasVersion()) {                                             \
      any_desc->SetVersion(desc.Version());                              \
    }                                                                    \
                                                                         \
    any_desc->ClearBlocks();                                             \
    for (size_t i = 0; i < desc.BlocksSize(); ++i) {                     \
      auto *cpp_block_desc = desc.GetBlock<cpp::BlockDesc>(i);           \
      auto any_block_desc =                                              \
          NT::BlockDesc(any_desc->AddBlock<PNT::proto::BlockDesc>());    \
      TransformBlockDescCppToAny(*cpp_block_desc, &any_block_desc);      \
    }                                                                    \
  }

TRANS_VAR_ANY_WITH_CPP_IMPL(naive_buffer::VarDesc);
TRANS_OP_ANY_WITH_CPP_IMPL(naive_buffer::OpDesc);
TRANS_BLOCK_ANY_WITH_CPP_IMPL(BlockDesc, naive_buffer, naive_buffer);
TRANS_PROGRAM_ANY_WITH_CPP_IMPL(ProgramDesc, naive_buffer, naive_buffer);

TRANS_VAR_ANY_WITH_CPP_IMPL(fbs::VarDesc);
TRANS_OP_ANY_WITH_CPP_IMPL(fbs::OpDesc);
TRANS_BLOCK_ANY_WITH_CPP_IMPL(BlockDesc, fbs, fbs);
TRANS_PROGRAM_ANY_WITH_CPP_IMPL(ProgramDesc, fbs, fbs);

#ifndef LITE_ON_TINY_PUBLISH
TRANS_VAR_ANY_WITH_CPP_IMPL(pb::VarDesc);
TRANS_OP_ANY_WITH_CPP_IMPL(pb::OpDesc);
TRANS_BLOCK_ANY_WITH_CPP_IMPL(BlockDesc, pb, framework);
TRANS_PROGRAM_ANY_WITH_CPP_IMPL(ProgramDesc, pb, framework);
#endif

#undef TRANS_VAR_ANY_WITH_CPP_IMPL
#undef TRANS_OP_ANY_WITH_CPP_IMPL
#undef TRANS_BLOCK_ANY_WITH_CPP_IMPL
#undef TRANS_PROGRAM_ANY_WITH_CPP_IMPL

}  // namespace lite
}  // namespace paddle

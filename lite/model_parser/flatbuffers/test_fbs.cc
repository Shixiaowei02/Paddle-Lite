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

#include <iostream>
#include <memory>
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/model_parser/flatbuffers/param_generated.h"

using paddle::lite::fbs::proto::CombinedParamsDesc;


int main() {

  flatbuffers::EndianCheck();

  const std::string path {"/shixiaowei02/flatbuffers_test/params.fbs"};
  std::unique_ptr<paddle::lite::model_parser::ByteReader> reader{ new paddle::lite::model_parser::BinaryFileReader{path}};

  auto offset = reader->ReadForward<flatbuffers::uoffset_t>();

  std::unique_ptr<char[]> desc {new char[sizeof(CombinedParamsDesc)]};
  reader->ReadForward(desc.get(), sizeof(CombinedParamsDesc));
  const CombinedParamsDesc* desc = reinterpret_cast<const CombinedParamsDesc*>(desc.get());

  std::cout << "enum value: " << CombinedParamsDesc::VT_PARAMS <<  std::endl;

  std::cout << "right add: " << paddle::lite::fbs::proto::GetCombinedParamsDesc(a.get()) << std::endl;

  std::cout << "desc pointer: " << desc << std::endl;
  std::cout << "params pointer: " << desc->params() << std::endl;


  return 0;
}




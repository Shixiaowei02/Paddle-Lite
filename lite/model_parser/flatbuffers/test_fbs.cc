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

#include <type_traits>
#include <iostream>
#include <memory>
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/model_parser/flatbuffers/param_generated.h"
#include "lite/model_parser/flatbuffers/param_desc.h"
#include "lite/model_parser/flatbuffers/vector_view.h"

void PrintBuffer(const void* pBuff, unsigned int nLen)
{
    if (NULL == pBuff || 0 == nLen)
    {
        return;
    }

    const int nBytePerLine = 16;
    const unsigned char* p = (const unsigned char*)pBuff;
    char szHex[3*nBytePerLine+1] = {0};

    printf("-----------------begin-------------------\n");
    for (unsigned int i=0; i<nLen; ++i)
    {
        int idx = 3 * (i % nBytePerLine);
        if (0 == idx)
        {
            memset(szHex, 0, sizeof(szHex));
        }
#ifdef WIN32
        sprintf_s(&szHex[idx], 4, "%02x ", p[i]);// buff长度要多传入1个字节
#else
        snprintf(&szHex[idx], 4, "%02x ", p[i]); // buff长度要多传入1个字节
#endif
        
        // 以16个字节为一行，进行打印
        if (0 == ((i+1) % nBytePerLine))
        {
            printf("%s\n", szHex);
        }
    }

    // 打印最后一行未满16个字节的内容
    if (0 != (nLen % nBytePerLine))
    {
        printf("%s\n", szHex);
    }

    printf("------------------end-------------------\n");
}

using paddle::lite::fbs::proto::CombinedParamsDesc;

struct alignas(1) Memory {
  void Check() const {
    static_assert(sizeof(flatbuffers::uoffset_t) == 
      sizeof(flatbuffers::Vector<flatbuffers::Offset<
      paddle::lite::fbs::proto::ParamDesc>>));
    CHECK_EQ(offset, 12U);
    CHECK_EQ(params_offset, 4U);
  }

  const CombinedParamsDesc* GetCombinedParamsDesc() const {
    return reinterpret_cast<const CombinedParamsDesc*>(&pdesc);
  }

  flatbuffers::uoffset_t offset; // 4
  uint8_t pdesc_offset[12 - sizeof(flatbuffers::uoffset_t)]; // 8
  uint8_t pdesc[sizeof(CombinedParamsDesc)]; // 1
  uint8_t params_poffset[CombinedParamsDesc::VT_PARAMS - sizeof(CombinedParamsDesc)];  // 3
  flatbuffers::uoffset_t params_offset; // 4
};


int main() {
    const std::string path {"/shixiaowei02/flatbuffers_test/params.fbs"};
    flatbuffers::EndianCheck();

    std::unique_ptr<paddle::lite::model_parser::ByteReader> reader{ new paddle::lite::model_parser::BinaryFileReader{path}};
    Memory memory;
    reader->ReadForward(&memory, sizeof(Memory));
    memory.Check();
    paddle::lite::vector_view::StreamIterator<flatbuffers::Vector<
    flatbuffers::Offset<paddle::lite::fbs::proto::ParamDesc>>> iter(reader.get());


/*
    uint64_t cursor = 0;

    std::vector<flatbuffers::uoffset_t> params_offsets(memory.params_size + 1);
    params_offsets[0] = reader->length() - reader->cursor();
    for (size_t i = 0; i < memory.params_size; ++i) {
      params_offsets[i + 1] = reader->ReadScalarForward<flatbuffers::uoffset_t>() + i * sizeof(flatbuffers::uoffset_t);
      cursor += sizeof(flatbuffers::uoffset_t);
    }

    for (auto offset : params_offsets) {
      std::cout << "offset = " << offset << std::endl;
    }
    std::cout << "cursor = " << cursor << std::endl;
    CHECK_EQ(cursor, *params_offsets.rbegin());

   uint8_t buff__2[100];
   CHECK(buff__2 == buff__2);

    uint8_t buff__[368];
    reader->ReadForward(buff__, params_offsets[0] - params_offsets[3]);
    PrintBuffer(buff__, 368);

    std::vector<size_t> offsets{0, 136-12, 244-12};
    for (size_t i = 0; i < offsets.size(); ++i) {
      std::cout << "vtable: " << flatbuffers::ReadScalar<flatbuffers::soffset_t>(buff__ + offsets[i]) << std::endl;
      paddle::lite::fbs::proto::ParamDesc const* param_ = reinterpret_cast<paddle::lite::fbs::proto::ParamDesc const*>(buff__ + offsets[i]);
      std::cout << "param name: " << param_->name()->c_str() << std::endl;     
    }
*/


/*
    for (int32_t i = forward_sizes.size() - 1; i >=0; --i) {
      std::cout << "forward sizes " << i << " = " << forward_sizes[i] << std::endl;
      buffer.ResetLazy(forward_sizes[i]);
      reader->ReadForward(buffer.data(), forward_sizes[i]);
      paddle::lite::fbs::proto::ParamDesc const* param_ = reinterpret_cast<paddle::lite::fbs::proto::ParamDesc const*>(buffer.data());
      std::cout << "vtable offser: " << +*reinterpret_cast<const uint8_t*>(buffer.data()) << std::endl;
      std::cout << "param name: " << param_->name()->c_str() << std::endl;
      // paddle::lite::fbs::ParamDescView param_view(param_);
      // std::cout << param_view.Name() << std::endl;
    }

*/

  return 0;
}




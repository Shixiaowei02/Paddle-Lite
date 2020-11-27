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

class Derived_ : private flatbuffers::Table {
public:
  Derived_() = default;
  void ShowPointers() const {
    std::cout << "this = " << this << std::endl;
    std::cout << "base this = " << static_cast<flatbuffers::Table const*>(this) << std::endl;
  }
private:
  uint8_t a[1];
};

struct MemorySize {
  static constexpr size_t pdesc_offset() { return 12 - sizeof(flatbuffers::uoffset_t); }
  static constexpr size_t pdesc() { return sizeof(CombinedParamsDesc); }
  static constexpr size_t params_poffset() {
    static_assert(sizeof(flatbuffers::uoffset_t) == sizeof(flatbuffers::Vector<flatbuffers::Offset<paddle::lite::fbs::proto::ParamDesc>>));
    return CombinedParamsDesc::VT_PARAMS - pdesc(); }
};

struct alignas(1) Memory {
  void Check() const {
    CHECK_EQ(offset, 12U);
    CHECK_EQ(params_offset, 4U);
  }

  const CombinedParamsDesc* GetCombinedParamsDesc() const {
    return reinterpret_cast<const CombinedParamsDesc*>(&pdesc);
  }

  const flatbuffers::Vector<flatbuffers::Offset<paddle::lite::fbs::proto::ParamDesc>>* GetParamDescOffsetVec() const {
    return reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<paddle::lite::fbs::proto::ParamDesc>>*>(&params_size);
  }

  const flatbuffers::uoffset_t GetParamsSize() const {
    return params_size;
  }

  flatbuffers::uoffset_t offset; // 4
  uint8_t pdesc_offset[MemorySize::pdesc_offset()]; // 8
  uint8_t pdesc[MemorySize::pdesc()]; // 1
  uint8_t params_poffset[MemorySize::params_poffset()];  // 3
  flatbuffers::uoffset_t params_offset; // 4
  flatbuffers::uoffset_t params_size;
  char a[368];
};



int main() {
    char ppp[100];
    Derived_* a = reinterpret_cast<Derived_*>(ppp);
    a->ShowPointers();



    const std::string path {"/shixiaowei02/flatbuffers_test/params.fbs"};
    flatbuffers::EndianCheck();

    std::unique_ptr<paddle::lite::model_parser::ByteReader> reader{ new paddle::lite::model_parser::BinaryFileReader{path}};
    Memory memory;
    reader->ReadForward(&memory, sizeof(Memory));
    memory.Check();

    const flatbuffers::Vector<flatbuffers::Offset<paddle::lite::fbs::proto::ParamDesc>>* vector = memory.GetParamDescOffsetVec();
    //std::cout << static_cast<const void *>(vector->Get(0)) - static_cast<const void *>(vector->Data()) << std::endl;
    std::cout << "sizeof!! "<<sizeof(paddle::lite::fbs::proto::ParamDesc) << std::endl;
    PrintBuffer(reinterpret_cast<const void*>(vector->Get(1)), 10);
    std::cout << vector->Get(1) << std::endl;
    std::cout << vector->Get(1)->name() << std::endl;
    std::cout << vector->Get(1)->name()->c_str() << std::endl;
    std::cout << vector->Get(1)->variable_as_LoDTensorDesc()->dim()->Get(0) << std::endl;

    uint8_t buf__[150];
    buf__[0] = 10;
    std::memcpy(buf__, vector->Get(1), 150);
    uint8_t debug = buf__[0];
    std::cout << std::dec << "buf__[0] = " << debug << std::endl;
    const paddle::lite::fbs::proto::ParamDesc* param =
      reinterpret_cast<const paddle::lite::fbs::proto::ParamDesc*>(buf__);

    std::cout << "=======" << std::endl;
    std::cout << param << std::endl;
    PrintBuffer(reinterpret_cast<const void*>(param), 20);
    std::cout << param->name() << std::endl;
    std::cout << param->name()->c_str() << std::endl;
    std::cout << param->variable_as_LoDTensorDesc()->dim()->Get(0) << std::endl;
    //std::cout << (void*)vector->Get(2) - vector->Data()<< std::endl;



    /*
    std::vector<flatbuffers::uoffset_t> params_offsets(memory.params_size + 1);
    *params_offsets.begin() = reader->length() - reader->cursor();
    reader->ReadForward(params_offsets.data() + 1, (params_offsets.size() - 1) * sizeof(flatbuffers::uoffset_t));


    for (auto& offset: params_offsets) {
      std::cout << "offset: " << offset << std::endl;
    }

    std::vector<size_t> forward_sizes(params_offsets.size() - 1);
    for (size_t i = 0; i < params_offsets.size() - 1; ++i) {
      forward_sizes[i] = params_offsets[i] - params_offsets[i + 1];
    }

    paddle::lite::model_parser::Buffer buffer;

    for (int32_t i = forward_sizes.size() - 1; i >=0; --i) {
      std::cout << "forward sizes " << i << " = " << forward_sizes[i] << std::endl;
      buffer.ResetLazy(forward_sizes[i]);
      reader->ReadForward(buffer.data(), forward_sizes[i]);
      paddle::lite::fbs::proto::ParamDesc const* param_ = reinterpret_cast<paddle::lite::fbs::proto::ParamDesc const*>(buffer.data());
      std::cout << "param variable: " << param_->variable() << std::endl;
      // paddle::lite::fbs::ParamDescView param_view(param_);
      // std::cout << param_view.Name() << std::endl;
    }
    */



  return 0;
}




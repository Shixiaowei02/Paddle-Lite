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
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

enum class VarDataType {
    // Pod Types
    BOOL = 0,
    INT16 = 1,
    INT32 = 2,
    INT64 = 3,
    FP16 = 4,
    FP32 = 5,
    FP64 = 6,
    // Tensor<size_t> is used in C++.
    SIZE_T = 19,
    UINT8 = 20,
    INT8 = 21,

    // Other types that may need additional descriptions
    LOD_TENSOR = 7,
    SELECTED_ROWS = 8,
    FEED_MINIBATCH = 9,
    FETCH_LIST = 10,
    STEP_SCOPES = 11,
    LOD_RANK_TABLE = 12,
    LOD_TENSOR_ARRAY = 13,
    PLACE_LIST = 14,
    READER = 15,
    // Any runtime decided variable type is raw
    // raw variables should manage their own allocations
    // in operators like nccl_op
    RAW = 17,
    TUPLE = 18
};

class VarDescReadAPI {
 public:
  virtual std::string Name() const = 0;
  virtual VarDataType GetType() const = 0;
  virtual bool Persistable() const = 0;
  virtual std::vector<int64_t> GetShape() const = 0;
  virtual ~VarDescReadAPI() = default;
};

class VarDescWriteAPI {
 public:
  virtual void SetName(std::string name) { NotImplemented(); }
  virtual void SetType(VarDataType type) { NotImplemented(); }
  virtual void SetPersistable(bool persistable) { NotImplemented(); }
  virtual void SetShape(const std::vector<int64_t>& dims) { NotImplemented(); }
  virtual ~VarDescWriteAPI() = default;

 private:
  void NotImplemented() const {
    LOG(FATAL) << "VarDescWriteAPI is not available in model read-only mode.";
  }
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class VarDescAPI : public VarDescReadAPI, public VarDescWriteAPI {
 public:
  using VarDataType = lite::VarDataType;
  using Type = lite::VarDataType;
  virtual ~VarDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle

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

#include "lite/api/light_api.h"
#include <gperftools/heap-profiler.h>
#include <iostream>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

int main() {
  HeapProfilerStart("Test1");
  for (size_t i = 0; i < 10000; ++i) {
    const std::string model_path{
        "/shixiaowei02/Paddle-Lite-v2.7/Paddle-Lite/build_opt/"
        "mobilev1_opt_out.nb"};
    { paddle::lite::LightPredictor predictor(model_path, false); }
    if (i % 100 == 0) {
      HeapProfilerDump("here");
    }
  }
  HeapProfilerStop();
  return 0;
}

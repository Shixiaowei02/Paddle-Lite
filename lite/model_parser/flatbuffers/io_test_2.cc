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

#include <assert.h>
#include <chrono>  // NOLINT
#include <unistd.h>
#include "lite/api/light_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"

class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

int main(int argc, char* argv[]) {
  if (argc <= 4) {
    std::cout << "[FATAL] Usage: ./test_fbs_io {model_path} {warm_up} {iter}"
              << std::endl;
  }
  const std::string path(argv[1]);
  const int warm_up = (atoi(argv[2]));
  const int iter = (atoi(argv[3]));
  for (size_t i = 0; i < warm_up; ++i) {
    paddle::lite::LightPredictor predictor(path, false);
  }
  std::cout << "======= Timer start =======" << std::endl;
  double t = 0.f;
  Timer timer;
  for (size_t i = 0; i < iter; ++i) {
    timer.tic();
    paddle::lite::LightPredictor predictor(path, false);
    t = timer.toc();
    std::cout << "time = " << t << std::endl;
    usleep(100000);
  }

  paddle::lite::LightPredictor predictor(path, false);
  for (auto& name : predictor.GetInputNames()) {
    std::cout << "input_names: " << name << std::endl;
  }
  for (auto& name : predictor.GetOutputNames()) {
    std::cout << "output_names: " << name << std::endl;
  }
  auto* tensor_in = predictor.GetInput(0);
  const int N = 2;
  const int C = 3;
  const int H = 128;
  const int W = 128;

  tensor_in->Resize({N, C, H, W});
  float* data = tensor_in->mutable_data<float>();
  for (size_t i = 0; i < N * C * H * W; ++i) {
    data[i] = 1.f;
  }

  predictor.Run();

  auto* tensor_out = predictor.GetOutput(0);
  auto temp_shape = tensor_out->dims();
  float const* data_o = tensor_out->data<float>();
  for (size_t i = 0; i < temp_shape.production(); ++i) {
    std::cout << "out " << i << ": " << data_o[i] << std::endl;
  }

  return 0;
}

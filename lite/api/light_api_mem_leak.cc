#include "lite/api/light_api.h"
#include <gperftools/heap-profiler.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include <iostream>    

int main() {
  HeapProfilerStart("Test1");
  for (size_t i = 0; i < 10000; ++i) {
    const std::string model_path{"/shixiaowei02/Paddle-Lite-v2.7/Paddle-Lite/build_opt/mobilev1_opt_out.nb"};
    {
      paddle::lite::LightPredictor predictor(model_path, false);
      paddle::lite::LightPredictor predictor2(model_path, false);
      paddle::lite::LightPredictor* p;
      int size = 100;

      for (size_t j = 0; j < 10000; ++j) {
        if (j % 2 == 0) {
          p = &predictor;
          size = 100;
        } else {
          p = &predictor2;
          size = 200;
        }
        for (const auto& name: p->GetInputNames()) {
          auto* input_tensor = p->GetInputByName(name);
          input_tensor->Resize(paddle::lite::DDim(std::vector<int64_t>({1, 3, size, 100})));
          auto* data = input_tensor->mutable_data<float>();
          for (int i = 0; i < 1 * 3 * size * 100; i++) {
            data[i] = i;
          }
        }
        p->Run();
        if (j % 15 == 0) {
          HeapProfilerDump("here2");
        }
      }
    }
  }
  HeapProfilerStop();
  return 0;
}
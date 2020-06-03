#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"
#include <google/protobuf/text_format.h>
#include "gperftools/profiler.h"
#include <chrono>  // NOLINT

// Timer for timer
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

int main() {
  paddle::lite::cpp::ProgramDesc cpp_prog;
  //double t = 0;

  // Timer timer;
  //timer.tic();
  ProfilerStart("test_capture.prof");
  for (size_t i = 0; i < 1000; ++i) {
  paddle::lite::LoadModelFbs("/shixiaowei02/Paddle-Lite-FlatBuf/framework_test/save_model.bin", &cpp_prog);
  }
  ProfilerStop();
  //t = timer.toc();
  //std::cout << "time = " << t << std::endl;

/*

  paddle::framework::proto::ProgramDesc pb_proto_prog;
  paddle::lite::pb::ProgramDesc pb_prog(&pb_proto_prog);
  paddle::lite::TransformProgramDescCppToAny(cpp_prog, &pb_prog);
  std::string pb_str;
  google::protobuf::TextFormat::PrintToString(pb_proto_prog, &pb_str);
  std::cout << pb_str << std::endl;
*/
  return 0;
}

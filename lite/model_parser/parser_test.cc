#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/pb/program_desc.h"
#include "lite/model_parser/flatbuffers/program_desc.h"
#include "lite/model_parser/flatbuffers/ro/program_desc.h"
#include "lite/model_parser/pb/var_desc.h"
#include <google/protobuf/text_format.h>
#include "gperftools/profiler.h"
#include "lite/model_parser/compatible_pb.h"
#include "flatbuffers/idl.h"
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
#if 0
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
#elif 0

int main() {
  double t = 0;
  Timer timer;
  timer.tic();
  paddle::lite::cpp::ProgramDesc cpp_prog;
  paddle::lite::LoadModelFbs("/shixiaowei02/Paddle-Lite-FlatBuf/framework_test/save_model.bin", &cpp_prog);
  t = timer.toc();
  std::cout << "time = " << t << std::endl;
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
#elif 1
int main() {
  paddle::lite::cpp::ProgramDesc cpp_prog;
  paddle::lite::LoadModelFbs("/shixiaowei02/Paddle-Lite-FlatBuf/framework_test/save_model.bin", &cpp_prog);
  paddle::lite::fbs::ProgramDesc fbs_program;
  paddle::lite::TransformProgramDescCppToAny(cpp_prog, &fbs_program);
  /*
  LOG(INFO) << "(fbs_program.blocks.size() = " << fbs_program.blocks.size();
  LOG(INFO) << "(fbs_program.blocks[0])->ops = " << &((fbs_program.blocks[0])->ops);
  LOG(INFO) << "(fbs_program.blocks[0])->ops.size() = " << (fbs_program.blocks[0])->ops.size();
  LOG(INFO) << "(fbs_program.blocks[0])->vars.size() = " << (fbs_program.blocks[0])->vars.size();
  */

  auto& buffer = fbs_program.SyncBuffer();
  auto* data = buffer.data();
  auto* fbs_prog = paddle::lite::fbs::proto::GetProgramDesc(data);

  LOG(INFO) << "fbs_prog->blocks()->size() = " << fbs_prog->blocks()->size();
  LOG(INFO) << "*fbs_prog->blocks())[i]->ops() = " << (*fbs_prog->blocks())[0]->ops();
  LOG(INFO) << "*fbs_prog->blocks())[i]->vars() = " << (*fbs_prog->blocks())[0]->vars();
  for (size_t i = 0; i < fbs_prog->blocks()->size(); ++i) {
    for (size_t j = 0; j < (*fbs_prog->blocks())[i]->ops()->size(); ++j) {
      auto op = (*(*fbs_prog->blocks())[i]->ops())[j];
      std::cout << op->type() << std::endl;
    }
  }

/*
  paddle::lite::cpp::ProgramDesc cpp_prog_2;
  paddle::lite::TransformProgramDescAnyToCpp(paddle::lite::fbs::ro::ProgramDesc(
    const_cast<paddle::lite::fbs::proto::ProgramDesc*>(fbs_prog)), &cpp_prog_2);


  paddle::framework::proto::ProgramDesc pb_proto_prog;
  paddle::lite::pb::ProgramDesc pb_prog(&pb_proto_prog);
  paddle::lite::TransformProgramDescCppToAny(cpp_prog_2, &pb_prog);
  std::string pb_str;
  google::protobuf::TextFormat::PrintToString(pb_proto_prog, &pb_str);
  std::cout << pb_str << std::endl;
*/
  return 0;
}
#endif

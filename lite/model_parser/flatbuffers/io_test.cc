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

#include "lite/model_parser/flatbuffers/io.h"
#include "lite/model_parser/model_parser.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/pb/program_desc.h"
#include <google/protobuf/text_format.h>
#include "flatbuffers/idl.h"

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
//#include "lite/api/paddle_use_passes.h"


#include "lite/api/light_api.h"

namespace paddle {
namespace lite {
  namespace cpp = fbs;
}  // namespace lite
}  // namespace paddle

int main() {

  using paddle::lite::VectorView;
  //using namespace paddle::lite::cpp;
  std::shared_ptr<paddle::lite::cpp::ProgramDesc> prog(new paddle::lite::cpp::ProgramDesc());
  //paddle::lite::cpp::ProgramDesc prog;
  std::string path("/shixiaowei02/Paddle-Lite-FlatBuf/v5_3_5_gesture/flatbuffers/model.fbs");
  //std::string path("/shixiaowei02/Paddle-Lite-FlatBuf/framework_test/save_model.bin");
  LoadModel(path, prog.get());

/*
  std::string schemafile;
  flatbuffers::LoadFile("/shixiaowei02/Paddle-Lite-VectorView/Paddle-Lite/lite/model_parser/flatbuffers/framework.fbs", false, &schemafile);
  flatbuffers::Parser parser;
  parser.Parse(schemafile.c_str());
  std::string jsongen;
  if (!GenerateText(parser, prog->raw_buf(), &jsongen)) {
      std::cout << "Couldn't serialize parsed data to JSON!" << std::endl;
  }
  std::cout<< jsongen << std::endl;
  std::cout << "Here!!" << std::endl;
*/

/*
  paddle::framework::proto::ProgramDesc pb_proto_prog;
  paddle::lite::pb::ProgramDesc pb_prog(&pb_proto_prog);
  paddle::lite::TransformProgramDescCppToAny(*prog, &pb_prog);
  std::string pb_str;
  google::protobuf::TextFormat::PrintToString(pb_proto_prog, &pb_str);
  std::cout << pb_str << std::endl;
*/

  paddle::lite::LightPredictor predictor("/shixiaowei02/Paddle-Lite-FlatBuf/v5_3_5_gesture/flatbuffers/", false);
  
  return 0;
}

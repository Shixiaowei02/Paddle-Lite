#include "lite/model_parser/model_parser.h"

int main() {
  paddle::lite::cpp::ProgramDesc prog;
  paddle::lite::LoadModelFbs("/shixiaowei02/Paddle-Lite-FlatBuf/framework_test/save_model.bin", &prog);
  return 0;
}

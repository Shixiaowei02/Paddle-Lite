/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>
#include <tvm/relay/base.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/nn.h>
#include <topi/generic/default.h>
#include <topi/generic/injective.h>
#include <topi/nn.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <topi/xpu/xdnn_ops.h>
#include <topi/xpu/softmax.h>
#include <xtcl/xtcl.h>
#include "xpu_test_utils.h"

using namespace xtcl;
using namespace xtcl::network;
using namespace tvm::runtime;

TEST(XPUSoftmax, SoftmaxGraph) {
  using namespace tvm;
  const int m = 1;
  const int n = 1024;

  xNetworkBuilder nbuilder;
  xExpr A = nbuilder.CreateTensor("A", {m, n}, tvm::Float(32));
  xExpr net = nbuilder.CreateSoftmax(A, 0);
  xNetwork func = nbuilder.FinalizeNetwork(net);

  // build
  xTensorCompiler nCompiler(func);
  nCompiler.Build();
  xRuntimeInstance nRuntime = nCompiler.CreateRuntimeInstance();

  // set input value
  auto array_a =
      NDArray::Empty({m, n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  RandomNDArray(array_a, 0.0, 1.0, m * n);
  nRuntime.SetInput("A", array_a);

  // run
  nRuntime.Run();

  // get the result
  auto array_b =
      NDArray::Empty({m, n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  TensorNDArray resultDevice = nRuntime.GetOutput(0);
  resultDevice.CopyTo(array_b);

  // check the correctness
  auto result = (float*)array_b.ToDLPack()->dl_tensor.data;
  auto expect = new float[m * n];
  auto pa = (float*)array_a.ToDLPack()->dl_tensor.data;
  baidu::xpu::api::cpu_mock::softmax2d_forward(0, pa, expect, m, n);
  assert_allclose(result, expect, m * n, 1e-5);
}

TEST(XPURelu, SoftReluGraph) {
  // Add test logic here
}

TEST(XPULeakyRelu, SoftLeakyReluGraph) {
  // Add test logic here
  using namespace tvm;
  const int m = 1;
  const int n = 1024;

  xNetworkBuilder nbuilder;
  xExpr A = nbuilder.CreateTensor("A", {m, n}, tvm::Float(32));
  xExpr net = nbuilder.CreateLeakyRelu(A, 0.1);
  xNetwork func = nbuilder.FinalizeNetwork(net);

  // build
  xTensorCompiler nCompiler(func);
  nCompiler.Build();
  xRuntimeInstance nRuntime = nCompiler.CreateRuntimeInstance();

  // set input value
  auto array_a =
      NDArray::Empty({m, n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  RandomNDArray(array_a, 0.0, 1.0, m * n);
  nRuntime.SetInput("A", array_a);

  // don't run now ...
  //nRuntime.Run();
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}

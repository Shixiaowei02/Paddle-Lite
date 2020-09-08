// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(class__) \
  class__(const class__&) = delete;       \
  class__& operator=(const class__&) = delete;
#endif

#define LITE_UNIMPLEMENTED CHECK(false) << "Not Implemented";

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

/*
#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif

#ifndef UNLIKELY
//#define UNLIKELY(x) __built_expect(!!(x), 0)
#define UNLIKELY(x) (x)
#endif
 */

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

#if defined(__FLT_MAX__)
#define FLT_MAX __FLT_MAX__
#endif  // __FLT_MAX__

#define ATTRIBUTE_TLS
#elif __cplusplus >= 201103
#define ATTRIBUTE_TLS thread_local
#else
#error "C++11 support is required for paddle-lite compilation."
#endif

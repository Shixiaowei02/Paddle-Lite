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

#include "lite/utils/span.h"
#include <gtest/gtest.h>
#include <string>

namespace paddle {
namespace lite {

template <typename T>
void span_vector_test() {
  std::vector<T> vec({1, 2, 3});
  lite::Span<T> span(vec);
  ASSERT_EQ(span.size(), vec.size());
  for (size_t i = 0; i < span.size(); ++i) {
    ASSERT_EQ(span[i], vec[i]);
  }
  ASSERT_EQ(span.front(), vec.front());
  ASSERT_EQ(span.back(), vec.back());
  ASSERT_EQ(span.at(0), vec.at(0));
  ASSERT_EQ(*span.begin(), *vec.begin());
  ASSERT_EQ(*(span.end() - 1), *(vec.end() - 1));
  ASSERT_EQ(*span.rend(), *vec.rend());
  ASSERT_EQ(span.size(), static_cast<size_t>(3));
  ASSERT_EQ(span.size_bytes(), 3 * sizeof(T));
  ASSERT_EQ(span.empty(), false);
}

template <typename T>
void span_vector_const_test() {
  const std::vector<T> vec({1, 2, 3});
  lite::Span<const T> span(vec);
  ASSERT_EQ(span.size(), vec.size());
  for (size_t i = 0; i < span.size(); ++i) {
    ASSERT_EQ(span[i], vec[i]);
  }
  ASSERT_EQ(span.front(), vec.front());
  ASSERT_EQ(span.back(), vec.back());
  ASSERT_EQ(span.at(0), vec.at(0));
  ASSERT_EQ(*span.begin(), *vec.begin());
}

template <>
void span_vector_test<std::string>() {
  std::vector<std::string> vec({"Bryce", "John", "Bob!"});
  lite::Span<std::string> span(vec);
  ASSERT_EQ(span.size(), vec.size());
  size_t i = 0;
  for (const auto& e : span) {
    ASSERT_EQ(e, vec[i]);
    ++i;
  }
}

template <typename T>
void span_array_test() {
  T arr[3] = {1, 2, 3};
  lite::Span<T> span(arr);
  ASSERT_EQ(span.size(), size_t(3));
  for (size_t i = 0; i < span.size(); ++i) {
    ASSERT_EQ(span[i], arr[i]);
  }
}

TEST(span, test) {
  paddle::lite::span_vector_test<int64_t>();
  paddle::lite::span_vector_test<float>();
  paddle::lite::span_vector_test<std::string>();
  paddle::lite::span_array_test<int64_t>();
}

TEST(span, const_test) {
  paddle::lite::span_vector_const_test<int64_t>();
  paddle::lite::span_vector_const_test<float>();
}

TEST(span, mutable_and_swap_test) {
  int const value = 128;
  std::vector<int64_t> vec({1, 2, 3});
  lite::Span<int64_t> span(vec);
  span.front() = value - 1;
  span[1] = value;
  span.back() = value + 1;
  lite::Span<int64_t> span_other;
  span_other.swap(span);
  ASSERT_EQ(vec.front(), value - 1);
  ASSERT_EQ(vec[1], value);
  ASSERT_EQ(vec.back(), value + 1);
}

}  // namespace lite
}  // namespace paddle

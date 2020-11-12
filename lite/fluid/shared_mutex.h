/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <climits>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT

namespace paddle {
namespace lite {
namespace details {

class SharedMutex {
 public:
  SharedMutex() : state_{0} {}
  ~SharedMutex() { std::lock_guard<mutex_t> _(mut_); }

  SharedMutex(const SharedMutex&) = delete;
  SharedMutex& operator=(const SharedMutex&) = delete;

  void lock() {
    std::unique_lock<mutex_t> lk(mut_);
    while (state_ & write_entered_) {
      gate1_.wait(lk);
    }
    state_ |= write_entered_;
    while (state_ & n_readers_) {
      gate2_.wait(lk);
    }
  }

  bool try_lock() {
    std::unique_lock<mutex_t> lk(mut_);
    if (state_ == 0) {
      state_ = write_entered_;
      return true;
    }
    return false;
  }

  void unlock() {
    std::lock_guard<mutex_t> _(mut_);
    state_ = 0;
    gate1_.notify_all();
  }

  void lock_shared() {
    std::unique_lock<mutex_t> lk(mut_);
    while ((state_ & write_entered_) || (state_ & n_readers_) == n_readers_) {
      gate1_.wait(lk);
    }
    count_t num_readers = (state_ & n_readers_) + 1;
    state_ &= ~n_readers_;
    state_ |= num_readers;
  }

  bool try_lock_shared() {
    std::unique_lock<mutex_t> lk(mut_);
    count_t num_readers = state_ & n_readers_;
    if (!(state_ & write_entered_) && num_readers != n_readers_) {
      ++num_readers;
      state_ &= ~n_readers_;
      state_ |= num_readers;
      return true;
    }
    return false;
  }

  void unlock_shared() {
    std::lock_guard<mutex_t> _(mut_);
    count_t num_readers = (state_ & n_readers_) + 1;
    state_ &= ~n_readers_;
    state_ |= num_readers;
    if ((state_ & write_entered_) && (num_readers == 0)) {
      gate2_.notify_one();
    } else if (num_readers == n_readers_ - 1) {
      gate1_.notify_one();
    }
  }

 private:
  typedef ::std::mutex mutex_t;
  typedef unsigned int count_t;

  mutex_t mut_;
  std::condition_variable gate1_;
  std::condition_variable gate2_;
  count_t state_;

  static constexpr count_t write_entered_ = 1U
                                            << (sizeof(count_t) * CHAR_BIT - 1);
  static constexpr count_t n_readers_ = ~write_entered_;
};

}  // namespace details
}  // namespace lite
}  // namespace paddle

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

#pragma once

#include <string>
#include <vector>
#include <stack>
#include "flatbuffers/flatbuffers.h"
#include "lite/model_parser/base/vector_view.h"
#include "lite/model_parser/base/io.h"

namespace paddle {
namespace lite {
namespace vector_view {

template <typename T>
class StreamIterator;

template <typename T>
struct ElementTraits<T*,
                     typename std::enable_if<std::is_class<T>::value>::type> {
  typedef flatbuffers::Offset<T> element_type;
};

template <>
struct ElementTraits<std::string, void> {
  typedef flatbuffers::Offset<flatbuffers::String> element_type;
};

template <typename T>
struct VectorTraits<T, Flatbuffers> {
  typedef flatbuffers::Vector<typename ElementTraits<T>::element_type>
      vector_type;
  typedef typename vector_type::const_iterator const_iterator;
  typedef StreamIterator<T> const_stream_iterator;
  typedef typename const_iterator::value_type value_type;
  typedef const typename const_iterator::reference const_reference;
  typedef value_type subscript_return_type;
};

struct FBSStrIterator {
  typedef flatbuffers::VectorIterator<
      flatbuffers::Offset<flatbuffers::String>,
      typename flatbuffers::IndirectHelper<
          flatbuffers::Offset<flatbuffers::String>>::return_type>
      VI;

  FBSStrIterator() = default;
  explicit FBSStrIterator(const VI& iter) { iter_ = iter; }
  const VI& raw_iter() const { return iter_; }

  bool operator==(const FBSStrIterator& other) const {
    return iter_ == other.raw_iter();
  }

  bool operator<(const FBSStrIterator& other) const {
    return iter_ < other.raw_iter();
  }

  bool operator!=(const FBSStrIterator& other) const {
    return iter_ != other.raw_iter();
  }

  ptrdiff_t operator-(const FBSStrIterator& other) const {
    return iter_ - other.raw_iter();
  }

  std::string operator*() const { return iter_.operator*()->str(); }
  std::string operator->() const { return iter_.operator->()->str(); }

  FBSStrIterator& operator++() {
    iter_++;
    return *this;
  }

  FBSStrIterator& operator--() {
    iter_--;
    return *this;
  }

  FBSStrIterator operator+(const size_t& offset) {
    return FBSStrIterator(iter_ + offset);
  }

  FBSStrIterator operator-(const size_t& offset) {
    return FBSStrIterator(iter_ - offset);
  }

 private:
  VI iter_;
};

template <typename T>
bool operator==(const StreamIterator<T>& lhs, const StreamIterator<T>& rhs) {
  bool res = true;
  res = res && (lhs.reader_ == rhs.reader_);
  res = res && (lhs.outset_ == rhs.outset_);
  if (lhs.reader_ && rhs.reader_) {
    res = res && (lhs.bytes_offset() == rhs.bytes_offset());
  }
  return res;
}

template <typename T>
class StreamIterator<flatbuffers::Vector<flatbuffers::Offset<T>>> {
public:
StreamIterator() = default;
explicit StreamIterator(model_parser::ByteReader* reader) : reader_{reader} {
  CHECK(reader_) << "The reader pointer of stream iterator is nullptr";
  outset_ = reader_->cursor();
  size_ = reader_->ReadScalarForward<flatbuffers::uoffset_t>();
  VLOG(5) << "The size of stream vector is " << size_;
  InitRecord();
}
~StreamIterator() = default;

StreamIterator& operator++() {
  voffset_.current = voffset_.next;
  CHECK(section_bytes_.size()) << "Nothing to read.";

  size_t data_size = section_bytes_.top();
  data_.ResetLazy(data_size + std::abs(voffset_.current));
  std::memcpy(data_.data(), next_header_.data(), std::abs(voffset_.current));
  std::memcpy(data_.data() + std::abs(voffset_.current), &voffset_.current, sizeof(flatbuffers::soffset_t));
  reader_->ReadForward(data_.data() + std::abs(voffset_.current) + sizeof(flatbuffers::soffset_t), data_size 
    - sizeof(flatbuffers::soffset_t));

  section_bytes_.pop();

  if (section_bytes_.size()) {
    voffset_.next = reader_->ReadScalarForward<flatbuffers::soffset_t>();
    CHECK_GT(voffset_.next, 0) << "The vtable offset of back elements should be greater than 0.";
    next_header_.ResetLazy(voffset_.next);
    std::memcpy(next_header_.data(), data_.data() + std::abs(voffset_.current) + data_size - voffset_.next 
      , voffset_.next);
  } else {
    voffset_.next = 0;
    reader_ = nullptr;
    outset_ = 0;
  }
  return *this;
}

const T& operator*() {
  void* p = data_.data() + std::abs(voffset_.current);
  return *reinterpret_cast<const T*>(p);
}

const T* operator->() {
  void* p = data_.data() + std::abs(voffset_.current);
  return reinterpret_cast<const T*>(p);
}

template <typename U>
friend bool operator==(const StreamIterator<U>& lhs, const StreamIterator<U>& rhs);

private: 
  struct VOffset {
    flatbuffers::soffset_t current;
    flatbuffers::soffset_t next;
  };
  size_t bytes_offset() const {
    return reader_->cursor() - outset_;
  }
  void InitRecord() {
    std::vector<size_t> section_offsets(size_ + 1);
    section_offsets[0] = reader_->length() - reader_->cursor();
    for (size_t i = 0; i < size_; ++i) {
      section_offsets[i + 1] = reader_->ReadScalarForward<flatbuffers::uoffset_t>() + 
        i * sizeof(flatbuffers::uoffset_t);
    }
    CHECK_EQ(bytes_offset() - sizeof(flatbuffers::uoffset_t), *section_offsets.rbegin());
    for (auto p = section_offsets.begin(); p != section_offsets.end() - 1; ++p) {
      section_bytes_.push(*p - *(p + 1));
    }
    voffset_.next = reader_->ReadScalarForward<flatbuffers::soffset_t>();
    CHECK_LT(voffset_.next, 0) << "The vtable offset of first element should be less than 0.";
    next_header_.ResetLazy(std::abs(voffset_.next));
    operator++();
  }
  model_parser::Buffer data_;
  model_parser::Buffer next_header_;
  VOffset voffset_{0, 0};
  model_parser::ByteReader* reader_{nullptr};
  std::stack<size_t> section_bytes_;
  size_t outset_{0};
  size_t size_{0};
};

}  // namespace vector_view

template <>
class VectorView<std::string, Flatbuffers> {
 public:
  typedef vector_view::VectorTraits<std::string, Flatbuffers> Traits;
  explicit VectorView(typename Traits::vector_type const* cvec) {
    cvec_ = cvec;
  }
  std::string operator[](size_t i) const { return cvec_->operator[](i)->str(); }
  vector_view::FBSStrIterator begin() const {
    if (!cvec_) {
      return vector_view::FBSStrIterator();
    }
    return vector_view::FBSStrIterator(cvec_->begin());
  }
  vector_view::FBSStrIterator end() const {
    if (!cvec_) {
      return vector_view::FBSStrIterator();
    }
    return vector_view::FBSStrIterator(cvec_->end());
  }
  size_t size() const {
    if (!cvec_) {
      return 0;
    }
    return cvec_->size();
  }
  operator std::vector<std::string>() const {
    VLOG(5) << "Copying elements out of VectorView will damage performance.";
    std::vector<std::string> tmp;
    tmp.resize(size());
    for (size_t i = 0; i < size(); ++i) {
      tmp[i] = cvec_->operator[](i)->str();
    }
    return tmp;
  }
  ~VectorView() = default;

 private:
  typename Traits::vector_type const* cvec_;
};

}  // namespace lite
}  // namespace paddle

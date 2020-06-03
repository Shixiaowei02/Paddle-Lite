lite_cc_library(fbs_op_desc SRCS op_desc.cc DEPS flatbuffers)
lite_cc_library(fbs_var_desc SRCS var_desc.cc DEPS flatbuffers)
lite_cc_library(fbs_block_desc SRCS block_desc.cc DEPS flatbuffers)
lite_cc_library(fbs_program_desc SRCS program_desc.cc DEPS flatbuffers)

set(fbs_wrapper fbs_op_desc fbs_var_desc fbs_block_desc fbs_program_desc PARENT_SCOPE)

add_subdirectory(ro)

lite_cc_library(fbs_var_desc_ SRCS var_desc.cc)
lite_cc_library(fbs_block_desc_ SRCS block_desc.cc)
lite_cc_library(fbs_program_desc_ SRCS program_desc.cc)
lite_cc_library(fbs_op_desc_ SRCS op_desc.cc)

set(XTCL_PATH "/shixiaowei02/xtml/xmir-output")

if (NOT XTCL_PATH)
    message(STATUS "==== No XTCL_SUBGRAPH! ====")
    return()
else()
    # Print path of subgraph.
    message(STATUS "==== Build XTCL_SUBGRAPH! ====")
    message(STATUS "XTCL_PATH: ${XTCL_PATH}")

    find_library(XTCL_XPURT_SHARED_PATH NAMES libxpurt.so
        PATHS ${XTCL_PATH}/shlib/
        NO_DEFAULT_PATH
        DOC "Path to lite library.")
    find_library(XTCL_TVM_SHARED_PATH NAMES libtvm.so
        PATHS ${XTCL_PATH}/shlib/
        NO_DEFAULT_PATH
        DOC "Path to lite library.")
    find_library(XTCL_XPUAPI_SHARED_PATH NAMES libxpuapi.so
        PATHS ${XTCL_PATH}/shlib/
        NO_DEFAULT_PATH
        DOC "Path to lite library.")
    find_library(XTCL_XTCL_STATIC_PATH NAMES libxtcl.a
        PATHS ${XTCL_PATH}/lib/
        NO_DEFAULT_PATH
        DOC "Path to lite library.")

    add_library(xpurt SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpurt PROPERTY IMPORTED_LOCATION ${XTCL_XPURT_SHARED_PATH})
    add_library(tvm SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET tvm PROPERTY IMPORTED_LOCATION ${XTCL_TVM_SHARED_PATH})
    add_library(xpuapi SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XTCL_XPUAPI_SHARED_PATH})
    add_library(xtcl STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xtcl PROPERTY IMPORTED_LOCATION ${XTCL_XTCL_STATIC_PATH})

    # Make target model_optimize_tool. The sequence of glog gflags needs to be attention.
    message(STATUS "XTCL_XPURT_SHARED_PATH: ${XTCL_XPURT_SHARED_PATH}")
    message(STATUS "XTCL_TVM_SHARED_PATH: ${XTCL_TVM_SHARED_PATH}")
    message(STATUS "XTCL_XPUAPI_SHARED_PATH: ${XTCL_XPUAPI_SHARED_PATH}")
    include_directories(${XTCL_PATH}/include)

    set_source_files_properties(xpu_xdnn_xtcl_test.cc PROPERTIES COMPILE_FLAGS "-Wno-error")
    cc_binary(test_leaky_relu SRCS xpu_xdnn_xtcl_test.cc DEPS xtcl xpurt tvm xpuapi glog gflags gtest)

    # make op teller
    # cc_library(lite_op_teller SRCS op_teller.cc DEPS framework_proto device_context boost)

    # make op teller
    # cc_library(lite_engine SRCS engine.cc DEPS api_full_static)
endif()

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

SET(FLATBUFFERS_SOURCES_DIR ${CMAKE_SOURCE_DIR}/third-party/flatbuffers)
SET(FLATBUFFERS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flatbuffers)
SET(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_INSTALL_DIR}/include" CACHE PATH "flatbuffers include directory." FORCE)
IF(WIN32)
  set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/lib64/libflatbuffers.lib" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ELSE(WIN32)
  set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/lib64/libflatbuffers.a" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${FLATBUFFERS_INCLUDE_DIR})

SET(OPTIONAL_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                  "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                  "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
                  "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
                  "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
                  "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
                  "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

ExternalProject_Add(
    extern_flatbuffers
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  ""
    GIT_TAG         "v1.12.0"
    SOURCE_DIR      ${FLATBUFFERS_SOURCES_DIR}
    PREFIX          ${FLATBUFFERS_INCLUDE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DBUILD_STATIC_LIBS=ON
                    -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${CROSS_COMPILE_CMAKE_ARGS}
                    ${OPTIONAL_ARGS}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
IF(WIN32)
  IF(NOT EXISTS "${FLATBUFFERS_INSTALL_DIR}/lib64/libflatbuffers.lib")
    add_custom_command(TARGET extern_flatbuffers POST_BUILD
            COMMAND cmake -E copy ${FLATBUFFERS_INSTALL_DIR}/lib64/flatbuffers_static.lib ${FLATBUFFERS_INSTALL_DIR}/lib64/libflatbuffers.lib
            )
  ENDIF()
ENDIF(WIN32)
ADD_LIBRARY(flatbuffers STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${FLATBUFFERS_LIBRARIES})
ADD_DEPENDENCIES(flatbuffers extern_flatbuffers)

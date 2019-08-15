# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(SUPG)

if (WIN32)
    if (CMAKE_GENERATOR MATCHES "Visual Studio 15 2017")
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/Release/opennn.lib)
	else()
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/libopennn.a)
	endif()
endif (WIN32)

if (UNIX)
    if (CMAKE_GENERATOR MATCHES "Visual Studio 15 2017")
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/Release/opennn.lib)
	else()
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/libopennn.a)
	endif()
endif (UNIX)

include_directories(./data)

FILE(COPY ./data/SUPG.txt DESTINATION "${PROJECT_BINARY_DIR}/data")

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)
include_directories(
    /vagrant/isoglib/BuildDebug
      /vagrant/isoglib/isoglib
      /vagrant/isoglib/isoglib/Core
      /vagrant/isoglib/isoglib/Problem
      /vagrant/isoglib/isoglib/TestCase
      /vagrant/isoglib/isoglib/LocalMatrices
      /vagrant/isoglib/isoglib/Epetra
      /u/sw/pkgs/toolchains/gcc-glibc/7/base/include/
      /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/eigen/3.3.3/include/eigen3/
      /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/trilinos/12.6.3/include
      /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/boost/1.63.0/include/boost/predef/os/
      /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/boost/1.63.0/include/
    )

add_executable(SUPG main.cpp)

target_link_libraries(SUPG ${PROJECT_LINK_LIBS})

target_link_libraries(SUPG /vagrant/isoglib/BuildDebug/libisoglib.so)
target_link_libraries(SUPG /vagrant/isoglib/BuildDebug/libisoglib_epetra.so)

target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/base/lib/liblua.a)
target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/prefix/lib/libdl.so)

target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/metis/5/lib/libparmetis.so)

target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/trilinos/12.6.3/lib/libepetra.so)
target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/trilinos/12.6.3/lib/libctrilinos.so)
target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/pkgs/trilinos/12.6.3/lib/libml.so)

target_link_libraries(SUPG /u/sw/pkgs/toolchains/gcc-glibc/7/base/lib/libmpi.so)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CXXFLAGS "${CXXFLAGS} -fPIC")
project(RAHE)
cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_BUILD_TYPE Release)

find_package(Threads REQUIRED)
if(THREAD_HAVE_PTHREAD_ARG)
	target_compile_options(PUBLIC RAHE "-pthread")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#FIND_PACKAGE(OpenBlas)

find_package(OpenCV REQUIRED)

#add_definitions(-DBLOCKMODE)
#find the files for darknet
file(GLOB_RECURSE sources darknet/src/*.c darknet/src/*.h RectangleBinPack/*.cpp  RectangleBinPack/*.h src/*)
set(sources ${sources} CppMT/common.cpp CppMT/gui.cpp CppMT/CMT.cpp 
    CppMT/Consensus.cpp CppMT/Fusion.cpp CppMT/Matcher.cpp CppMT/Tracker.cpp
    CppMT/fastcluster/fastcluster.cpp)
file(GLOB data darknet/data/*)
file(GLOB labels danknet/data/labels/*)
include_directories(darknet/src RectangleBinPack/ src/ CppMT/ ${OpenBLAS_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
add_executable(RAHE ${sources})


if(CMAKE_THREAD_LIBS_INIT)
	target_link_libraries(RAHE "${CMAKE_THREAD_LIBS_INIT}")
endif()
target_link_libraries(RAHE ${OpenBLAS_LIB} ${OpenCV_LIBS})

file(COPY ${data} DESTINATION data)
file(COPY ${labels} DESTINATION data/labels)
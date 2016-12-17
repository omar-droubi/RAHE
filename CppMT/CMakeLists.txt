project (CMT)

cmake_minimum_required (VERSION 2.6)

find_package(OpenCV REQUIRED)

#add_definitions ("-Wall")

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(cmt common.cpp gui.cpp CMT.cpp 
    Consensus.cpp Fusion.cpp Matcher.cpp Tracker.cpp
    fastcluster/fastcluster.cpp)

target_link_libraries(cmt ${OpenCV_LIBS})


CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(image_process)

SET(ALL_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
	)
INCLUDE_DIRECTORIES(${ALL_INCLUDE})

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

SET(CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/png.cpp
)

SET(ALL_SRC
    ${CORE_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

SET(ALL_LIB
    opencv-core
    opencv-imgproc
    opencv-imgcodecs
)

ADD_EXECUTABLE(${PROJECT_NAME}
    ${ALL_SRC}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ALL_LIB})

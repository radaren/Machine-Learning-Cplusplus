cmake_minimum_required(VERSION 3.1.0)
project(multi)
SET(ALL_CODE_LIST
	MUL_DEFINE.h
	nb.hpp
	textReader.hpp
	NBAB.hpp
	Adaboost.hpp
	Adaboost.cpp
	main.cpp
	NBAB.cpp
	nb.cpp
	textReader.cpp
)
message(ALL_CODE_LIST)
add_executable(multi ${ALL_CODE_LIST})
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
target_compile_features(multi PRIVATE cxx_range_for)

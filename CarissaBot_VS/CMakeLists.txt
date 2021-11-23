# CMakeList.txt : CMake project for CarissaBot_VS, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("CarissaBot_VS")
find_package(Torch REQUIRED)

# Add source to this project's executable.
add_executable (CarissaBot_VS "CarissaBot_VS.cpp" "CarissaBot_VS.h")
add_executable (Demo "demo.cpp")
target_link_libraries(CarissaBot_VS "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CarissaBot_VS
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CarissaBot_VS>)
endif (MSVC)

# TODO: Add tests and install targets if needed.

cmake_minimum_required(VERSION 3.5)
project(AtomicExec)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if("$ENV{ATOMIC_EXEC_ALONE}" STREQUAL "1")
    message("In alone.cmake")
    include(alone.cmake)
else()
    message("In intergrated.cmake")
    include(integrated.cmake)
endif()
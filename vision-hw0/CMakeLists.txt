cmake_minimum_required(VERSION 3.16)
project(vision-hw0 C)

set(CMAKE_C_STANDARD 99)

add_executable(vision-hw0 src/args.c src/args.h src/image.h src/load_image.c
    src/process_image.c src/stb_image.h src/stb_image_write.h src/test.c src/test.h)

target_link_libraries(vision-hw0 m)
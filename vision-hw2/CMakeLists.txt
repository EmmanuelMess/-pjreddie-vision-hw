cmake_minimum_required(VERSION 3.16)
project(vision-hw2 C)

set(CMAKE_C_STANDARD 99)

add_executable(vision-hw2
    src/args.c
    src/args.h
    src/harris_image.c
    src/image.h
    src/load_image.c
    src/main.c
    src/matrix.c
    src/matrix.h
    src/panorama_image.c
    src/stb_image.h
    src/stb_image_write.h
    src/test.c
    src/test.h
    src/filter_image.c
    src/process_image.c
    src/resize_image.c)

target_link_libraries(vision-hw2 m)
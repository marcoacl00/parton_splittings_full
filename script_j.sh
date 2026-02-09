#!/bin/bash

#rm -rf build/*
cmake -S . -B build
cmake --build build

./build/main 0
#./build/main 1
./build/main 2
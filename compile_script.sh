#!/bin/bash
nvcc -c -I /usr/local/cuda/include main.cu
gcc -o a.out main.o -L /usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lstdc++

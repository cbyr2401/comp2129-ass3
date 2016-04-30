#!/usr/bin/env bash
gcc -Wall -Werror -std=gnu11 -pthread main.c matrix.c -o matrix
./matrix 2 1
#-fsanitize=address
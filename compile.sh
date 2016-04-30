#!/usr/bin/env bash
gcc -Wall -Werror -std=gnu11 -pthread main.c matrix.c -o matrix
#-fsanitize=address
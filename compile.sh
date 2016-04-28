#!/usr/bin/env bash
gcc -Wall -Werror -std=gnull matrix.c -o matrix
#-fsanitize=address
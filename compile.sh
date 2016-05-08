#!/usr/bin/env bash
gcc -Wall -Werror -DHAVE_STRUCT_TIMESPEC -D_GNU_SOURCE=1 -std=gnu11 -pthread main.c matrix.c -o matrix -lm
#-fsanitize=address
#pragma once

#ifdef __CUDACC__
#define CUDF_HOST_DEVICE __host__ __device__
#else
#define CUDF_HOST_DEVICE
#endif

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>

CUDF_HOST_DEVICE int h_atoi(const char *src) {
    int s = 0;
    bool isMinus = false;

    while (*src == ' ') {
        src++;
    }

    if (*src == '+' || *src == '-') {
        if (*src == '-') {
            isMinus = true;
        }
        src++;
    } else if (*src < '0' || *src > '9') {
        s = 2147483647;
        return s;
    }

    while (*src != '\0' && *src >= '0' && *src <= '9') {
        s = s * 10 + *src - '0';
        src++;
    }
    return s * (isMinus ? -1 : 1);
}

CUDF_HOST_DEVICE int h_itoa(int n, char s[]) {
    int i, j, sign;
    sign = n;
    if (sign < 0) {
        n = -n;
    }
    i = 0;
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    if (sign < 0) {
        s[i] = '-';
    }
    for (j = 0; j < (i + 1) / 2; j++) {
        char tmp = s[j];
        s[j] = s[i - j];
        s[i - j] = tmp;
    }
    s[i + 1] = '\0';
    return 0;
}
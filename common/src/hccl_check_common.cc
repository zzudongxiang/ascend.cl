#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <climits>
#include <sys/wait.h>
#include "hccl_test_common.h"
#include <sys/syscall.h>
#include "hccl_check_common.h"

int check_buf_result_float(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0; // j = 0;
    // int n = 0;
    int err = 0;
    float *c_buf = (float *)check_buf;
    float *result = (float *)result_buf;
    u64 first_err_pos = ULLONG_MAX;
    for (i = 0; i < count; ++i) {
        if (fabsf(c_buf[i] - result[i]) > HCCL_EPSION_FLOAT) {
            if (fabsf(result[i]) > 0) {
                if (fabsf(fabsf(c_buf[i] - result[i]) / result[i]) > (HCCL_EPSION_FLOAT * 100)) {
                    if (first_err_pos == ULLONG_MAX) {
                        first_err_pos = i;
                    }
                    err++;
                }
            } else {
                if (fabsf(c_buf[i] - result[i]) > 0.001) {
                    if (first_err_pos == ULLONG_MAX) {
                        first_err_pos = i;
                    }
                    err++;
                }
            }
        }
    }

    if (err > 0) {
        printf("check buf[%llu] error, exp:%f, act:%f \n", first_err_pos, c_buf[first_err_pos], result[first_err_pos]);
    }
    if (err > 0) {
        printf("total err is %d\n", err);
    }
    return err;
}

int check_buf_result_int8(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0;
    s8 *c_buf = (s8 *)check_buf;
    s8 *result = (s8 *)result_buf;
    int err = 0;
    u64 first_err_pos = ULLONG_MAX;
    for (i = 0; i < count; ++i) {
        if (c_buf[i] != result[i]) {
            if (first_err_pos == ULLONG_MAX) {
                first_err_pos = i;
            }
            err++;
        }
    }

    if (err > 0) {
        printf("result buf[%llu] is not right,exp: %d, act:%d \n", first_err_pos, c_buf[first_err_pos],
            result[first_err_pos]);
    }
    if (err > 0) {
        printf("total err is %d\n", err);
    }
    return err;
}

int check_buf_result_half(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0;
    u16 *result = (u16 *)result_buf;
    u16 *s = (u16 *)check_buf;
    int err = 0;
    u64 first_err_pos = ULLONG_MAX;

    for (i = 0; i < count; ++i) {
        if (s[i] != result[i]) {
            if (first_err_pos == ULLONG_MAX) {
                first_err_pos = i;
            }
            err++;
        }
    }

    if (err > 0) {
        printf("result buf[%llu] is not right,exp: %u, act:%u \n", first_err_pos, s[first_err_pos],
            result[first_err_pos]);
    }
    if (err > 0) {
        printf("total err is %d\n", err);
    }
    return err;
}

int check_buf_result_int32(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0;
    int *c_buf = (int *)check_buf;
    int *result = (int *)result_buf;
    int err = 0;
    u64 first_err_pos = ULLONG_MAX;
    for (i = 0; i < count; ++i) {
        if (c_buf[i] != result[i]) {
            if (first_err_pos == ULLONG_MAX) {
                first_err_pos = i;
            }
            err++;
        }
    }

    if (err > 0) {
        printf("result buf[%llu] is not right,exp: %d, act:%d \n", first_err_pos, c_buf[first_err_pos],
            result[first_err_pos]);
    }
    if (err > 0) {
        printf("total err is %d\n", err);
    }
    return err;
}

int check_buf_result_int64(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0;
    s64 *c_buf = (s64 *)check_buf;
    s64 *result = (s64 *)result_buf;
    int err = 0;
    u64 first_err_pos = ULLONG_MAX;
    for (i = 0; i < count; ++i) {
        if (c_buf[i] != result[i]) {
            if (first_err_pos == ULLONG_MAX) {
                first_err_pos = i;
            }
            err++;
        }
    }

    if (err > 0) {
        printf("result buf[%llu] is not right,exp: %lld, act:%lld \n", first_err_pos, c_buf[first_err_pos],
            result[first_err_pos]);
    }
    if (err > 0) {
        printf("total err is %d\n", err);
    }

    return err;
}

int check_buf_result_u64(const void *result_buf, const void *check_buf, u64 count)
{
    u64 i = 0;
    u64 *c_buf = (u64 *)check_buf;
    u64 *result = (u64 *)result_buf;
    int err = 0;
    u64 first_err_pos = ULLONG_MAX;
    for (i = 0; i < count; ++i) {
        if (c_buf[i] != result[i]) {
            if (first_err_pos == ULLONG_MAX) {
                first_err_pos = i;
            }
            err++;
        }
    }

    if (err > 0) {
        printf("result buf[%llu] is not right,exp: %llu, act:%llu \n", first_err_pos, c_buf[first_err_pos],
            result[first_err_pos]);
    }

    if (err > 0) {
        printf("total err is %d\n", err);
    }

    return err;
}
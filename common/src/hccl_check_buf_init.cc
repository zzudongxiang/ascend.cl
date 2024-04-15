#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <climits>
#include <sys/wait.h>
#include "hccl_test_common.h"
#include <sys/syscall.h>
#include "hccl_check_buf_init.h"
#include <map>

void host_buf_init_fp32(void* dst_buf, u64 count, int val)
{
    float* f_tmp = NULL;
    f_tmp = (float*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        f_tmp[j] = val;
    }
    return;
}

void host_buf_init_int8(void* dst_buf, u64 count, int val)
{
    char* tmp = NULL;
    tmp = (char*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        tmp[j] = val % 128;
    }
    return;
}

void host_buf_init_int32(void* dst_buf, u64 count, int val)
{
    int* t_tmp = NULL;
    t_tmp = (int*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        t_tmp[j] = val;
    }
    return;
}

void host_buf_init_fp16(void* dst_buf, u64 count, int val)
{
    unsigned short* f16_temp = NULL;
    f16_temp = (u16*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        f16_temp[j] = fp16_ieee_from_fp32_value(val);
    }
    return;
}

void host_buf_init_int16(void* dst_buf, u64 count, int val)
{
    short* s16_tmp = NULL;
    s16_tmp = (s16*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        s16_tmp[j] = val;
    }
    return;
}

void host_buf_init_int64(void* dst_buf, u64 count, int val)
{
    int64_t* tmp_buf = NULL;
    tmp_buf = (int64_t*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        tmp_buf[j] = val;
    }
    return;
}

void host_buf_init_bfp16(void* dst_buf, u64 count, int val)
{
    unsigned short *f16_temp = (u16*)dst_buf;
    for(u64 j = 0; j < count; ++j)
    {
        f16_temp[j] = fp32tobf16(val);  //fp32转换bf16
    }
    return;
}

void memory_dump(void *ptr, int len) {
    int i;

    for (i = 0; i < len; i++) {
        if (i % 8 == 0 && i != 0)
            printf(" ");
        if (i % 16 == 0 && i != 0)
            printf("\n");
        printf("%02x ", *((uint8_t *)ptr + i));
    }
    printf("\n");
}

void hccl_host_buf_init(void* dst_buf, u64 count, int dtype, int val)
{
    if(functionMap.find(dtype) != functionMap.end())
    {
        functionMap[dtype](dst_buf, count, val);
    }
    memory_dump(dst_buf, 512);
    return;
}

void reduce_check_buf_init_fp32(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    float* f_tmp = NULL;
    f_tmp = (float*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f_tmp[j] = val * rank_size;
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f_tmp[j] = pow(val, rank_size);
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f_tmp[j] = val;
        }
    }
    return;
}

void reduce_check_buf_init_int8(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    char* tmp = NULL;
    tmp = (char*)dst_buf;
    int n = 0;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            n = (val % 128) * rank_size;
            if(n > 127)
            {
                n = 127;
            }
            tmp[j] = n;//大于128取127
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            n = ((int)pow(val % 128, rank_size));//大于128取127
            if(n > 127)
            {
                n = 127;
            }
            tmp[j] = n;//大于128取127
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            if(val > 127)
            {
                val = 127;
            }
            tmp[j] = val % 128;
        }
    }
    return;
}

void reduce_check_buf_init_int32(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    int*  t_tmp = NULL;
    t_tmp = (int*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            t_tmp[j] = val * rank_size;
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            t_tmp[j] = pow(val, rank_size);
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            t_tmp[j] = val;
        }
    }
    return;
}

void reduce_check_buf_init_fp16(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    u16* f16_temp = NULL;
    f16_temp = (u16*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp16_ieee_from_fp32_value(val * rank_size);
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp16_ieee_from_fp32_value(pow(val, rank_size));
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp16_ieee_from_fp32_value(val);
        }
    }
    return;
}

void reduce_check_buf_init_int16(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    s16* s16_temp = NULL;
    s16_temp = (s16*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            s16_temp[j] = val * rank_size;
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            s16_temp[j] = pow(val, rank_size);
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            s16_temp[j] = val;
        }
    }
    return;
}

void reduce_check_buf_init_int64(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    int64_t* temp = nullptr;
    temp = (int64_t*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            temp[j] = val * rank_size;
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            temp[j] = pow(val, rank_size);
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            temp[j] = val;
        }
    }
    return;
}

void reduce_check_buf_init_bfp16(void* dst_buf, u64 count, int val, int op, int rank_size)
{
    unsigned short *f16_temp = (u16*)dst_buf;
    if(op == HCCL_REDUCE_SUM)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp32tobf16(val * rank_size); 
        }
    }
    else if(op == HCCL_REDUCE_PROD)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp32tobf16(pow(val, rank_size));
        }
    }
    else if(op == HCCL_REDUCE_MIN || op == HCCL_REDUCE_MAX)
    {
        for(u64 j = 0; j < count; ++j)
        {
            f16_temp[j] = fp32tobf16(val);
        }
    }
    return;
}

void hccl_reduce_check_buf_init(void *dst_buf, u64 count, int dtype, int op, int val, int rank_size)
{
    if(functionReduceMap.find(dtype) != functionReduceMap.end())
    {
        functionReduceMap[dtype](dst_buf, count, val, op, rank_size);
    }
    return;
}


int alltoall_check_result_uint64(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    u64 *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        u64 check_val = i + 1;
        result = (u64 *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%llu, act:%llu\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_fp32(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    float *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        float check_val = i + 1;
        result = (float *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%f, act:%f\n", i, j, check_val, result[j]);
                ret++;
            }
                
        }
    }
    return ret;
}

int alltoall_check_result_int8(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    char *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        char check_val = i + 1;
        result = (char *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_int32(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    int *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        int check_val = i + 1;
        result = (int *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_int64(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    long long *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        long long check_val = i + 1;
        result = (long long *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%lld, act:%lld\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_int16(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    short *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        short check_val = i + 1;
        result = (short *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_fp16(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    u16 *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        float val = i + 1;
        u16 check_val = fp16_ieee_from_fp32_value(val);
        result = (u16 *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_uint8(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    uint8_t *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        uint8_t check_val = i + 1;
        result = (uint8_t *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_uint16(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    uint16_t *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        uint16_t check_val = i + 1;
        result = (uint16_t *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_uint32(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    uint32_t *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        uint32_t check_val = i + 1;
        result = (uint32_t *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(result[j] != check_val)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int alltoall_check_result_bfp16(const void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_size, int dtype)
{
    int ret = 0;
    u16 *result = NULL;
    for(int i = 0; i < rank_size; ++i)
    {
        u16 check_val = i + 1;
        result = (u16 *)check_buf + recv_disp[i];
        for(u64 j = 0; j < recv_counts[i]; ++j)
        {
            if(fabs(result[j] - check_val) / abs(result[j]) > 0.001)
            {
                printf("check data from rank %d  result[%llu] error, exp:%d, act:%d\n", i, j, check_val, result[j]);
                ret++;
            }
        }
    }
    return ret;
}

int hccl_alltoallv_check_result(void *check_buf, u64 *recv_counts, u64 *recv_disp, int rank_id, int rank_size, int dtype)
{
    int ret = 0;
    if(rank_size < 1)   //接收数据为0则不进行数据校验
    {
        return ret;
    }

    if(functionAllToAllMap.find(dtype) != functionAllToAllMap.end())
    {
        ret = functionAllToAllMap[dtype](check_buf, recv_counts, recv_disp, rank_size, dtype);
    }
    return ret;
}

std::map<int,HostBufInitFunc> functionMap = {
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_FP32, host_buf_init_fp32),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_INT8, host_buf_init_int8),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_UINT8, host_buf_init_int8),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_INT32, host_buf_init_int32),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_UINT32, host_buf_init_int32),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_FP16, host_buf_init_fp16),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_INT16, host_buf_init_int16),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_UINT16, host_buf_init_int16),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_INT64, host_buf_init_int64),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_FP64, host_buf_init_int64),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_UINT64, host_buf_init_int64),
    std::pair<int,HostBufInitFunc>(HCCL_DATA_TYPE_BFP16, host_buf_init_bfp16)
};

std::map<int,ReduceCheckBufInitFunc> functionReduceMap = {
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_FP32, reduce_check_buf_init_fp32),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_INT8, reduce_check_buf_init_int8),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_INT32, reduce_check_buf_init_int32),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_FP16, reduce_check_buf_init_fp16),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_INT16, reduce_check_buf_init_int16),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_INT64, reduce_check_buf_init_int64),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_UINT64, reduce_check_buf_init_int64),
    std::pair<int,ReduceCheckBufInitFunc>(HCCL_DATA_TYPE_BFP16, reduce_check_buf_init_bfp16)
};

std::map<int,AllToAllCheckResult> functionAllToAllMap = {
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_UINT64, alltoall_check_result_uint64),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_FP32, alltoall_check_result_fp32),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_INT8, alltoall_check_result_int8),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_INT32, alltoall_check_result_int32),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_INT64, alltoall_check_result_int64),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_INT16, alltoall_check_result_int16),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_FP16, alltoall_check_result_fp16),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_UINT8, alltoall_check_result_uint8),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_UINT16, alltoall_check_result_uint16),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_UINT32, alltoall_check_result_uint32),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_FP64, alltoall_check_result_int64),
    std::pair<int,AllToAllCheckResult>(HCCL_DATA_TYPE_BFP16, alltoall_check_result_bfp16)
};

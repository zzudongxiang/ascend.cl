#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_opbase_rootinfo_base.h"

namespace hccl
{
HcclOpBaseTest::HcclOpBaseTest()
{
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    check_buf = nullptr;
    send_buff = nullptr;
    recv_buff = nullptr;
}

HcclOpBaseTest::~HcclOpBaseTest()
{

}

int HcclOpBaseTest::hccl_op_base_test()
{
    return 0;
}

void HcclOpBaseTest::init_data_count()
{
    switch(dtype)
    {
        case HCCL_DATA_TYPE_FP32:
            data->count = (data->data_size + sizeof(float) - 1)/sizeof(float); //count向上取整
            data->type_size = sizeof(float);
            break;
        case HCCL_DATA_TYPE_INT32:
            data->count = (data->data_size + sizeof(int) - 1)/sizeof(int);
            data->type_size = sizeof(int);
            break;
        case HCCL_DATA_TYPE_BFP16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
            data->count = (data->data_size + sizeof(short) - 1)/sizeof(short);
            data->type_size = sizeof(short);
            break;
        case HCCL_DATA_TYPE_INT8:
            data->count = (data->data_size + sizeof(signed char) - 1)/sizeof(signed char);
            data->type_size = sizeof(signed char);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_FP64:
            data->count = (data->data_size + sizeof(long long) - 1)/sizeof(long long);
            data->type_size = sizeof(long long);
            break;
        case HCCL_DATA_TYPE_UINT64:
            data->count = (data->data_size + sizeof(unsigned long long) - 1)/sizeof(unsigned long long);
            data->type_size = sizeof(unsigned long long);
            break;
        case HCCL_DATA_TYPE_UINT8:
            data->count = (data->data_size + sizeof(unsigned char) - 1)/sizeof(unsigned char);
            data->type_size = sizeof(unsigned char);
            break;
        case HCCL_DATA_TYPE_UINT16:
            data->count = (data->data_size + sizeof(unsigned short) - 1)/sizeof(unsigned short);
            data->type_size = sizeof(unsigned short);
            break;
        case HCCL_DATA_TYPE_UINT32:
            data->count = (data->data_size + sizeof(unsigned int) - 1)/sizeof(unsigned int);
            data->type_size = sizeof(unsigned int);
            break;
        default:
            data->count = (data->data_size + sizeof(float) - 1)/sizeof(float);
            data->type_size = sizeof(float);
            break;
    }
    return;
}

int HcclOpBaseTest::init_buf_val()
{
    return 0;
}

int HcclOpBaseTest::check_buf_result()
{
    return 0;
}

void HcclOpBaseTest::no_verification()
{
    check = 0; //不进行校验
    if (rank_id == root_rank && print_dump) {
        printf("Warning: The calculation result overflows, No verification is performed.\n");
        print_dump = false;
    }
    return;
}

void HcclOpBaseTest::is_data_overflow()
{
    if (op_type == HCCL_REDUCE_PROD) {
        if (dtype == HCCL_DATA_TYPE_FP16 && rank_size >= 16) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_FP32 && rank_size >= 128) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_INT8 && rank_size >= 7) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_INT32 && rank_size >= 31) {
            no_verification();
        }
    } else if (op_type == HCCL_REDUCE_SUM) {
        if(dtype == HCCL_DATA_TYPE_INT8 && rank_size >= 63) {
            no_verification();
        }
    }

    return;
}

void HcclOpBaseTest::print_execution_time(double average_time_us, double algorithm_bandwith_GBytes_s)
{
    //不开启结果校验场景
    if (check == 0)
    {
        if (rank_id == root_rank) {
            if (print_header)
            {
                printf("%-15s | %-12s | %-18s | %s\n", data_size, aveg_time, alg_bandwidth, verification_result);
                print_header = false;
            }
            printf("%-17llu | %-14.2f | %-20.5f | NULL\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        }
        return;
    }

    // 开启结果校验，部分rank结果校验失败场景
    bool check_result[rank_size];
    if (check_err != 0)
    {
        check_result[rank_id] = false; // 结果校验失败
        printf("rank id %d, check result failed\n", rank_id);
    } else {
        check_result[rank_id] = true; // 结果校验成功
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, check_result, sizeof(bool), MPI_BYTE, MPI_COMM_WORLD);

    bool result = true;
    for (int p = 0; p < rank_size; p++)
    {
        if (check_result[p] == false)
        {
            result = false;
            break;
        }
    }
    if (rank_id == root_rank)
    {
        if (print_header)
        {
            printf("%-15s | %-12s | %-18s | %s\n", data_size, aveg_time, alg_bandwidth, verification_result);
            print_header = false;
        }

        if (!result)
        {
            printf("%-17llu | %-14.2f | %-20.5f | failed\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        } else {
            printf("%-17llu | %-14.2f | %-20.5f | success\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        }
    }
    return;
}


int HcclOpBaseTest::destory_check_buf()
{
    return 0;
}
}
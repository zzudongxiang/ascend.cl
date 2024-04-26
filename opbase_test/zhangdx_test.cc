#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_allgather_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"
using namespace hccl;

HcclTest* init_opbase_ptr(HcclTest* opbase)
{
    opbase = new hccl::HcclOpBaseAllgatherTest();

    return opbase;
}

void delete_opbase_ptr(HcclTest* opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl
{
HcclOpBaseAllgatherTest::HcclOpBaseAllgatherTest() : HcclOpBaseTest()
{
    
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    check_buf = nullptr;
    send_buff = nullptr;
    recv_buff = nullptr;
}

HcclOpBaseAllgatherTest::~HcclOpBaseAllgatherTest()
{

}

int HcclOpBaseAllgatherTest::init_buf_val()
{
    //初始化校验内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize * rank_size));
    hccl_host_buf_init((char*)check_buf, data->count * rank_size, dtype, val);
    return 0;
}

int HcclOpBaseAllgatherTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&recv_buff_temp, malloc_kSize * rank_size));
    ACLCHECK(aclrtMemcpy((void*)recv_buff_temp, malloc_kSize * rank_size, (void*)recv_buff, malloc_kSize * rank_size, ACL_MEMCPY_DEVICE_TO_HOST));
    int ret = 0;
    switch(dtype)
    {
        case HCCL_DATA_TYPE_FP32:
            ret = check_buf_result_float((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
            ret = check_buf_result_int8((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
            ret = check_buf_result_int32((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            ret = check_buf_result_half((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_FP64:
            ret = check_buf_result_int64((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        case HCCL_DATA_TYPE_UINT64:
            ret = check_buf_result_u64((char*)recv_buff_temp, (char*)check_buf, data->count * rank_size);
            break;
        default:
            ret++;
            printf("no match datatype\n");
            break;
    }
    if(ret != 0)
    {
        check_err++;
    }

    return 0;
}

void HcclOpBaseAllgatherTest::cal_execution_time(float time)
{
    double total_time_us              = time * 1000;
    double average_time_us            = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize * rank_size / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

int HcclOpBaseAllgatherTest::destory_check_buf()
{
    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFreeHost(recv_buff_temp));
    ACLCHECK(aclrtFreeHost(check_buf));
    return 0;
}

int HcclOpBaseAllgatherTest::hccl_op_base_test() //主函数
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }
    
    // 获取数据量和数据类型
    init_data_count();

    data->count = (data->count + rank_size - 1) / rank_size;
    malloc_kSize = data->count * data->type_size;

    //申请集合通信操作的内存
    ACLCHECK(aclrtMalloc((void**)&send_buff, malloc_kSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc((void**)&recv_buff, malloc_kSize * rank_size, ACL_MEM_MALLOC_HUGE_FIRST));

    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init((char*)host_buf, data->count, dtype, val);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    DUMP_DEBUG("zhangdx", "host_init", rank_id, 
        host_buf, malloc_kSize, 
        send_buff, malloc_kSize, data->count,
        recv_buff, malloc_kSize * rank_size, data->count);

    // 准备校验内存
    if (check == 1) {
        ACLCHECK(init_buf_val());
    }

    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAllGather((void *)send_buff, (void*)recv_buff, data->count, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAllGather((void *)send_buff, (void*)recv_buff, data->count, (HcclDataType)dtype, hccl_comm, stream));
    }
    //等待stream中集合通信任务执行完成
    ACLCHECK(aclrtRecordEvent(end_event, stream));

    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    if (check == 1) {
        ACLCHECK(check_buf_result()); // 校验计算结果
    }

    cal_execution_time(time);

    ACLCHECK(aclrtMallocHost((void**)&recv_buff_temp, malloc_kSize * rank_size));
    ACLCHECK(aclrtMemcpy((void*)recv_buff_temp, malloc_kSize * rank_size, (void*)recv_buff, malloc_kSize * rank_size, ACL_MEMCPY_DEVICE_TO_HOST));
    DUMP_DEBUG("zhangdx", "host_done", rank_id, 
        recv_buff_temp, malloc_kSize * rank_size, 
        send_buff, malloc_kSize, data->count,
        recv_buff, malloc_kSize * rank_size, data->count);

    //销毁集合通信内存资源
    ACLCHECK(aclrtFree(send_buff));
    ACLCHECK(aclrtFree(recv_buff));
    if (check == 1) {
        ACLCHECK(destory_check_buf());
    }
    return 0;
}
}
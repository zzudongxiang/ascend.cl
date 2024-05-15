#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_alltoall_rootinfo_test.h"
#include "hccl_rootinfo_base.h"
#include "hccl_check_buf_init.h"
using namespace hccl;

HcclTest* init_opbase_ptr(HcclTest* opbase)
{
    opbase = new HcclOpBaseAlltoallTest();

    return opbase;
}

void delete_opbase_ptr(HcclTest* opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl {
HcclOpBaseAlltoallTest::HcclOpBaseAlltoallTest()
    : HcclOpBaseTest(),
      sendCount_(0),
      recvCount_(0)
{
}

HcclOpBaseAlltoallTest::~HcclOpBaseAlltoallTest()
{
}

int HcclOpBaseAlltoallTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)check_buf, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));

    u64* recv_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    u64* recv_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    for(int i = 0; i < rank_size; ++i)
    {
        recv_counts[i] = data->count / rank_size;
        recv_disp[i] = i * data->count / rank_size;
    }

    int ret = 0;
    ret = hccl_alltoallv_check_result(check_buf, recv_counts, recv_disp, rank_id, rank_size, dtype);
    if(ret != 0)
    {
        check_err++;
    }
    free(recv_counts);
    free(recv_disp);
    return 0;
}

void HcclOpBaseAlltoallTest::cal_execution_time(float time)
{
    double total_time_us = time * 1000;
    double average_time_us = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

void HcclOpBaseAlltoallTest::free_send_recv_buf()
{
}

int HcclOpBaseAlltoallTest::destory_check_buf()
{
    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFreeHost(check_buf));
    return 0;
}

int HcclOpBaseAlltoallTest::hccl_op_base_test() //主函数
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }
    // 获取数据量和数据类型
    init_data_count();
    data->count = (data->count + rank_size - 1) / rank_size * rank_size;
    malloc_kSize = data->count * data->type_size;

    //申请集合通信操作的内存
    ACLCHECK(aclrtMalloc((void**)&send_buff, malloc_kSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc((void**)&recv_buff, malloc_kSize, ACL_MEM_MALLOC_HUGE_FIRST));

    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init((char*)host_buf, data->count, dtype, rank_id + 1);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    sendCount_ = data->count / rank_size;
    recvCount_ = data->count / rank_size;

    DUMP_INIT("alltoall", rank_id,
        host_buf, malloc_kSize, 
        send_buff, malloc_kSize, sendCount_,
        recv_buff, malloc_kSize, recvCount_);

    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAlltoAll((void *)send_buff, sendCount_, (HcclDataType)dtype,\
            (void*)recv_buff, recvCount_, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAlltoAll((void *)send_buff, sendCount_, (HcclDataType)dtype,\
            (void*)recv_buff, recvCount_, (HcclDataType)dtype, hccl_comm, stream));
    }
    //等待stream中集合通信任务执行完成
    ACLCHECK(aclrtRecordEvent(end_event, stream));

    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    // 校验计算结果
    if (check == 1) {
        ACLCHECK(check_buf_result());
    } else {
        ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
        ACLCHECK(aclrtMemcpy((void*)check_buf, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));
        DUMP_DONE("alltoall", rank_id, host_buf,
            check_buf, malloc_kSize, 
            send_buff, malloc_kSize, sendCount_,
            recv_buff, malloc_kSize, recvCount_);
    }

    cal_execution_time(time);

    //销毁集合通信内存资源
    ACLCHECK(aclrtFree(send_buff));
    ACLCHECK(aclrtFree(recv_buff));
    free_send_recv_buf();
    if (check == 1) {
        ACLCHECK(destory_check_buf());
    }
    return 0;
}
}
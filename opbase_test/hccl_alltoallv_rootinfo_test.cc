#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_alltoallv_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"
using namespace hccl;

HcclTest* init_opbase_ptr(HcclTest* opbase)
{
    opbase = new HcclOpBaseAlltoallvTest();

    return opbase;
}

void delete_opbase_ptr(HcclTest* opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl {
HcclOpBaseAlltoallvTest::HcclOpBaseAlltoallvTest() : HcclOpBaseTest()
{
    
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    send_buff = nullptr;
    send_counts = nullptr;
    send_disp = nullptr;
    recv_buff = nullptr;
    recv_counts = nullptr;
    recv_disp = nullptr;
}

HcclOpBaseAlltoallvTest::~HcclOpBaseAlltoallvTest()
{
}

void HcclOpBaseAlltoallvTest::malloc_send_recv_buf()
{
    send_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    send_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    for(int i = 0; i < rank_size; ++i)
    {
        send_counts[i] = data->count / rank_size;
        send_disp[i] = i * data->count / rank_size;
    }

    recv_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    recv_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    for(int i = 0; i < rank_size; ++i)
    {
        recv_counts[i] = data->count / rank_size;
        recv_disp[i] = i * data->count / rank_size;
    }
    return;
}

int HcclOpBaseAlltoallvTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)check_buf, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));

    int ret = 0;
    ret = hccl_alltoallv_check_result(check_buf, recv_counts, recv_disp, rank_id, rank_size, dtype);
    if(ret != 0)
    {
        check_err++;
    }

    // dump检查的内存
    char bin_path[MAX_PATH_LEN];
    memset(bin_path, 0, MAX_PATH_LEN);
    sprintf(bin_path, "/root/Workdir/hccl_test/log/alltoallv_check_rank_%d.bin", rank_id);
    printf("rank_id: %d, start ptr: %p, len: %llu, log_path: %s\r\n", rank_id, check_buf, (long long unsigned int)malloc_kSize, bin_path);
    mem_dump_file((char*)check_buf, malloc_kSize, bin_path);

    return 0;
}

void HcclOpBaseAlltoallvTest::cal_execution_time(float time)
{
    double total_time_us = time * 1000;
    double average_time_us = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

void HcclOpBaseAlltoallvTest::free_send_recv_buf()
{
    free(send_counts);
    free(send_disp);
    free(recv_counts);
    free(recv_disp);
}

int HcclOpBaseAlltoallvTest::destory_check_buf()
{
    ACLCHECK(aclrtFreeHost(host_buf));
    ACLCHECK(aclrtFreeHost(check_buf));
    return 0;
}

int HcclOpBaseAlltoallvTest::hccl_op_base_test() //主函数
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

    //申请sendcounts和send_disp
    malloc_send_recv_buf();

    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init((char*)host_buf, data->count, dtype, rank_id + 1);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // dump初始化的内存
    char bin_path[MAX_PATH_LEN];
    memset(bin_path, 0, MAX_PATH_LEN);
    sprintf(bin_path, "/root/Workdir/hccl_test/log/alltoallv_init_rank_%d.bin", rank_id);
    printf("rank_id: %d, start ptr: %p, len: %llu, log_path: %s\r\n", rank_id, host_buf, (long long unsigned int)malloc_kSize, bin_path);
    mem_dump_file((char*)host_buf, malloc_kSize, bin_path);

    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAlltoAllV((void *)send_buff, send_counts, send_disp, (HcclDataType)dtype,\
            (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAlltoAllV((void *)send_buff, send_counts, send_disp, (HcclDataType)dtype,\
            (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
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
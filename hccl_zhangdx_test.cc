#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_allreduce_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"
using namespace hccl;

HcclTest *init_opbase_ptr(HcclTest *opbase)
{
    opbase = new HcclOpBaseAllreduceTest();
    return opbase;
}

void delete_opbase_ptr(HcclTest *opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}



namespace hccl
{

    HcclOpBaseAllreduceTest::HcclOpBaseAllreduceTest() : HcclOpBaseTest()
    {
        host_buf = nullptr;
        recv_buff_temp = nullptr;
        check_buf = nullptr;
        send_buff = nullptr;
        recv_buff = nullptr;
    }

    HcclOpBaseAllreduceTest::~HcclOpBaseAllreduceTest()
    {
    }

    int HcclOpBaseAllreduceTest::init_buf_val()
    {
        // 初始化校验内存
        ACLCHECK(aclrtMallocHost((void **)&check_buf, malloc_kSize));
        hccl_reduce_check_buf_init((char *)check_buf, data->count, dtype, op_type, val, rank_size);
        return 0;
    }

    int HcclOpBaseAllreduceTest::check_buf_result()
    {
        // 获取输出内存
        ACLCHECK(aclrtMallocHost((void **)&recv_buff_temp, malloc_kSize));
        ACLCHECK(aclrtMemcpy((void *)recv_buff_temp, malloc_kSize, (void *)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));
        int ret = 0;
        switch (dtype)
        {
        case HCCL_DATA_TYPE_FP32:
            ret = check_buf_result_float((char *)recv_buff_temp, (char *)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT8:
            ret = check_buf_result_int8((char *)recv_buff_temp, (char *)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT32:
            ret = check_buf_result_int32((char *)recv_buff_temp, (char *)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_BFP16:
            ret = check_buf_result_half((char *)recv_buff_temp, (char *)check_buf, data->count);
            break;
        case HCCL_DATA_TYPE_INT64:
            ret = check_buf_result_int64((char *)recv_buff_temp, (char *)check_buf, data->count);
            break;
        default:
            ret++;
            printf("no match datatype\n");
            break;
        }
        if (ret != 0)
        {
            check_err++;
        }
        return 0;
    }

    void HcclOpBaseAllreduceTest::cal_execution_time(float time)
    {
        double total_time_us = time * 1000;
        double average_time_us = total_time_us / iters;
        double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;
        print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
        return;
    }

    int HcclOpBaseAllreduceTest::destory_check_buf()
    {
        ACLCHECK(aclrtFreeHost(host_buf));
        ACLCHECK(aclrtFreeHost(recv_buff_temp));
        ACLCHECK(aclrtFreeHost(check_buf));
        return 0;
    }

    int HcclOpBaseAllreduceTest::hccl_op_base_test() // 主函数
    {
        // 获取数据量和数据类型
        init_data_count();
        malloc_kSize = data->count * data->type_size;
        // 申请集合通信操作的内存
        ACLCHECK(aclrtMalloc((void **)&send_buff, malloc_kSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACLCHECK(aclrtMalloc((void **)&recv_buff, malloc_kSize, ACL_MEM_MALLOC_HUGE_FIRST));
        
        is_data_overflow();

        // 初始化输入内存
        ACLCHECK(aclrtMallocHost((void **)&host_buf, malloc_kSize));
        hccl_host_buf_init((char *)host_buf, data->count, dtype, val);
        ACLCHECK(aclrtMemcpy((void *)send_buff, malloc_kSize, (void *)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

        // 准备校验内存
        if (check == 1)
        {
            ACLCHECK(init_buf_val());
        }

        // 执行集合通信操作
        for (int j = 0; j < warmup_iters; ++j)
        {
            HCCLCHECK(HcclAllReduce((void *)send_buff, (void *)recv_buff, data->count, (HcclDataType)dtype, (HcclReduceOp)op_type, hccl_comm, stream));
        }

        ACLCHECK(aclrtRecordEvent(start_event, stream));

        for (int i = 0; i < iters; ++i)
        {
            HCCLCHECK(HcclAllReduce((void *)send_buff, (void *)recv_buff, data->count, (HcclDataType)dtype, (HcclReduceOp)op_type, hccl_comm, stream));
        }
        // 等待stream中集合通信任务执行完成
        ACLCHECK(aclrtRecordEvent(end_event, stream));

        ACLCHECK(aclrtSynchronizeStream(stream));

        float time;
        ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

        if (check == 1)
        {
            ACLCHECK(check_buf_result()); // 校验计算结果
        }

        cal_execution_time(time);

        // 销毁集合通信内存资源
        ACLCHECK(aclrtFree(send_buff));
        ACLCHECK(aclrtFree(recv_buff));
        if (check == 1)
        {
            ACLCHECK(destory_check_buf());
        }
        return 0;
    }
}

int main1(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int ret = 0;

    //构造执行器
    HcclTest *hccl_test = nullptr;
    hccl_test = init_opbase_ptr(hccl_test);
    if(hccl_test == nullptr) {
        printf("hccl_test is null\n");
        ret = -1;
        goto hccltesterr3;
    }

    //解析命令行入参
    ret = hccl_test->parse_cmd_line(argc, argv);
    if (ret == 1) {
        //启动--help
        ret = 0;
        goto hccltesterr2;
    } else if(ret == -1) {
        //入参解析失败
        printf("This is an error in parse_cmd_line.\n");
        goto hccltesterr2;
    }

    //查找本host上的所有MPI拉起的进程
    ret = hccl_test->get_mpi_proc();
    if (ret != 0) {
        printf("This is an error in get_mpi_proc.\n");
        goto hccltesterr2;
    }

    //校验命令行参数
    ret = hccl_test->check_cmd_line();
    if (ret != 0) {
        printf("This is an error in check_cmd_line.\n");
        goto hccltesterr2;
    }

    //获取hccltest的环境变量
    ret = hccl_test->get_env_resource();
    if (ret != 0) {
        printf("This is an error in get_env.\n");
        goto hccltesterr1;
    }

    //初始化集合通信域
    ret = hccl_test->init_hcclComm();
    if (ret != 0) {
        printf("This is an error in init_hcclComm.\n");
        goto hccltesterr2;
    }

    //启动测试
    ret = hccl_test->opbase_test_by_data_size(hccl_test);
    if (ret != 0) {
        printf("This is an error in opbase_test_by_data_size.\n");
        goto hccltesterr0;
    }

hccltesterr0:
    //销毁集合通信域
    ret = hccl_test->destory_hcclComm();
    if (ret != 0) {
        printf("This is an error in destory_hcclComm.\n");
    }
hccltesterr1:
    //销毁环境变量申请的资源
    ret = hccl_test->release_env_resource();
    if (ret != 0) {
        printf("This is an error in release_env_resource.\n");
    }
hccltesterr2:
    //删除构造器
    delete_opbase_ptr(hccl_test);
hccltesterr3:
    //释放MPI所用资源
    MPI_Finalize();
    return ret;
}

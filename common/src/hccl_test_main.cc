#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include "hccl_test_common.h"
#include "mpi.h"

using namespace hccl;

extern HcclTest* init_opbase_ptr(HcclTest* opbase);
extern void delete_opbase_ptr(HcclTest* opbase);

int main(int argc, char *argv[])
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

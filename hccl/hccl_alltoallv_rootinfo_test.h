#ifndef __HCCL_ALLTOALLV_ROOTINFO_TEST_H_
#define __HCCL_ALLTOALLV_ROOTINFO_TEST_H_
#include "hccl_test_common.h"
#include "mpi.h"
#include "hccl_check_common.h"
#include "hccl_rootinfo_base.h"
namespace hccl {
class HcclOpBaseAlltoallvTest:public HcclOpBaseTest
{
public:
    HcclOpBaseAlltoallvTest();
    virtual ~HcclOpBaseAlltoallvTest();
    virtual int hccl_op_base_test(); //主函数

private:
    void malloc_send_recv_buf();
    virtual int check_buf_result();//（recv_buf拷贝到recvbufftemp,并且校验正确性）需要调用check_buf_init，校验正确性要调用check_buf_result_float
    void free_send_recv_buf();
    void cal_execution_time(float time);//统计耗时
    virtual int destory_check_buf();//集合通信销毁

    unsigned long long *send_counts;
    unsigned long long *send_disp;
    unsigned long long *recv_counts;
    unsigned long long *recv_disp;
};
}
#endif
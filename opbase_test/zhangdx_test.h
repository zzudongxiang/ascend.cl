#ifndef __HCCL_BROCAST_ROOTINFO_TEST_H_
#define __HCCL_BROCAST_ROOTINFO_TEST_H_
#include "hccl_test_common.h"
#include "mpi.h"
#include "hccl_check_common.h"
#include "hccl_opbase_rootinfo_base.h"
namespace hccl {
class HcclOpBaseBrocastTest:public HcclOpBaseTest
{
public:
    HcclOpBaseBrocastTest();
    virtual ~HcclOpBaseBrocastTest();

    virtual int hccl_op_base_test(); //主函数
private:
    virtual int init_buf_val();  //（初始化host_buf，初始化check_buf，拷贝到send_buf） 其中需要调用hccl_host_buf_init
    virtual int check_buf_result();//（recv_buf拷贝到recvbufftemp,并且校验正确性）需要调用check_buf_init，校验正确性要调用check_buf_result_float
    void cal_execution_time(float time);//统计耗时
    virtual int destory_check_buf();//集合通信销毁
    void *buff;
};
}
#endif
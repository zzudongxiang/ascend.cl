#ifndef __HCCL_P2P_ROOTINFO_TEST_H_
#define __HCCL_P2P_ROOTINFO_TEST_H_

#include "mpi.h"
#include "hccl_test_common.h"
#include "hccl_check_common.h"
#include "hccl_rootinfo_base.h"

namespace hccl
{
    class HcclOpBaseP2PTest : public HcclOpBaseTest
    {
    public:
        HcclOpBaseP2PTest();
        virtual ~HcclOpBaseP2PTest();
        virtual int hccl_op_base_test();

    private:
        virtual int init_buf_val();
        virtual int check_buf_result();
        void cal_execution_time(float time);
        virtual int destory_check_buf();
    };
}

#endif
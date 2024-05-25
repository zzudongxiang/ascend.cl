#
#loading path
#--------------------------------------------------------------------------------------------------------------------------------------------------
CXXFLAGS := -std=c++11                      \
            -Wl,--copy-dt-needed-entries    \
            -fstack-protector-strong        \
            -fPIE -pie                      \
            -O2                             \
            -s                              \
            -Werror                         \
            -Wl,-z,relro                    \
            -Wl,-z,now                      \
            -Wl,-z,noexecstack              \
            # -DMEM_DUMP

Common_DIR = ../common/hccl
Utils_DIR  = ../common/utils
Opbase_DIR = ../hccl
ASCEND_DIR = /usr/local/Ascend/ascend-toolkit/latest

Common_SRC = $(wildcard ${Common_DIR}/*.cc)
Opbase_SRC = $(wildcard ${Opbase_DIR}/*.cc)
Utils_SRC  = $(wildcard ${Utils_DIR}/*.cc)

HCCL_INC_DIR = ${ASCEND_DIR}/include
HCCL_LIB_DIR = ${ASCEND_DIR}/lib64

ACL_INC_DIR = ${ASCEND_DIR}/include
ACL_LIB_DIR = ${ASCEND_DIR}/lib64

MPI_INC_DIR = ${MPI_HOME}/include
MPI_LIB_DIR = ${MPI_HOME}/lib

LIST = all_gather_test all_reduce_test alltoallv_test alltoall_test broadcast_test reduce_scatter_test reduce_test p2p_test

#
#library flags
#--------------------------------------------------------------------------------------------------------------------------------------------------
LIBS = -L$(HCCL_LIB_DIR) -lhccl     \
       -L$(ACL_LIB_DIR)  -lascendcl \
       -L$(MPI_LIB_DIR)  -lmpi

INCLUDEDIRS = -I$(Common_DIR)       \
              -I$(HCCL_INC_DIR)     \
              -I$(ACL_INC_DIR)      \
              -I$(MPI_INC_DIR)      \
              -I$(Utils_DIR)
#
#make
#--------------------------------------------------------------------------------------------------------------------------------------------------
all:
	@mkdir -p ../bin

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_allgather_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o all_gather_test $(LIBS)
	@printf "all_gather_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_allreduce_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o all_reduce_test $(LIBS)
	@printf "all_reduce_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_alltoallv_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o alltoallv_test $(LIBS)
	@printf "all_to_allv_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_alltoall_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o alltoall_test $(LIBS)
	@printf "all_to_all_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_brocast_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o broadcast_test $(LIBS)
	@printf "broad_cast_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_reducescatter_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o reduce_scatter_test $(LIBS)
	@printf "reduce_scatter_test completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_reduce_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o reduce_test $(LIBS)
	@printf "reduce_test compile completed\n\n"

	g++ $(CXXFLAGS) $(Common_SRC) $(Utils_SRC) ${Opbase_DIR}/hccl_p2p_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o p2p_test $(LIBS)
	@printf "p2p_test compile completed\n\n"

	mv $(LIST) ../bin

.PHONY: clean
clean:
	rm -rf ../bin/*_test

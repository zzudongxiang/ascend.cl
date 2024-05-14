#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "mpi.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_allgather_rootinfo_test.h"
#include "hccl_test_common.h"
#include <algorithm>

constexpr s32 HCCL_TEST_REDUCE_RESERVED = 4;
constexpr s32 HCCL_TEST_DATA_TYPE_RESERVED = 12;

HcclReduceOp test_ops[HCCL_TEST_REDUCE_RESERVED] = {HCCL_REDUCE_SUM, HCCL_REDUCE_PROD, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};
const char *test_opnames[HCCL_TEST_REDUCE_RESERVED] = {"sum", "prod", "max", "min"};

HcclDataType test_types[HCCL_TEST_DATA_TYPE_RESERVED] = {HCCL_DATA_TYPE_INT8,    /**< int8 */
                                        HCCL_DATA_TYPE_INT16,   /**< int16 */
                                        HCCL_DATA_TYPE_INT32,   /**< int32 */
                                        HCCL_DATA_TYPE_FP16,    /**< fp16 */
                                        HCCL_DATA_TYPE_FP32,    /**< fp32 */
                                        HCCL_DATA_TYPE_INT64,   /**< int64 */
                                        HCCL_DATA_TYPE_UINT64,  /**< uint64 */
                                        HCCL_DATA_TYPE_UINT8,   /**< uint8 */
                                        HCCL_DATA_TYPE_UINT16,  /**< uint16 */
                                        HCCL_DATA_TYPE_UINT32,  /**< uint32 */
                                        HCCL_DATA_TYPE_FP64,    /**< fp64 */
                                        HCCL_DATA_TYPE_BFP16};
const char *test_typenames[HCCL_TEST_DATA_TYPE_RESERVED] = {"int8", "int16", "int32", "fp16", "fp32", "int64", "uint64", "uint8", "uint16", "uint32", "fp64", "bfp16"};

int get_hccl_op_from_str (char *str) {
    for (int op=0; op < HCCL_TEST_REDUCE_RESERVED; op++) {
      if (strcmp(str, test_opnames[op]) == 0) {
        return op;
      }
    }

    return -1;
}

int get_hccl_dtype_from_str(char *str) {
    for (int t=0; t < HCCL_TEST_DATA_TYPE_RESERVED; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    return -1;
}

static void get_host_name(char* hostName, int maxlen)
{
    gethostname(hostName, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostName[i] == '.') {
            hostName[i] = '\0';
            return;
        }
    }
    return;
}

static uint64_t get_host_hash(const char* string) {
  // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static long parsesize(const char *value) {
    long long int units;
    long size;
    char* size_lit;

    size = strtol(value, &size_lit, 0);
    if (strlen(size_lit) == 1) {
        switch (*size_lit) {
        case 'G':
        case 'g':
            units = 1024*1024*1024;
            break;
        case 'M':
        case 'm':
            units = 1024*1024;
            break;
        case 'K':
        case 'k':
            units = 1024;
            break;
        default:
            return -1;
        }
    } else if (strlen(size_lit) == 0){
        units = 1;
    } else {
        return -1;
    }

    return size * units;
}

u32 sal_str_len(const char *s, u32 maxLen = INT_MAX)
{
    return strnlen(s, maxLen);
}

int is_all_digit(const char *strNum)
{
    // 参数有效性检查
   if (strNum == NULL)
   {
       printf("Error: ptr [%s] is NULL\n", strNum);
       return -1;
   }

    u32 nLength = sal_str_len(strNum);
    for (u32 index = 0; index < nLength; index++) {
        if (!isdigit(strNum[index])) {
            printf("Error:In judge all digit, Check whether the value of [-i -n -r -w -c -p] is a positive integer.\n");
            return -1;
        }
    }
    return 0;
}

long strtol_alldigit(const char *optarg)
{
    long ret = is_all_digit(optarg);
    if (ret != 0) {
        return ret;
    }
    
    return strtol(optarg, NULL, 0);
}

namespace hccl
{
HcclTest::HcclTest()
{
    data = new DataSize;
    data->step_factor = 1;
    proc_rank = 0;
    proc_size = 0;
}

HcclTest::~HcclTest()
{
    delete data;
    data = nullptr;
}

struct option HcclTest::longopts[] =
{
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"root", required_argument, 0, 'r'},
    {"iters", required_argument, 0, 'n'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"check", required_argument, 0, 'c'},
    {"npus", required_argument, 0, 'p'},
    {"help", no_argument, 0, 'h'},
};

void HcclTest::print_help()
{
    printf("USAGE: ./test \n\t"
    "[-b,--minbytes <min size in bytes>] \n\t"
    "[-e,--maxbytes <max size in bytes>] \n\t"
    "[-i,--stepbytes <increment size>] \n\t"
    "[-f,--stepfactor <increment factor>] \n\t"
    "[-n,--iters <iteration count>] \n\t"
    "[-o,--op <sum/prod/min/max>] \n\t"
    "[-d,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64/bfp16>] \n\t"
    "[-r,--root <root>] \n\t"
    "[-w,--warmup_iters <warmup iteration count>] \n\t"
    "[-c,--check <result verification> 0:disabled 1:enabled.] \n\t"
    "[-p,--npus <npus used for one node>] \n\t"
    "[-h,--help]\n");
    return;
}

int HcclTest::check_data_count()
{
    if (data_parsed_begin < 0 || data_parsed_end < 0) {
        printf("invalid size specified for [-b,--minbytes] or [-e,--maxbytes]\n");
        return -1;
    }
    data->min_bytes = (u64)data_parsed_begin;
    data->max_bytes = (u64)data_parsed_end;

    if (stepbytes_flag != 0 && temp_step_bytes < 0) {
        printf("Error: [-i,--stepbytes] must be greater than or equal to 0.\n");
        return -1;
    }

    if (data->max_bytes < data->min_bytes) {
        printf("invalid option: maxbytes < minbytes, Check the [-b,--minbytes] and [-e,--maxbytes] options.\n");
        return -1;
    } else {
        if (stepbytes_flag != 0) {// 用户配置了增量步长
            data->step_bytes = temp_step_bytes;
        } else {// 用户未配置增量步长
            if (data->max_bytes == data->min_bytes) {
                data->step_bytes = 1;// 用户配置数据量的起始值和结束值相同，但未配置增量步长，为防止进入死循环，设置增量步长为1
            }
            if (data->max_bytes > data->min_bytes) {
                data->step_bytes = (data->max_bytes - data->min_bytes)/10;
            }
        }
    }

    if (stepfactor_flag !=0 && data->step_factor <= 1.0) {
        printf("Error: [-f,--stepfactor] Must be greater than 1.0f, Start step mod.\n");
        return -1;
    }

    if (stepfactor_flag !=0 && stepbytes_flag != 0) {
        printf("Warning: [-f,--stepfactor] and [-i,--stepbytes] are set, [-f,--stepfactor] is enabled by default.\n");
    }

    return 0;
}

int HcclTest::check_cmd_line()
{
    int ret = 0;
    ret = check_data_count();
    if (ret != 0)
    {
        return ret;
    }

    if (dtype == -1)
    {
        printf("Error: [-d,--datatype] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (op_type == -1)
    {
        printf("Error: [-o,--op] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (warmup_iters < 0) {
        printf("Error: [-w,--warmup_iters] is invalid, warmup_iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (iters < 0) {
        printf("Error: [-n,--iters] is invalid, iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (hccl_root >= rank_size || hccl_root < 0) //如果指定的root rank大于等于rank_size
    {
        printf("Error: [-r,--root <root>] is invalid, root rank must be greater than or equal to 0 and less than or equal to %d.\n", rank_size - 1);
        return -1;
    }

    if (check != 1 && check != 0)
    {
        printf("Error: [-c,--check] is invalid, check should be 0 or 1\n");
        return -1;
    }

    if (dev_count == 0)
    {
        printf("Error: The number of device is 0.Check whether the package is correct.\n");
        return -1;
    }

    if (npus < 1 || npus > dev_count)
    {
        printf("Error: [-p,--npus <npus used for one node>] is invalid, npus must be greater than or equal to 1 and less than or equal to %d.\n", dev_count);
        return -1;
    }
    return 0;
}

int HcclTest::get_env_resource()
{
    //支持profiling
    const char* profiling_env = getenv("HCCL_TEST_PROFILING");
    if (profiling_env != NULL) {
        profiling_flag = atoi(profiling_env);
        u32 nLength = sal_str_len(profiling_env);
        //校验：入参为字符
        for (u32 index = 0; index < nLength; index++) {
            if (!isdigit(profiling_env[index])) {
                printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
                return -1;
            }
        }
        //校验：入参非0非1
        if (profiling_flag != 0 && profiling_flag != 1) {
            printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
            return -1;
        }
    }

    //开启profiling
    if (profiling_flag == 1) {
        std::string prof_path = "/var/log/npu/profiling";
        const char* profiling_env_path = getenv("HCCL_TEST_PROFILING_PATH");
        if (profiling_env_path != NULL) {
            prof_path = profiling_env_path;
        }
        aclprofInit(prof_path.c_str(), prof_path.size());
        uint32_t profSwitch = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU | ACL_PROF_HCCL_TRACE | ACL_PROF_MSPROFTX | ACL_PROF_RUNTIME_API;
        uint32_t deviceIdList = dev_id;
        int devNum = 1;
        profiling_config = aclprofCreateConfig(&deviceIdList, devNum, ACL_AICORE_PIPE_UTILIZATION, nullptr, profSwitch);
        ACLCHECK(aclprofStart(profiling_config));
    }

    return 0;
}

int HcclTest::release_env_resource()
{
    if (profiling_flag == 1) {
        ACLCHECK(aclprofStop(profiling_config));
        aclprofFinalize();
    }

    return 0;
}

int HcclTest::parse_opt(int opt)
{
    switch(opt) {
        case 'b':
            data_parsed_begin = parsesize(optarg);
            break;
        case 'e':
            data_parsed_end = parsesize(optarg);
            break;
        case 'i':
            stepbytes_flag++;
            temp_step_bytes = strtol_alldigit(optarg);
            break;
        case 'f':
            stepfactor_flag++;
            char *temp;
            data->step_factor = strtof(optarg, &temp);
            break;
        case 'n':
            iters = strtol_alldigit(optarg);
            break;
        case 'o':
            op_flag++;
            op_type = get_hccl_op_from_str(optarg);
            break;
        case 'd':
            dtype = get_hccl_dtype_from_str(optarg);
            break;
        case 'r':
            hccl_root = strtol_alldigit(optarg);
            break;
        case 'w':
            warmup_iters = strtol_alldigit(optarg);
            break;
        case 'c':
            check = strtol_alldigit(optarg);
            break;
        case 'p':
            npus = strtol_alldigit(optarg);
            npus_flag = 1;
            break;
        case 'h':
            print_help();
            return 1;
        default:
            printf("invalid option \n");
            printf("Try [-h --help] for more information.\n");
            return -1;
        }
    return 0;
}

int HcclTest::parse_cmd_line(int argc, char* argv[])
{
    int opt = -1;
    int longindex = 0;
    int ret = 0;
    long parsed;
    while(-1 != (opt = getopt_long(argc, argv, "o:d:b:e:i:f:r:n:w:c:p:h", longopts, &longindex)))
    {
        ret = parse_opt(opt);
        if (ret != 0)
        {
            return ret;
        }
    }

    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc) {
            printf("%s ", argv[optind++]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

int HcclTest::getAviDevs(const char* devs, std::vector<int>& dev_ids)
{

    std::string use_devs(devs);
    std::string pattern = ",";
    std::string::size_type pos; 
    use_devs += pattern;
    size_t val_size = use_devs.size();
    for(size_t i = 0; i < val_size; ++i)
    {
        pos = use_devs.find(pattern, i);
        if(pos < val_size)
        {
            std::string s = use_devs.substr(i, pos);
            int tmp_rank = atoi(s.c_str());
            dev_ids.push_back(tmp_rank); 
            i = pos + pattern.size() - 1;
        }
    }	

    return 0;
}

int HcclTest::get_mpi_proc()
{
    //获取当前进程在所属进程组的编号
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    ACLCHECK(aclrtGetDeviceCount(&dev_count));

    // 入参没有-p参数，npus默认值为device count
    if (npus_flag == 0) {
        npus = dev_count;
    }

    rank_id = proc_rank;
    rank_size = proc_size;

    const char* devs = getenv("HCCL_TEST_USE_DEVS");
    printf("rank_id: %d, rank_size: %d, HCCL_TEST_USE_DEVS: %s\r\n", rank_id, rank_size, devs);
    if(devs != NULL)
    {
        std::vector<int> dev_ids;
        int ret = getAviDevs(devs, dev_ids);
        sort(dev_ids.begin(), dev_ids.end());

        int local_rank;
        local_rank = proc_rank % npus;
        for(int i = 0; i < dev_ids.size(); i++)
        {
            if (local_rank == i)
            {
                dev_id = dev_ids[i];
                break;
            }
        }
    } else {
        dev_id = proc_rank % npus;
    }

    return 0;
}

int HcclTest::hccl_op_base_test()
{
    return 0;
}

int HcclTest::set_device_sat_mode()
{
    const char *soc_name_ptr = aclrtGetSocName();
    if (soc_name_ptr == nullptr) {
        printf("aclrtGetSocName failed");
        return -1;
    }

    std::string support_soc_name = "Ascend910B";
    if (support_soc_name.compare(0, support_soc_name.length(), soc_name_ptr, 0, support_soc_name.length()) == 0) {
        ACLCHECK(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
    }
    return 0;
}

int HcclTest::init_hcclComm()
{
    //设备资源初始化
    ACLCHECK(aclInit(NULL));

    // 指定集合通信操作使用的设备
    ACLCHECK(aclrtSetDevice(dev_id));

    // 关闭溢出检测
    int ret = set_device_sat_mode();
    if (ret != 0)
    {
        printf("set_device_sat_mode execute failed, Detailed logs are stored in path: /root/ascend/log/");
        return ret;
    }

    // 在root_rank获取root_info
    if(rank_id == root_rank) {
#ifndef MEM_DUMP
        printf("the minbytes is %llu, maxbytes is %llu, iters is %d, warmup_iters is %d\n", data->min_bytes, data->max_bytes, iters, warmup_iters);
#endif
        HCCLROOTRANKCHECK(HcclGetRootInfo(&comm_id));
    }
    //将root_info广播到通信域内的其他rank
    MPI_Bcast(&comm_id, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //初始化集合通信域
    HCCLCHECK(HcclCommInitRootInfo(rank_size, &comm_id, rank_id, &hccl_comm));

    ACLCHECK(aclrtCreateEvent(&start_event));
    ACLCHECK(aclrtCreateEvent(&end_event));

    // 创建任务stream
    ACLCHECK(aclrtCreateStream(&stream));
    return 0;
}


int HcclTest::opbase_test_by_data_size(HcclTest* hccl_test)
{
    int ret = 0;
    for (data->data_size = data->min_bytes;\
        data->data_size <= data->max_bytes;\
        (data->step_factor <= 1.0 ? data->data_size += data->step_bytes : data->data_size *= data->step_factor))
    {
        ret = hccl_test->hccl_op_base_test();
        if (ret != 0)
        {
            printf("hccl_op_base execute failed, Detailed logs are stored in path: /root/ascend/log/");
            return ret;
        }
    }
    return 0;
}

int HcclTest::destory_hcclComm()
{
    //销毁任务流
    ACLCHECK(aclrtDestroyStream(stream));

    ACLCHECK(aclrtDestroyEvent(start_event));
    ACLCHECK(aclrtDestroyEvent(end_event));
    //销毁集合通信域
    HCCLCHECK(HcclCommDestroy(hccl_comm));
    //重置设备
    ACLCHECK(aclrtResetDevice(dev_id));
    //设备去初始化
    ACLCHECK(aclFinalize());
    return 0;
}
}

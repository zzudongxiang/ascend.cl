#ifndef _MEM_DUMP_H_
#define _MEM_DUMP_H_

#define MAX_PATH_LEN 100
#define ROOT_PATH "/root/Workdir/hccl_test/log"

void mem_dump(void *ptr, uint64_t len);
void mem_dump_file(void *ptr, uint64_t len, char *file);

// DUMP_DEBUG("allgather", "host_init", rank_id,
//      buff_ptr, buff_size, 
//      send_buff, send_malloc, send_trans, 
//      recv_buff, recv_malloc, recv_trans);
#ifdef MEM_DUMP
    #define DUMP_DEBUG(FUNC_TAG, PTR_TAG, rank_id, buff_ptr, buff_size, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) \
    do { \
        char bin_path[MAX_PATH_LEN]; \
        memset(bin_path, 0, MAX_PATH_LEN); \
        sprintf(bin_path, ROOT_PATH "/" FUNC_TAG "_" PTR_TAG "_rank_%d.bin", rank_id); \
        mem_dump_file((char*)buff_ptr, buff_size, bin_path); \
        printf("dump rank_id: %d >> " PTR_TAG "_buff: %p (malloc: %llu) send_buff: %p (malloc: %llu, trans: %llu), recv_buff: %p (malloc: %llu, trans: %llu), log_path: %s", \
            rank_id, buff_ptr, (long long unsigned int)buff_size, \
            send_buff, (long long unsigned int)send_malloc, (long long unsigned int)send_trans, \
            recv_buff, (long long unsigned int)recv_malloc, (long long unsigned int)recv_trans, \
            bin_path); \
    } while (0)
#else
    #define DUMP_DEBUG(FUNC_TAG, PTR_TAG, rank_id, buff_ptr, buff_size, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) void(0)
#endif

#endif

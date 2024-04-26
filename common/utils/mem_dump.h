#ifndef _MEM_DUMP_H_
#define _MEM_DUMP_H_

#define MAX_PATH_LEN 100
#define B_TO_MB (double)1 / 1024 / 1024
#define B_TO_GB (double)1 / 1024 / 1024 / 1024
#define ROOT_PATH "/root/Workdir/hccl_test/log"

void mem_dump(void *ptr, uint64_t len);
void mem_dump_file(void *ptr, uint64_t len, char *file);

#ifdef MEM_DUMP

// DUMP_DEBUG("allgather", "init", rank_id,
//      host_buff, host_malloc,
//      send_buff, send_malloc, send_trans,
//      recv_buff, recv_malloc, recv_trans);
#define DUMP_DEBUG(FUNC_TAG, PTR_TAG, rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) \
    do                                                                                                                                         \
    {                                                                                                                                          \
        char bin_path[MAX_PATH_LEN];                                                                                                           \
        memset(bin_path, 0, MAX_PATH_LEN);                                                                                                     \
        sprintf(bin_path, ROOT_PATH "/" FUNC_TAG "_" PTR_TAG "_rank_%d.bin", rank_id);                                                         \
        mem_dump_file((char *)host_buff, host_malloc, bin_path);                                                                               \
        printf("func_" FUNC_TAG "_" PTR_TAG "_rank_%d >> log_path: %s\r\n", rank_id, bin_path);                                                \
        printf("func_" FUNC_TAG "_" PTR_TAG "_rank_%d >> host_buff: %p (malloc: %.2f GB, value: %d)\r\n",                                      \
               rank_id,                                                                                                                        \
               host_buff, host_malloc *B_TO_GB, rank_id + 1);                                                                                  \
        printf("func_" FUNC_TAG "_" PTR_TAG "_rank_%d >> send_buff: %p (malloc: %.2f GB, trans: %.2f MB)\r\n",                                 \
               rank_id,                                                                                                                        \
               send_buff, send_malloc *B_TO_GB, send_trans *B_TO_MB);                                                                          \
        printf("func_" FUNC_TAG "_" PTR_TAG "_rank_%d >> recv_buff: %p (malloc: %.2f GB, trans: %.2f MB)\r\n",                                 \
               rank_id,                                                                                                                        \
               recv_buff, recv_malloc *B_TO_GB, recv_trans *B_TO_MB);                                                                          \
    } while (0)

// DUMP_INIT("allgather", rank_id,
//      host_buff, host_malloc,
//      send_buff, send_malloc, send_trans,
//      recv_buff, recv_malloc, recv_trans);
#define DUMP_INIT(FUNC_TAG, rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans)           \
    do                                                                                                                                         \
    {                                                                                                                                          \
        DUMP_DEBUG(FUNC_TAG, "init", rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans); \
        printf("func_" FUNC_TAG "_init_rank_%d >> addr_shift: host(%p)->send(%p): %.2f GB\r\n",                                                \
               rank_id, host_buff, send_buff,                                                                                                  \
               ((double)*(u64 *)&host_buff - *(u64 *)&send_buff) * B_TO_GB);                                                                   \
        printf("func_" FUNC_TAG "_init_rank_%d >> addr_shift: host(%p)->recv(%p): %.2f GB\r\n",                                                \
               rank_id, host_buff, recv_buff,                                                                                                  \
               ((double)*(u64 *)&host_buff - *(u64 *)&recv_buff) * B_TO_GB);                                                                   \
        printf("func_" FUNC_TAG "_init_rank_%d >> addr_shift: recv(%p)->send(%p): %.2f GB\r\n",                                                \
               rank_id, recv_buff, send_buff,                                                                                                  \
               ((double)*(u64 *)&recv_buff - *(u64 *)&send_buff) * B_TO_GB);                                                                   \
                                                                                                                                               \
    } while (0)

// DUMP_DONE("allgather", rank_id, init_buff,
//      host_buff, host_malloc,
//      send_buff, send_malloc, send_trans,
//      recv_buff, recv_malloc, recv_trans);
#define DUMP_DONE(FUNC_TAG, rank_id, init_buff, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) \
    do                                                                                                                                          \
    {                                                                                                                                           \
        DUMP_DEBUG(FUNC_TAG, "done", rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans);  \
        printf("func_" FUNC_TAG "_done_rank_%d >> addr_shift: init(%p)->host(%p): %.2f GB\r\n",                                                 \
               rank_id, init_buff, host_buff,                                                                                                   \
               ((double)*(u64 *)&init_buff - *(u64 *)&host_buff) * B_TO_GB);                                                                    \
        printf("func_" FUNC_TAG "_done_rank_%d >> addr_shift: init(%p)->send(%p): %.2f GB\r\n",                                                 \
               rank_id, init_buff, send_buff,                                                                                                   \
               ((double)*(u64 *)&init_buff - *(u64 *)&send_buff) * B_TO_GB);                                                                    \
        printf("func_" FUNC_TAG "_done_rank_%d >> addr_shift: init(%p)->recv(%p): %.2f GB\r\n",                                                 \
               rank_id, init_buff, recv_buff,                                                                                                   \
               ((double)*(u64 *)&init_buff - *(u64 *)&recv_buff) * B_TO_GB);                                                                    \
    } while (0)

#else

#define DUMP_DEBUG(FUNC_TAG, PTR_TAG, rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) void(0)
#define DUMP_INIT(FUNC_TAG, rank_id, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) void(0)
#define DUMP_DONE(FUNC_TAG, rank_id, init_buff, host_buff, host_malloc, send_buff, send_malloc, send_trans, recv_buff, recv_malloc, recv_trans) void(0)

#endif

#endif

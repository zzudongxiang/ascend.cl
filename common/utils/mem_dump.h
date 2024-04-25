#ifndef _MEM_DUMP_H_
#define _MEM_DUMP_H_

#define MAX_PATH_LEN 100

void mem_dump(void *ptr, uint64_t len);
void mem_dump_file(void *ptr, uint64_t len, char *file);

#endif

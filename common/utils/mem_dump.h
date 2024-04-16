#ifndef _MEM_DUMP_H_
#define _MEM_DUMP_H_

void mem_dump(void *ptr, int len);
void mem_dump_file(void *ptr, int len, const char *file);

#endif

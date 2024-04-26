#include <fstream>
#include <iostream>
#include "mem_dump.h"

void mem_dump(void *ptr, uint64_t len)
{
    printf("===================================================================\n");
    printf("start ptr: %p, len: %llu\r\n", ptr, (long long unsigned int)len);
    printf("-------------------------------------------------------------------\n");
    for (uint64_t i = 0; i < len; i++)
    {
        if (i % 8 == 0 && i != 0)
            printf(" ");
        if (i % 16 == 0 && i != 0)
            printf("\n");
        printf("%02x ", *((uint8_t *)ptr + i));
    }
    printf("===================================================================\n");
}

void mem_dump_file(void *ptr, uint64_t len, char *file)
{return;
    std::ofstream dump_file;
    dump_file.open(file);
    for (uint64_t i = 0; i < len; i++)
    {
        dump_file << *((uint8_t *)ptr + i);
    }
    dump_file.close();
}

typedef union
{
    float fp32;
    uint8_t src[4];
} MEM_VAL_DEF;

MEM_VAL_DEF read_mem(void *ptr, uint8_t dtype_len)
{
    MEM_VAL_DEF mem_value;
    for(uint8_t i = 0; i < dtype_len; i++)
    {
        mem_value.src[i] = *((uint8_t *)ptr + i);
    }
    return mem_value;
}

void print_mem(void *ptr, int64_t mem_start, uint64_t mem_count, MEM_VAL_DEF mem_value)
{
    // int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64/bfp16
    printf("%p (%.2f MB) >> fp32: %f (0x%02X%02X%02X%02X)\n", 
        (uint8_t *)ptr + mem_start,
        (double)mem_count * B_TO_MB,
        *(float *)&mem_value.fp32,
        mem_value.src[0], mem_value.src[1], mem_value.src[2], mem_value.src[3]);
}

void mem_dump_info(void *ptr, uint64_t len, uint8_t dtype_len)
{
    uint8_t print_count = 1;
    uint64_t mem_count = 1;
    uint64_t mem_start = 0;
    MEM_VAL_DEF mem_value = read_mem(ptr, dtype_len);
    printf("===================================================================\n");
    printf("start ptr: %p, len: %.2f GB\r\n", ptr, (double)len / B_TO_GB);
    printf("-------------------------------------------------------------------\n");
    for (uint64_t i = dtype_len; i < len; i += dtype_len)
    {
        MEM_VAL_DEF tmp_mem_value = read_mem((uint8_t *)ptr + i, dtype_len);
        if (tmp_mem_value.fp32 != mem_value.fp32)
        {
            print_mem(ptr, mem_start, mem_count, mem_value);
            mem_value = tmp_mem_value;
            mem_count = dtype_len;
            mem_start = i;
            print_count++;
        }
        else if (print_count > 50)
        {
            printf("... ...\n");
            break;
        }
        else mem_count++;
    }
    print_mem(ptr, mem_start, mem_count, mem_value);
    printf("===================================================================\n");
}

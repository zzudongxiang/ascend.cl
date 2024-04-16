#include <fstream>
#include <iostream>

/**
 * @brief Dump内存信息到控制台
 *
 * @param ptr 内存的地址位置
 * @param len 打印的内存长度
 * 
 * @example mem_dump(check_buf, 512);
 */
void mem_dump(void *ptr, int len)
{
#ifdef MEM_DUMP
    int i;
    printf("--------------------------------------------------\n");
    printf("start ptr: %p, len: %d\r\n", ptr, len);
    printf("--------------------------------------------------\n");
    for (i = 0; i < len; i++)
    {
        if (i % 8 == 0 && i != 0)
            printf(" ");
        if (i % 16 == 0 && i != 0)
            printf("\n");
        printf("%02x ", *((uint8_t *)ptr + i));
    }
    printf("\n");
    printf("--------------------------------------------------\n\n");
#endif
}

/**
 * @brief Dump内存信息到文件
 *
 * @param ptr 待Dump的内存地址
 * @param len Dump的内存长度
 * @param file 保存的文件名
 * 
 * @example mem_dump_file(buffer, 512, "./log/dump.log");
 */
void mem_dump_file(void *ptr, int len, const char *file)
{
#ifdef MEM_DUMP
    std::ofstream dump_file;
    dump_file.open(file);
    for (int i = 0; i < len; i++)
    {
        dump_file << *((uint8_t *)ptr + i);
    }
    dump_file.close();
#endif
}

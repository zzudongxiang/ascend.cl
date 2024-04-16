#include <iostream>

void memory_dump(void *ptr, int len)
{
#ifdef MEM_DUMP
    int i;
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
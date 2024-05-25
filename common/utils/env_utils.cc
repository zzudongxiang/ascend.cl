#include "env_utils.h"

int get_number_env(const char *env_name, int default_value)
{
    int env_value = default_value;
    const char *env_str = getenv(env_name);
    if (env_str != NULL)
    {
        env_value = atoi(env_str);
        u32 nLength = sal_str_len(env_str);
        for (u32 index = 0; index < nLength; index++)
        {
            if (!isdigit(env_str[index]))
            {
                printf("Check whether %s is number.\n", env_name);
                return default_value;
            }
        }
    }
    return env_value;
}

bool get_bool_env(const char *env_name, bool default_value)
{
    int env_value = get_number_env(env_name, default_value ? 1 : 0);
    if (env_value != 0 && env_value != 1)
    {
        printf("Check whether %s is 0 or 1.\n", env_name);
        return default_value;
    }
    else
        return env_value == 1;
}

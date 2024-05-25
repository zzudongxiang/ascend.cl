#ifndef _ENV_UTILS_H_
#define _ENV_UTILS_H_

#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include "hccl_test_common.h"

int get_number_env(const char *env_name, int default_value = 0);
bool get_bool_env(const char *env_name, bool default_value = false);

#endif

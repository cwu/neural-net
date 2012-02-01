#ifndef REAL_NUM_H
#define REAL_NUM_H

#include <stdbool.h>

#define TOL 1e-9

typedef float real_num;
#define REAL_NUM_FORMAT "%f"

bool eq_real_num(real_num x, real_num y);

#endif


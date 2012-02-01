#include "real_num.h"
#include <math.h>

bool eq_real_num(real_num x, real_num y)
{
	return fabs(x - y) <= TOL;
}


#include "ann_func_test.h"
#include "../annetwork.h"

void test_equals();

void ann_func_test_all()
{
}


void test_equals()
{
	ANNetwork *ann1, *ann2;
	ann1 = ANN_Load("load_test1.txt");
	ann2 = ANN_Load("load_test2.txt");

	assert (ANN_equals(ann1, ann2));
}


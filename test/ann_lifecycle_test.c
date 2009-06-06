#include <stdio.h>
#include <assert.h>

#include "ann_lifecycle_test.h"
#include "../annetwork.c"
#include "../real_num.h"

#define TEST_MAX_LAYERS 5

#define assert_eq(x,y) 	do{\
								if(!eq_real_num((real_num)x,(real_num)y))\
								{\
									printf("actual " REAL_NUM_FORMAT " expected "\
											REAL_NUM_FORMAT "\n\n", (real_num)x, (real_num)y);\
									assert(eq_real_num((real_num)x,(real_num)y) == 1);\
								}\
						}while(0)

void test_ANN_Create_Destroy();
void test_ANN_Load();
void test_ANN_Save();

void test_ANN_Lifecycle()
{
	printf("\tStarting ANN Setup and Destroy Tests\n\n");

	test_ANN_Create_Destroy();
	test_ANN_Load();
	test_ANN_Save();

	printf("\tANN Setup and Destroy Tests Passed\n\n");
}


void test_ANN_Create_Destroy()
{
	printf("\t\t\tstarting ann_create_destroy test\n");
	unsigned int n_layers, layer;
	unsigned int n_neurons[TEST_MAX_LAYERS] = { 100, 200, 34, 1, 300 };
	for (n_layers = 2; n_layers < TEST_MAX_LAYERS; n_layers++)
	{
		ANNetwork * ann = ANN_Create(n_layers, n_neurons, 0.14, 0.3044);

		assert_eq(ann->n_layers, n_layers);
		assert_eq(ann->learn_rate , 0.14);
		assert_eq(ann->momentum , 0.3044);

		for (layer = 0; layer < n_layers; layer++)
		{
			assert(ann->n_neurons[layer] == n_neurons[layer]);
		}

		ANN_Destroy(ann);
	}
	printf("\t\t\tpassed ann_create_destroy test\n\n");
}

void test_ANN_Load()
{
	printf("\t\t\tstarting ann_load test\n");
	ANNetwork * ann = ANN_Load("load_test_1.txt");

	#ifdef VERBOSE
	printf("\t\tpassed ann_load load file\n");
	#endif
	
	assert_eq(ann->n_layers, 3);
	assert_eq(ann->learn_rate, 0.1);
	assert_eq(ann->momentum, 1.0);

	#ifdef VERBOSE
	printf("\t\tpassed ann_load setup asserts\n");
	#endif
	
	assert_eq(ann->n_neurons[0], 2);
	assert_eq(ann->n_neurons[1], 2);
	assert_eq(ann->n_neurons[2], 1);
	
	#ifdef VERBOSE
	printf("\t\tpassed ann_load number of neurons asserts\n");
	#endif

	assert_eq(ann->theta[1][0], 0.8);
	assert_eq(ann->theta[1][1], -0.1);
	assert_eq(ann->theta[2][0], 0.3);
	
	#ifdef VERBOSE
	printf("\t\tpassed ann_load theta asserts\n");
	#endif

	assert_eq(ann->weights[1][0][0], 0.5);
	assert_eq(ann->weights[1][0][1], 0.9);
	assert_eq(ann->weights[1][1][0], 0.4);
	assert_eq(ann->weights[1][1][1], 1.0);
	
	assert_eq(ann->weights[2][0][0], -1.2);
	assert_eq(ann->weights[2][1][0], 1.1);
	
	#ifdef VERBOSE
	printf("\t\ttpassed ann_load weights asserts\n\n");
	#endif

	ANN_Destroy(ann);

	printf("\t\t\tpassed ann_load test\n\n");
}

void test_ANN_Save()
{
	printf("\t\t\tstarted ann_save test\n");

	
	
	printf("\t\t\tpassed ann_save test\n\n");
}

#undef TEST_MAX_LAYERS


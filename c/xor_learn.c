#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "real_num.h"
#include "annetwork.h"

int main()
{
	srand(time(NULL));

	unsigned int n_neurons[3] = { 2, 2, 1 };
	ANNetwork *ann = ANN_Create(3, n_neurons, 0.1, 1.0); //ANN_Load("load_xor.txt");
	ANN_FillRandom(ann);

	ANN_Print(ann);

	int epoch = ANN_TrainFile(ann, "xor_training_set.txt");
	if ( epoch == -1 )
	{
		printf("Training failed");
		exit(1);
	}

	printf("epoch: %d\n", epoch);
	

	real_num inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	real_num output[1];
	real_num answers[4][1] = { {0},{1},{1},{0} };
	int training_set;
	for (training_set = 0; training_set < 4; training_set++)
	{
		ANN_FeedForward(ann, inputs[training_set], output);
		printf(REAL_NUM_FORMAT" xor "REAL_NUM_FORMAT" => "REAL_NUM_FORMAT "...%s\n",
				inputs[training_set][0], inputs[training_set][1], output[0],
				(fabs(answers[training_set][0] - output[0]) <= 1e-1 ) ? "PASS" : "FAIL" );
	}
	
	//ANN_Print(ann);
	free(ann);

	printf("\n");

	return 0;
}


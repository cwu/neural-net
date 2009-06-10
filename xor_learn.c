#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "real_num.h"
#include "annetwork.h"

int main()
{
	ANNetwork *ann = ANN_Load("load_xor.txt");

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
				(fabs(answers[training_set][0] - output[0]) <= 1e-2 ) ? "PASS" : "FAIL" );
	}
	
	//printf("\nfeed forward of 1 xor 1 results in "REAL_NUM_FORMAT"\n\n", output[0]);
	//ANN_Print(ann);
	free(ann);

	return 0;
}


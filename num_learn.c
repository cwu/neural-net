#include <stdio.h>
#include <math.h>
#include "annetwork.h"

void display_set(real_num inputs[10][45], real_num answers[10][10]);
void load_data_set(real_num inputs[10][45], real_num answers[10][10]);
int correct(int num, real_num output[], real_num answers[10][10]);

int main()
{
	unsigned int n_neurons[3] = { 45, 10, 10 };
	ANNetwork *ann = ANN_Create(3, n_neurons, 0.1, 0);

	real_num inputs[10][45];
	real_num output[10];
	real_num answers[10][10];
	load_data_set(inputs, answers);
	display_set(inputs, answers);

	ANN_FillRandom(ann);

	int epoch = ANN_TrainFile(ann, "num_training_set.txt");

	printf("epoch : %d\n\n", epoch);

	printf("------------------------------ 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 -\n");
	int num, i;
	for (num = 0; num < 10; num++)
	{
		printf("num : %d ==> neural network -> ", num);

		ANN_FeedForward(ann, inputs[num], output);
		for (i = 0; i < 10; i++)
			printf("%3.1f ", output[i]);

		printf("...%s\n", correct(num, output, answers) ? "PASS" : "FAIL");
	}

	printf("\n");

	ANN_Destroy(ann);

	return 0;
}

void display_set(real_num inputs[10][45], real_num answers[10][10])
{
	printf("============= start inputs ==============\n\n");
	int num, row, col;
	for (num = 0; num < 10; num++)
	{
		printf("\tNumber: %d\n", num);
		printf("--------- 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 -\n");
		printf("answers: ");
		for (col = 0; col < 10; col++)
		{
			printf("%3.1f ", answers[num][col]);
		}
		printf("\n\n");
		for (row = 0; row < 9; row++)
		{
			for (col = 0; col < 5; col++)
			{
				printf(" %c", (fabs(inputs[num][row * 5 + col]) <= 1e-1) ? ' ' : '.');
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("============= end inputs ==============\n\n");
}

int correct(int num, real_num output[],real_num answers[10][10])
{
	int i;
	for (i = 0; i < 10; i++)
	{
		if (fabs(output[i] - answers[num][i]) >= 1e-1)
			return 0;	
	}
	return 1;
}

void load_data_set(real_num inputs[10][45], real_num answers[10][10])
{
	FILE *fin = fopen("num_training_set.txt", "r");

	// skip past the header info
	int dummy_int;
	real_num dummy_real;
	fscanf(fin, "%d %d %f", &dummy_int, &dummy_int, &dummy_real);

	// load the inputs and set the answers
	int training_set, i; 
	for (training_set = 0; training_set < 10; training_set++)
	{
		for (i = 0; i < 45; i++)
			fscanf(fin, REAL_NUM_FORMAT, &inputs[training_set][i]);

		for (i = 0; i < 10; i++)
			fscanf(fin, REAL_NUM_FORMAT, &answers[training_set][i]);
	}


	fclose(fin);
}


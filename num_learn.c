#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "annetwork.h"

void display_set(real_num inputs[10][45], real_num answers[10][10]);
void display_num(int num, real_num pattern[], real_num answer[]);
void load_data_set(real_num inputs[10][45], real_num answers[10][10]);
int correct(int num, real_num output[], real_num answers[10][10]);

inline int random_int(int range) { return rand () % range ; };
inline void invert(real_num * x) { *x = (fabs(*x) <= 1e-2) ? 1 : 0; };

int main()
{
	srand(time(NULL));


	// load and display the training set
	real_num inputs[10][45];
	real_num output[10];
	real_num answers[10][10];
	load_data_set(inputs, answers);
	display_set(inputs, answers);

	// create the neural network and fill it with random weights
	unsigned int n_neurons[3] = { 45, 10, 10 };
	ANNetwork *ann = ANN_Create(3, n_neurons, 0.1, 0);
	ANN_FillRandom(ann);

	// train the neural network
	int epoch = ANN_TrainFile(ann, "num_training_set.txt");

	// display the results of the training
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


	// allow the user to interactively test the neural network
	printf("Interactive mode...loading 1\n\n");

	// load pattern 1 first
	real_num pattern[45];
	memcpy(pattern, inputs[1], sizeof(real_num) * 45);
	num = 1;

	int row, col;
	char command;
	do
	{
		printf("enter a command (h to display a list of commands): ");
		while ((command = getchar()) == ' ' || command == '\n'); // make sure we're only reading valid characters and not junk
		printf("\n");	

		switch (command) 
		{
			case 'n':
				printf("=============== Feeding the Neural Network ================\n\n");
				ANN_FeedForward(ann, pattern, output);
				display_num(num, pattern, answers[num]);
				printf("\n========================= Results =========================\n\n");
				printf("------------------ 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 -\n");
				printf("neural network -> ", num);
				for (i = 0; i < 10; i++)
					printf("%3.1f ", output[i]);
				printf("\n\n========================= End =============================\n");
				break;

			case 'p':
				display_num(num, pattern, answers[num]);
				break;

			case 'l':
				printf("enter the number to load: ");
				int temp_num;
				scanf("%d", &temp_num);
				if (temp_num < 0 || temp_num > 9)
				{
					printf("please enter a number 0-9\n");
				}
				else // valid number
				{
					num = temp_num;
					memcpy(pattern, inputs[num], sizeof(real_num) * 45);

					printf("loaded %d\n",num);
					display_num(num, pattern, answers[num]);
				}
				break;

			case 'i':
				printf("enter the [row column] to invert: ");
				scanf("%d %d", &row, &col);
				if (row < 0 || col < 0 || row >= 9 || col >= 5)
					printf("row or col out of bounds, row must be 0-8 ; col must be 0-4\n");
				else
					invert(&pattern[row * 5 + col]);
				break;

			case 'r':
				printf("enter the number of squares to invert: ");
				int n_squares, square;
				scanf("%d", &n_squares);
				for (square = 0; square < n_squares; square++)
				{
					row = random_int(9);
					col = random_int(5);
					invert(&pattern[row * 5 + col]);
				}
				printf("\tfinished inverting %d squares\n", n_squares);
				break;

			case 'q':
				break;

			case 'h':
				printf("n - ask the neural network to recognize the pattern\n");
				printf("i - invert a square\n");
				printf("r - invert a specified number of random squares\n");
				printf("l - load a number\n");
				printf("p - prints the current pattern\n");
				printf("q - quit\n");
				break;

			default:
				printf("no '%c' command\n", command);
				break;
		}

		printf("\n");
	}
	while (command != 'q');

	ANN_Destroy(ann);

	return 0;
}

void display_set(real_num inputs[10][45], real_num answers[10][10])
{
	printf("============= start inputs ==============\n\n");
	int num;
	for (num = 0; num < 10; num++)
	{
		display_num(num, inputs[num], answers[num]);
	}
	printf("============= end inputs ==============\n\n");
}

void display_num(int num, real_num pattern[], real_num answer[])
{
	printf("\tNumber: %d\n", num);
	printf("--------- 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 -\n");
	printf("answers: ");
	int row, col;
	for (col = 0; col < 10; col++)
	{
		printf("%3.1f ", answer[col]);
	}
	printf("\n\n");

	for (row = 0; row < 9; row++)
	{
		for (col = 0; col < 5; col++)
		{
			printf(" %c", (fabs(pattern[row * 5 + col]) <= 1e-1) ? ' ' : '.');
		}
		printf("\n");
	}
	printf("\n");
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


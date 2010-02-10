#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "annetwork.h"

#define assert_malloc(ptr) 	do{\
								if(ptr==NULL){\
									fprintf(stderr,\
										"error: %s:%u could not allocate space for %s\n\n",\
										__FILE__, __LINE__, #ptr);\
									return NULL;\
								}\
						 	}while(0)

#define INPUT_LAYER 0
#define HIDDEN_START 1
struct NeuralNetwork {
	unsigned int n_layers;
	unsigned int *n_neurons;	

	real_num **theta;
	real_num ***weights;

	real_num **outputs;
	real_num **errors;

	real_num learn_rate;
	real_num momentum;
};

// Static function prototypes 
#define RAND_WEIGHT_RANGE 4.8
static inline real_num _RandNum(real_num min, real_num max) { return rand() / ((real_num) RAND_MAX + 1) * (max - min) + min / 2; }
static inline real_num _Limiter(real_num x) { return 1.0 / (1.0 + exp(-x)); }

static int _AllocateSetVectors(TrainingSet *set, int n_input, int n_answer);
static void _FeedForward(ANNetwork *ann, real_num input[]);
static real_num _PropogateErrors(ANNetwork *ann, real_num answers[]);

ANNetwork* ANN_Create(unsigned int n_layers, unsigned int *n_neurons, real_num learn_rate, real_num momentum)
{
	assert( ((int)n_layers) >= 2);

	ANNetwork *ann = malloc(sizeof(ANNetwork));
	assert_malloc(ann);

#ifdef VERBOSE
	printf("\n\tallocated ann\n");
#endif

	ann->n_layers = n_layers;
	ann->n_neurons = malloc(sizeof(*n_neurons) * n_layers);
	assert_malloc(ann->n_neurons);
	memcpy(ann->n_neurons, n_neurons, sizeof(*n_neurons) * n_layers);

#ifdef VERBOSE
	printf("\tset up n_layers and n_neurons 1d\n");
#endif

	ann->theta = malloc(sizeof(real_num*) * n_layers);
	assert_malloc(ann->theta);
	ann->outputs = malloc(sizeof(real_num*) * n_layers);
	assert_malloc(ann->outputs);
	ann->errors = malloc(sizeof(real_num*) * n_layers);
	assert_malloc(ann->errors);
	ann->weights = malloc(sizeof(real_num**) * n_layers);
	assert_malloc(ann->weights);

#ifdef VERBOSE
	printf("\tset up theta, error, output, and weights 1d\n");
#endif

	ann->theta[INPUT_LAYER] = NULL;
	ann->errors[INPUT_LAYER] = NULL;
	ann->outputs[INPUT_LAYER] = malloc(sizeof(real_num) * n_neurons[INPUT_LAYER]);
	assert_malloc(ann->outputs);
	ann->weights[INPUT_LAYER] = NULL;
	int layer, neuron;
	for (layer = HIDDEN_START; layer < n_layers; layer++)
	{
		ann->theta[layer] = malloc(sizeof(real_num) * n_neurons[layer]);
		assert_malloc(ann->theta[layer]);
		ann->errors[layer] = malloc(sizeof(real_num) * n_neurons[layer]);
		assert_malloc(ann->errors[layer]);
		ann->outputs[layer] = malloc(sizeof(real_num) * n_neurons[layer]);
		assert_malloc(ann->outputs[layer]);

		ann->weights[layer] = malloc(sizeof(real_num*) * n_neurons[layer-1]);
		assert_malloc(ann->weights[layer]);
		for (neuron = 0; neuron < n_neurons[layer-1]; neuron++)
		{
			ann->weights[layer][neuron] = malloc(sizeof(real_num) * n_neurons[layer]);
			assert_malloc(ann->weights[layer][neuron]);
		}
	}

#ifdef VERBOSE
	printf("\tfinished setting up theta and weights\n");
#endif

	ann->learn_rate = learn_rate;
	ann->momentum = momentum;

#ifdef VERBOSE
	printf("\tdone creating the ANN\n\n");
#endif

	return ann;
}

void ANN_Destroy(ANNetwork *ann)
{
#ifdef VERBOSE
	printf("\n\tstart destroying the ANN\n\n");
#endif

	// free the theta value arrays
	int layer;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		free(ann->theta[layer]);
		free(ann->errors[layer]);
		free(ann->outputs[layer]);
	}
	free(ann->theta);
	free(ann->errors);
	free(ann->outputs);

#ifdef VERBOSE
	printf("\tfreed the thetas\n");
#endif
	
	// free the weights
	int neuron;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		for (neuron = 0; neuron < ann->n_neurons[layer-1]; neuron++)
			free(ann->weights[layer][neuron]);
		free(ann->weights[layer]);
	}
	free(ann->weights);

#ifdef VERBOSE
	printf("\tfreed the weights\n");
#endif
	
	// free the number of neurons array
	free(ann->n_neurons);
	
#ifdef VERBOSE
	printf("\tfreed the n_neurons\n");
#endif

	// free the neural network
	free(ann);

#ifdef VERBOSE
	printf("\tdone destorying the ANN\n\n");
#endif
}

ANNetwork* ANN_Load(char *filename)
{
	FILE *fin = fopen(filename, "r");
	if ( fin == NULL )
	{
#ifdef VERBOSE
		printf("file : %s could not be opening for loading the ANN\n", filename);
#endif
		return NULL;
	}

	unsigned int n_layers;
	real_num learn_rate, momentum;

	// read in the number of layers, learning rate and momentum constant
	fscanf(fin, "%u " REAL_NUM_FORMAT " " REAL_NUM_FORMAT, &n_layers, &learn_rate, &momentum);

	unsigned int n_neurons[n_layers];
	// read the number of neurons in each layer
	assert_malloc(n_neurons);
	int layer;
	for (layer = 0; layer < n_layers; layer++)
		fscanf(fin, "%u", n_neurons + layer);

	// create the neural network and if successful, load the thetas and weights
	ANNetwork *ann = ANN_Create(n_layers, n_neurons, learn_rate, momentum);
	if ( ann != NULL)
	{
		real_num read_value;
		int neuron, next_neuron;

		// read in the theta values
		for (layer = HIDDEN_START; layer < n_layers; layer++)
		{
			for (neuron = 0; neuron < n_neurons[layer]; neuron++)
			{
				fscanf(fin, REAL_NUM_FORMAT, &read_value);
				ann->theta[layer][neuron] = read_value;
			}
		}

		// read in the weights
		for (layer = HIDDEN_START; layer < n_layers; layer++)
		{
			for (neuron = 0; neuron < n_neurons[layer-1]; neuron++)
			{
				for (next_neuron = 0; next_neuron < n_neurons[layer]; next_neuron++)
				{
					fscanf(fin, REAL_NUM_FORMAT, &read_value);
					ann->weights[layer][neuron][next_neuron] = read_value;
				}
			}
		}

	}

	fclose(fin);
	return ann;
}

int ANN_Save(char *filename, ANNetwork *ann)
{
	FILE *fout = fopen(filename, "w");

	if ( fout == NULL )
	{
#ifdef VERBOSE
		printf("file : %s could not be opening for saving the ANN\n", filename);
#endif
		return -1;
	}

	// ouptut the number of layers, learning rate and momentum constant
	fprintf(fout, "%u " REAL_NUM_FORMAT " " REAL_NUM_FORMAT"\n",
						ann->n_layers, ann->learn_rate, ann->momentum);

	// print out the number of neurons in each layer
	int layer;
	for (layer = 0; layer < ann->n_layers; layer++)
		fprintf(fout, "%u ", ann->n_neurons[layer]);

	// print out the theta values
	int neuron, next_neuron;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
			fprintf(fout, REAL_NUM_FORMAT" ", ann->theta[layer][neuron]);

	// print out the weights
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
		for (neuron = 0; neuron < ann->n_neurons[layer-1]; neuron++)
			for (next_neuron = 0; next_neuron < ann->n_neurons[layer]; next_neuron++)
				fprintf(fout, REAL_NUM_FORMAT" ", ann->weights[layer][neuron][next_neuron]);
	fflush(fout);
	fclose(fout);
	
	return 0;
}

void ANN_Print(ANNetwork *ann)
{
	printf("============= Start Neural Network =============\n\n");

	// ouptut the number of layers, learning rate and momentum constant
	printf("n_layers: %u learn_rate: " REAL_NUM_FORMAT " momentum: "REAL_NUM_FORMAT" \n",
						ann->n_layers, ann->learn_rate, ann->momentum);

	// print out the number of neurons in each layer
	printf("n_neurons per layer: ");
	int layer;
	for (layer = 0; layer < ann->n_layers; layer++)
		printf("%u ", ann->n_neurons[layer]);
	printf("\n\n");

	// print out the theta values
	printf("\t\ttheta:\n");
	int neuron, next_neuron;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		printf("\tlayer %d:\n", layer);
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
			printf(REAL_NUM_FORMAT" ", ann->theta[layer][neuron]);
		printf("\n");
	}

	// print out the weights
	printf("\t\tweights:\n");
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		printf("\tlayer %d:\n", layer);
		for (neuron = 0; neuron < ann->n_neurons[layer-1]; neuron++)
			for (next_neuron = 0; next_neuron < ann->n_neurons[layer]; next_neuron++)
				printf("from %d to %d -> "REAL_NUM_FORMAT"\n",
						neuron, next_neuron, ann->weights[layer][neuron][next_neuron]);
	}

	printf("\n");

	// if verbose print out the output and errors
#ifdef VERBOSE
	printf("\t\toutputs:\n");
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		printf("\tlayer %d:\n", layer);
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
			printf(REAL_NUM_FORMAT" ", ann->outputs[layer][neuron]);
		printf("\n");
	}
	printf("\n");

	printf("\t\terrors:\n");
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		printf("\tlayer %d:\n", layer);
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
			printf(REAL_NUM_FORMAT" ", ann->errors[layer][neuron]);
		printf("\n");
	}
	printf("\n");
#endif

	printf("============== End Neural Network ==============\n\n");
}

int ANN_Equals(ANNetwork * ann1, ANNetwork* ann2)
{
	// check if either is NULL
	if ( ann1 == NULL || ann2 == NULL )
		return false;

	// check if we have two pointers to the same neural network
	if ( ann1 == ann2 )
		return true;

	// check if the neural network has the same number of layers, input neurons,
	// learning rate and mometum constants
	if ( ann1->n_layers != ann2->n_layers ||
		   	ann1->n_neurons[INPUT_LAYER] != ann2->n_neurons[INPUT_LAYER] || // it was asserted n_layers >= 2 
			!eq_real_num(ann1->learn_rate, ann2->learn_rate) ||
			!eq_real_num(ann1->momentum, ann2->momentum) )
		return false;
	
	// check if the number of neurons and each weight is the same for both neural
	// networks
	int layer, neuron, next_neuron;
	for (layer = HIDDEN_START; layer < ann1->n_layers; layer++)
	{
		if ( ann1->n_neurons[layer] != ann1->n_neurons[layer] )
			return false;

		for (next_neuron = 0; next_neuron < ann1->n_neurons[layer]; next_neuron++)
		{
			if ( !eq_real_num(ann1->theta[layer][next_neuron],
						ann2->theta[layer][next_neuron]) )
				return false;

			for (neuron = 0; neuron < ann1->n_neurons[layer - 1]; neuron++)
				if ( !eq_real_num(ann1->weights[layer][neuron][next_neuron], 
						ann2->weights[layer][neuron][next_neuron]) )
					return false;
		}
	}

	return true;
}

void ANN_FillRandom(ANNetwork *ann)
{
	if ( ann == NULL )
		return;

	int layer, neuron, next_neuron, n_input;
	real_num range;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		// number of neurons which outputs to each neuron in this layer is # of neurons in prev. layer + 1
		n_input = ann->n_neurons[layer - 1] + 1; 
		for (next_neuron = 0; next_neuron < ann->n_neurons[layer]; next_neuron++)
		{
			// initialize weights to a random number [RAND_WEIGHT_RANGE/n,RAND_WEIGHT_RANGE 2.4/n]
			// where n is the number of input to the specific neuron
			range = (real_num) RAND_WEIGHT_RANGE / n_input;
			ann->theta[layer][next_neuron] = _RandNum(-range, range);
			for (neuron = 0; neuron < ann->n_neurons[layer - 1]; neuron++)
				ann->weights[layer][neuron][next_neuron] = _RandNum(-range, range);
		}
	}
}

void ANN_FeedForward(ANNetwork *ann, real_num input[], real_num output[])
{
	// calculate the output and then copy it into the output array
	_FeedForward(ann, input);
	memcpy(output, ann->outputs[ann->n_layers - 1], sizeof(real_num) * ann->n_neurons[ann->n_layers-1]);
}

int ANN_TrainFile(ANNetwork *ann, char *filename)
{
	FILE *fin = fopen(filename, "r");
	if ( fin == NULL )
	{
#ifdef VERBOSE	
		printf("file : %s could not be opened for training\n", filename);
#endif
		return -1;
	}

	// initialize the training set
	TrainingSet set;
	fscanf(fin, "%d %d " REAL_NUM_FORMAT, &set.n_training_sets, &set.max_epoch, &set.desired_error);

	// find the number of input and output neurons
	int n_input = ann->n_neurons[INPUT_LAYER];
	int n_answers = ann->n_neurons[ann->n_layers - 1];

	// allocate the input and answers vector
	int malloc_error = _AllocateSetVectors(&set, n_input, n_answers);
	if ( malloc_error )
	{
#ifdef VERBOSE
		printf("malloc_error could not allocate input and answers vectors for training from a"
				"file...needed %u bytes\n", sizeof(real_num) * set.n_training_sets * 
				(n_input + n_answers + 2 ));
#endif
		return -1;
	}
	
	int training_set, neuron;
	for (training_set = 0; training_set < set.n_training_sets; training_set++)
	{
		
		// read in the inputs
		for (neuron = 0; neuron < n_input; neuron++)
		{
			fscanf(fin, REAL_NUM_FORMAT, set.inputs[training_set] + neuron);
		}

		// read in the answers
		for (neuron = 0; neuron < n_answers; neuron++)
		{
			fscanf(fin, REAL_NUM_FORMAT, set.answers[training_set] + neuron);
		}
	}

	// perform the training
	int epoch = ANN_Train(ann, set);

	// free the input and answer vectors
	for (training_set = 0; training_set < set.n_training_sets; training_set++)
	{
		free(set.inputs[training_set]);
		free(set.answers[training_set]);
	}
	free(set.inputs);
	free(set.answers);
	
	return epoch;
}

int ANN_Train(ANNetwork *ann, TrainingSet set)
{
	int epoch = 0, training_set;
	real_num sumSqrErrors;
	for (epoch = 0; epoch < set.max_epoch; epoch++)
	{
		sumSqrErrors = 0;
		for (training_set = 0; training_set < set.n_training_sets; training_set++)
		{
			sumSqrErrors += ANN_Learn(ann, set.inputs[training_set], set.answers[training_set]);
		}
		
		if ( sumSqrErrors <= set.desired_error ) {
			return epoch;
		}
	}

	return epoch;
}

real_num ANN_Learn(ANNetwork *ann, real_num input[], real_num answer[])
{
	_FeedForward(ann, input);
	return _PropogateErrors(ann, answer);
}


static void _FeedForward(ANNetwork *ann, real_num input[])
{
	assert(ann != NULL);
	assert(input != NULL);

	// copy the input into the output of the input layer
	memcpy(ann->outputs[INPUT_LAYER], input, sizeof(real_num) * ann->n_neurons[INPUT_LAYER]);

	int layer, prev_neuron, neuron;
	real_num output;
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
	{
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
		{
			output = -ann->theta[layer][neuron];
			for (prev_neuron = 0; prev_neuron < ann->n_neurons[layer - 1]; prev_neuron++)
			{
				output += ann->outputs[layer - 1][prev_neuron] * ann->weights[layer][prev_neuron][neuron];
			}
			ann->outputs[layer][neuron] = _Limiter(output);
		}
	}
}

static real_num _PropogateErrors(ANNetwork *ann, real_num answers[])
{
	assert(ann != NULL);
	assert(answers != NULL);

	// keep track of the sumSqrErrors
	real_num sumSqrErrors = 0;

	// calculate the errors
	int layer = ann->n_layers - 1; // output layer
	int neuron, prev_neuron, next_neuron;
	real_num output, error;
	for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
	{
		output = ann->outputs[layer][neuron];
		error = answers[neuron] - output; // expected - actual
		sumSqrErrors += error * error; // sum up the square of the errors
		ann->errors[layer][neuron] = output * ( 1 - output ) * error; // error gradient

		// correct the theta of the output neuron
		real_num correction = ann->learn_rate * -1 * ann->errors[layer][neuron];
		ann->theta[layer][neuron] += correction;

		// correct weights
		for (prev_neuron = ann->n_neurons[layer - 1] -1; prev_neuron >= 0; prev_neuron--)
		{
			correction = ann->learn_rate * ann->outputs[layer - 1][prev_neuron] *
							ann->errors[layer][neuron];
			ann->weights[layer][prev_neuron][neuron] += correction;
		}
	}

	
	// back propogate the errors and perform weight correction
	for (layer = ann->n_layers - 2 ; layer >= HIDDEN_START; layer--) // start at hidden end
	{
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
		{
			// propgate the errors
			error = 0;
			for (next_neuron = ann->n_neurons[layer + 1] - 1; next_neuron >= 0; next_neuron--)
			{
				error += ann->errors[layer + 1][next_neuron] *
				   				ann->weights[layer + 1][neuron][next_neuron];
			}

			// calculate the error gradient for this neuron
			output = ann->outputs[layer][neuron];
			ann->errors[layer][neuron] = output * ( 1 - output ) * error;

			// correct theta
			real_num correction = ann->learn_rate * -1 * ann->errors[layer][neuron];
			ann->theta[layer][neuron] += correction;

			// correct weights
			for (prev_neuron = ann->n_neurons[layer - 1] -1; prev_neuron >= 0; prev_neuron--)
			{
				correction = ann->learn_rate * ann->outputs[layer - 1][prev_neuron] *
								ann->errors[layer][neuron];
				ann->weights[layer][prev_neuron][neuron] += correction;
			}
		}
	}

	return sumSqrErrors;
}

static int _AllocateSetVectors(TrainingSet *set, int n_input, int n_answer)
{
	set->inputs = malloc(sizeof(real_num*) * set->n_training_sets);
	if ( set->inputs == NULL )
	{
		return 1;
	}
	
	set->answers = malloc(sizeof(real_num*) * set->n_training_sets);
	if ( set->answers == NULL )
	{
		free(set->inputs);
		return 1;
	}

	int malloc_error = 0;
	int training_set;
	for (training_set = 0; training_set < set->n_training_sets; training_set++)
	{
		set->inputs[training_set] = malloc (sizeof(real_num) * n_input);
		if ( set->inputs[training_set] == NULL )
		{
			malloc_error = 1;
			break;
		}

		set->answers[training_set] = malloc (sizeof(real_num) * n_answer);
		if ( set->answers[training_set] == NULL )
		{
			malloc_error = 1;
			free(set->inputs[training_set]);
			break;
		}
	}

	// free allocated memory if allocation for the vectors fail
	if ( malloc_error )
	{
		for (training_set = training_set - 1; training_set >= 0; training_set--)
		{
			free(set->inputs[training_set]);
			free(set->answers[training_set]);
		}
		free(set->inputs);
		free(set->answers);
	}

	return malloc_error;
}


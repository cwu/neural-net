#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
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
	real_num learn_rate;
	real_num momentum;
};

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
	ann->weights = malloc(sizeof(real_num**) * n_layers);
	assert_malloc(ann->weights);

	#ifdef VERBOSE
	printf("\tset up theta and weights 1d\n");
	#endif

	ann->theta[INPUT_LAYER] = NULL;
	ann->weights[INPUT_LAYER] = NULL;
	int layer, neuron;
	for (layer = HIDDEN_START; layer < n_layers; layer++)
	{
		ann->theta[layer] = malloc(sizeof(real_num) * n_neurons[layer]);
		assert_malloc(ann->theta[layer]);

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
	for (layer = 0; layer < ann->n_layers; layer++)
		free (ann->theta[layer]);
	free(ann->theta);

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
		return NULL;

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

	// ouptut the number of layers, learning rate and momentum constant
	fprintf(fout, "%u " REAL_NUM_FORMAT " " REAL_NUM_FORMAT,
						ann->n_layers, ann->learn_rate, ann->momentum);

	// print out the number of neurons in each layer
	int layer;
	for (layer = 0; layer < ann->n_layers; layer++)
		fprintf(fout, "%u", ann->n_neurons[layer]);

	int neuron, next_neuron;

	// print out the theta values
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
		for (neuron = 0; neuron < ann->n_neurons[layer]; neuron++)
			fprintf(fout, REAL_NUM_FORMAT, ann->theta[layer][neuron]);

	// print out the weights
	for (layer = HIDDEN_START; layer < ann->n_layers; layer++)
		for (neuron = 0; neuron < ann->n_neurons[layer-1]; neuron++)
			for (next_neuron = 0; next_neuron < ann->n_neurons[layer]; next_neuron++)
				fprintf(fout, REAL_NUM_FORMAT, ann->weights[layer][neuron][next_neuron]);
	fflush(fout);
	fclose(fout);
	
	return 0;
}

int ANN_Equals(ANNetwork * ann1, ANNetwork* ann2)
{
	if ( ann1 == NULL || ann2 == NULL )
		return false;

	if ( ann1 == ann2 )
		return true;

	if ( ann1->n_layers != ann2->n_layers ||
		   	ann1->n_neurons[INPUT_LAYER] != ann2->n_neurons[INPUT_LAYER] || // n_layers >= 2
			!eq_real_num(ann1->learn_rate, ann2->learn_rate) ||
			!eq_real_num(ann1->momentum, ann2->momentum) )
		return false;
	
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

			for (neuron = 0; neuron < ann1->n_neurons[layer-1]; neuron++)
				if ( !eq_real_num(ann1->weights[layer][neuron][next_neuron], 
						ann2->weights[layer][neuron][next_neuron]) )
					return false;
		}
	}

	return true;
}

int ANN_FillRandom(ANNetwork *ann)
{
	if ( ann == NULL )
		return false;
	
	
}



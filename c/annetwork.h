#ifndef ANNETWORK_H
#define ANNETWORK_H

#include "real_num.h"

typedef struct NeuralNetwork ANNetwork;

typedef struct TrainingSetTag {
	unsigned int n_training_sets;
	real_num **inputs;
	real_num **answers;

	int max_epoch;
	real_num desired_error;
} TrainingSet;

ANNetwork* ANN_Create(unsigned int n_layers, unsigned int *n_neurons, real_num learn_rate, real_num momentum);
void ANN_Destroy(ANNetwork * ann);

ANNetwork* ANN_Load(char *fileName);
int ANN_Save(char *filename, ANNetwork *ann);

int ANN_Equals(ANNetwork *ann1, ANNetwork *ann2);
void ANN_FillRandom(ANNetwork *ann);

void ANN_FeedForward(ANNetwork *ann, real_num input[], real_num output[]);
real_num ANN_Learn(ANNetwork *ann, real_num input[], real_num answer[]);

int ANN_TrainFile(ANNetwork * ann, char *filename);
int ANN_Train(ANNetwork *ann, TrainingSet set);

void ANN_Print(ANNetwork *ann);

#endif


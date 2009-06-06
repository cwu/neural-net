#ifndef ANNETWORK_H
#define ANNETWORK_H

#include "real_num.h"

typedef struct NeuralNetwork ANNetwork;

ANNetwork* ANN_Create(unsigned int n_layers, unsigned int *n_neurons, real_num learn_rate, real_num momentum);
void ANN_Destroy(ANNetwork * ann);

ANNetwork* ANN_Load(char *fileName);
int ANN_Save(char *filename, ANNetwork *ann);

int ANN_Equals(ANNetwork *ann1, ANNetwork *ann2);
void ANN_FillRandom(ANNetwork *ann);


/*
int ANN_TrainFile(ANNetwork * ann, char *filename);
int ANN_Train

void feedForward(NeuralNetwork * network, double input[]);
double propogateErrors(NeuralNetwork * n, double answers[]);
double limiter(double x);
double learn(NeuralNetwork * network, double input[], double answer[]);
int train(NeuralNetwork * dumb, double inputs[][MAX_N_IN_LAYER], double answers[][MAX_N_IN_LAYER],
			 int nTrainingSets, double accuracyThreshold, int epochLimit);
void printNetwork(NeuralNetwork * network);
double sse(NeuralNetwork * network, double inputs[][MAX_N_IN_LAYER], double answers[][MAX_N_IN_LAYER],
				int nTrainingSets);
*/

#endif


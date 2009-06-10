CC = gcc
DEFINE = -DVERBOSE
CFLAGS = -Wall $(DEFINE)
ANN_OBJ = annetwork.o real_num.o

xor: xor_learn.o $(ANN_OBJ)
	gcc -g $(CFLAGS) xor_learn.o $(ANN_OBJ) -o xor -lm

xor_learn.o: annetwork.h real_num.h

annetwork.o: annetwork.h real_num.h

real_num.o: real_num.h

clean:
	rm *.o

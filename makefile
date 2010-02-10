CC = gcc
DEFINE = -DVERBOSE
CFLAGS = -Wall $(DEFINE) -g
ANN_OBJ = annetwork.o real_num.o

all: num xor

num: num_learn.o $(ANN_OBJ)
	gcc $(CFLAGS) num_learn.o $(ANN_OBJ) -o num -lm

xor: xor_learn.o $(ANN_OBJ)
	gcc $(CFLAGS) xor_learn.o $(ANN_OBJ) -o xor -lm

num_learn.o: annetwork.h real_num.h

xor_learn.o: annetwork.h real_num.h

annetwork.o: annetwork.h real_num.h

real_num.o: real_num.h

clean:
	rm -f *.o num num.exe xor xor.exe

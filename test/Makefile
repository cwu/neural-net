CFLAGS = -Wall
OPTIONS = #-DVERBOSE

test: test.o ann_lifecycle_test.o annetwork.o real_num.o
	gcc $(CFLAGS) -g test.o ann_lifecycle_test.o real_num.o -o test

test.o: test.c
	gcc $(CFLAGS) $(OPTIONS) -c test.c

ann_lifecycle_test.o: ann_lifecycle_test.c ann_lifecycle_test.h ../annetwork.c
	gcc $(CFLAGS) $(OPTIONS) -c ann_lifecycle_test.c

annetwork.o: ../annetwork.h ../annetwork.c
	gcc $(CFLAGS) $(OPTIONS) -c ../annetwork.c

real_num.o: ../real_num.h ../real_num.c
	gcc $(CFLAGS) $(OPTIONS) -c ../real_num.c
	
clean:
	rm *.o


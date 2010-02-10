This is a simple artificial neural network I wrote in my spare time. It uses a Back propogation network.

The acutal network is in annetwork.{c,h} and there are two programs generated.

One is a xor learning program (main code is in xor_learn.c) and can be made by `make xor`.

The other is a number recognizer. Code in num_learn.c and can be made by `make num`. Running it will provide an interactive command line program to play around and test the neural network.

The txt files used for training take the form of

	<# training sets> <max epoch> <desired sum sqr error bound>

	<inputs {0,1} with spaces as a delimiter>

	<correct output {0,1} with spaces as a delimiter>

p.s. this includes a back propogation implementation in haskell under the haskell/ folder


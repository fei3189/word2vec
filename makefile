word2vec: *.cpp *.h
	g++  -lm -pthread -std=c++11 -Ofast -o word2vec -Wall *.cpp

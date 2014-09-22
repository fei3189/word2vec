//
//  main.cpp
//  word2vec
//
//  Copyright (C) 2014 Fei Jiang <f91.jiang@gmail.com>
//
//  Part of the code in main.cpp are from https://code.google.com/p/word2vec/
//

#include <iostream>
#include <string>
#include "word2vec.h"

void print_info() {
    printf("Noise-Constrastive Estimation for word vector estimation\n\n");
    
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 200\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-nce <int>\n");
    printf("\t\tPrior K (number of noise samples) for noise-contrastive estimation, default is 10\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 2)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the global learning rate; default is 0.05\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-ivlbl <int>\n");
    printf("\t\tUse the inversed LBL model; default is 0\n");
    printf("\t-dependent <int>\n");
    printf("\t\tWhether to use window position dependent weights; default is 0\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -nce 10 -binary 0 -ivlbl 0 -iter 1\n\n");
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        print_info();
        return 0;
    }
    
    int i;
    int embed_size = 200, binary = 0, ivlbl = 0, window = 5, nce = 10;
    int num_threads = 2, iter = 1, min_count = 5, dependent = 0;
    float rate = 0.05;
    std::string train_file, output_file;
    
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) embed_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) train_file = std::string(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-ivlbl", argc, argv)) > 0) ivlbl = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) rate = atof(argv[i + 1]);
    
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) output_file = std::string(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-nce", argc, argv)) > 0) nce = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-dependent", argc, argv)) > 0) dependent = atoi(argv[i + 1]);
    
    if (train_file.empty()) {
        std::cerr << "Training file missing" << std::endl;
        return -1;
    }
    if (output_file.empty()) {
        std::cerr << "Output file missing" << std::endl;
        return -1;
    }
    word2vec::model model(dependent, iter, ivlbl, nce, embed_size, window, rate);
    model.learn_vocab(train_file, min_count);
    model.train(train_file, num_threads);
    model.save(output_file, binary);
    
    return 0;
}
//
//  word2vec.h
//  word2vec
//
//  Created by 姜飞 on 14-9-12.
//  Copyright (c) 2014年 thuir. All rights reserved.
//

#ifndef word2vec_word2vec_h
#define word2vec_word2vec_h

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <ctime>

namespace word2vec {
    struct word_freq {
        std::string word;
        int freq;
    };
    
    class model {
    private:
        int option_share_inout;
        int option_dependent;
        int option_iter_num;
        int option_ivlbl;
        int option_nce;
        int window;
        int embed_size;
        
    private:
        // sum of square of gradients
        float *g_in;
        float *g_out;
        float *g_weight;
        
        // Init parameters
        static float g_init;
        static float v_init_bound;
        
        // input vector and output vector
        float *v_in;
        float *v_out;
        
        // noise distribution
        float *pn;
        
        // positive weights
        float *weight;
        
        float learning_rate;
        int max_doc_len;
        int vocab_size;
        
        long long total_words;
        long long trained_words;
        
        std::unordered_map<std::string, int> word2id;
        std::vector<std::string> id2word;
        std::vector<int> sample_table;  // sampling for noise-contrastive estimation
        
        void train_thread(const std::string &filename, int num_threads, int tid);
        int read_words(std::vector<int> &words, std::ifstream &fin, long long end);
        void train_ivlbl(const std::vector<int> &words);
        void train_vlbl(const std::vector<int> &words);
        void train_vlbl1(const std::vector<int> &words);
        void train_ivlbl1(const std::vector<int> &doc);
        int read_words1(std::vector<int> &words, std::ifstream &fin, long long end);
        void init_sample_table();
        void init(int dep, int iter, int ivlbl, int nce, int size, int win, float rate);
        
        model(const model &) = delete;
        model& operator=(const model &) = delete;
        
    public:
        int train(const std::string &filename, int num_threads);
        int learn_vocab(const std::string &filename, int min_count);
        int save(const std::string &filename, int binary);
        
        model(int dep, int iter, int ivlbl, int nce, int size, int win, float rate);
        model();
        ~model();
    };
}

#endif

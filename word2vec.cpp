//
//  word2vec.cpp
//  word2vec
//
//  Copyright (C) 2014 Fei Jiang <f91.jiang@gmail.com>
//

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "word2vec.h"

namespace word2vec {
    
    bool compare_words_by_freq(const word_freq &w1, const word_freq &w2) {
        return w1.freq > w2.freq;
    }
    
    float model::g_init = 1e-5;
    float model::v_init_bound = 0.01;
    
    model::model(int dep, int iter, int ivlbl, int nce, int size, int win, float rate)
    {
        init(dep, iter, ivlbl, nce, size, win, rate);
    }
    
    model::model() {
        init(0, 1, 1, 10, 200, 5, 0.05);
    }
    
    void model::init(int dep, int iter, int ivlbl, int nce, int size, int win, float rate) {
        option_dependent = dep;
        option_iter_num = iter;
        option_ivlbl = ivlbl;
        option_nce = nce;
        embed_size = size;
        window = win;
        learning_rate = rate;
        
        g_in = NULL;
        g_out = NULL;
        v_in = NULL;
        v_out = NULL;
        pn = NULL;
        weight = NULL;
        g_weight = NULL;
        
        max_doc_len = 10000; // Not very important if big enough.
        option_share_inout = 0; // Input vector and output vector share the same vector is set to 1
        total_words = 0;
        trained_words = 0;
        trained_words_before = 0;
        
        weight = new float[(window * 2 + 1) * embed_size];
        for (int i = 0; i < (window * 2 + 1) * embed_size; ++i) {
            weight[i] = 1.0f;
        }
        if (option_dependent) {
            g_weight = new float[window * 2 + 1];
            for (int i = 0; i < (window * 2 + 1) * embed_size; ++i) {
                weight[i] += v_init_bound * ((float)rand() / RAND_MAX - 0.5f);
            }
            for (int i = 0; i < window * 2 + 1; ++i) {
                g_weight[i] = g_init;
            }
        }
    }
    
    model::~model() {
        if (g_in)
            delete []g_in;
        if (!option_share_inout && g_out)
            delete []g_out;
        if (v_in)
            delete []v_in;
        if (!option_share_inout && v_out)
            delete []v_out;
        if (pn)
            delete []pn;
        if (weight)
            delete []weight;
        if (g_weight)
            delete []g_weight;
    }
    
    int model::learn_vocab(const std::string &filename, int min_count) {
        std::ifstream fin(filename.c_str(), std::ios::in);
        if (!fin) {
            std::cerr << "Can not open training file " << filename << std::endl;
            return -1;
        }
        
        std::unordered_map<std::string, int> word2count;
        std::string word;
        
        while (fin >> word) {
            word2count[word] += 1;
        }
        fin.close();
        
        std::vector<word_freq> wf;
        for (auto iter = word2count.begin(); iter != word2count.end(); ++iter) {
            if (iter->second >= min_count) {
                wf.push_back(word_freq(iter->first, iter->second));
            }
        }
        
        std::sort(wf.begin(), wf.end(), compare_words_by_freq);
        
        for (int i = 0; i < wf.size(); ++i) {
            id2word.push_back(wf[i].word);
            word2id[wf[i].word] = i;
            total_words += wf[i].freq;
        }
        vocab_size = (int)id2word.size();
        
        pn = new float[vocab_size];
        for (auto iter = word2id.begin(); iter != word2id.end(); ++iter) {
            pn[iter->second] = (float)word2count[iter->first] / total_words;
        }
        
        v_in = new float[vocab_size * embed_size];
        g_in = new float[vocab_size];
        for (int i = 0; i < vocab_size * embed_size; ++i)
            v_in[i] = v_init_bound * ((float)rand() / RAND_MAX - 0.5);
        for (int i = 0; i < vocab_size; ++i)
            g_in[i] = g_init;
        if (option_share_inout) {
            v_out = v_in;
            g_out = g_in;
        } else {
            v_out = new float[vocab_size * embed_size];
            g_out = new float[vocab_size];
            for (int i = 0; i < vocab_size * embed_size; ++i)
                v_out[i] = v_init_bound * ((float)rand() / RAND_MAX - 0.5);
            for (int i = 0; i < vocab_size; ++i)
                g_out[i] = g_init;
        }
        
        init_sample_table();
        
        std::cout << "Vocabulary size = " << word2id.size() << std::endl;
        
        return vocab_size;  // Size of vocabulary
    }
    
    void model::init_sample_table() {
        int expected_table_size = 20000000; // May not be equal to the final size
        for (int i = 0; i < vocab_size; ++i) {
            int k = pn[i] * expected_table_size + 0.5;
            while (k-- > 0)
                sample_table.push_back(i);
        }
    }
    
    // Read words from corpus, the cursor should not exceed 'end' for each thread.
    // Documents are separated by '\n'.
    // To use less memory, if a document is longer than max_doc_len, it will be truncated.
    int model::read_words(std::vector<int> &words, std::ifstream &fin, long long end) {
        words.clear();
        std::string w;
        char c;
        while (words.size() < max_doc_len) {
            c = fin.get();
            if (fin.eof())
                break;
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!w.empty()) {
                    auto it = word2id.find(w);
                    if (it != word2id.end())
                        words.push_back(it->second);
                    w.clear();
                    if (words.size() > max_doc_len)
                        return 0;
                }
                if (c == '\n') {
                    return 0;
                }
            } else {
                w.push_back(c);
            }
        }
        if (fin.eof())
            return 2;
        if (fin.tellg() >= end)
            return 1;
        return 0;
    }
    
    // buffer should be large enough for computing
    // to avoid time-consuming dynamic allocation in this function, as doc can be very short
    void model::train_ivlbl(const std::vector<int> &doc, float *buffer) {
        float *neu1 = buffer,  // size = embed_size
              *neu3 = buffer + embed_size,  // size = (2 * window + 1) * embed_size
              *neu4 = neu3 + (2 * window + 1) * embed_size;   // size = (2 * window + 1) * embed_size
        
        for (int i = 0; i < doc.size(); ++i) {
            for (int k = 0; k < (2 * window + 1) * embed_size; ++k)
                neu3[k] = neu4[k] = 0;
            for (int n = 0; n < option_nce + 1; ++n) {
                // n > 0 corresponds to noise samples, n = 0 corresponds to training samples
                for (int k = 0; k < embed_size; ++k) {
                    neu1[k] = 0;
                }
                int current = n > 0 ? sample_table[rand() % sample_table.size()] : doc[i];
                for (int j = -window; j <= window; ++j) {
                    if (j == 0 || i + j < 0 || i + j >= doc.size())
                        continue;
                    
                    float s = 0;
                    for (int k = 0; k < embed_size; ++k) {
                        s += weight[(j + window) * embed_size + k] * v_out[doc[i + j] * embed_size + k] * v_in[current * embed_size + k];
                    }
                    
                    float coef;
                    
                    // for computation stability, e^x can be extremely large
                    if (s < 0) {
                        float exp_s = exp(s);
                        coef = n > 0 ? (- exp_s / (option_nce * pn[current] + exp_s)) : (option_nce * pn[current] / (option_nce * pn[current] + exp_s));
                    } else {
                        float exp_s = exp(-s);
                        coef = n > 0 ? (- 1 / (option_nce * pn[current] * exp_s + 1)) : (option_nce * pn[current] * exp_s / (option_nce * pn[current] * exp_s + 1));
                    }
                    
                    for (int k = 0; k < embed_size; ++k) {
                        neu1[k] += coef * weight[(j + window) * embed_size + k] * v_out[doc[i + j] * embed_size + k];
                        neu3[(j + window) * embed_size + k] += coef * weight[(j + window) * embed_size + k] * v_in[current * embed_size + k];
                    }
                    if (option_dependent) {
                        for (int k = 0; k < embed_size; ++k) {
                            neu4[(j + window) * embed_size + k] += coef * v_out[doc[i + j] * embed_size + k] * v_in[current * embed_size + k];
                        }
                    }
                }
                float g1 = 0, sum_grad = 0;
                for (int k = 0; k < embed_size; ++k) {
                    g1 += neu1[k] * neu1[k];
                }
                g_in[current] += g1 / embed_size;
                sum_grad = sqrtf(g_in[current]);
                for (int k = 0; k < embed_size; ++k) {
                    v_in[current * embed_size + k] += learning_rate * neu1[k] / sum_grad;
                }
            }
            
            // output vectors
            for (int j = -window; j <= window; ++j) {
                if (j == 0 || i + j < 0 || i + j >= doc.size())
                    continue;
                float g1 = 0, sum_grad = 0;
                for (int k = 0; k < embed_size; ++k) {
                    g1 += neu3[(j + window) * embed_size + k] * neu3[(j + window) * embed_size + k];
                }
                g_out[doc[i + j]] += g1 / embed_size;
                sum_grad = sqrtf(g_out[doc[i + j]]);
                for (int k = 0; k < embed_size; ++k) {
                    v_out[doc[i + j] * embed_size + k] += learning_rate * neu3[(j + window) * embed_size + k] / sum_grad;
                }
                
                // Compute the gradient of weights
                if (option_dependent) {
                    g1 = 0, sum_grad = 0;
                    for (int k = 0; k < embed_size; ++k) {
                        g1 += neu4[(j + window) * embed_size + k] * neu4[(j + window) * embed_size + k];
                    }
                    g_weight[j + window] += g1 / embed_size;
                    sum_grad = sqrtf(g_weight[j + window]);
                    for (int k = 0; k < embed_size; ++k) {
                        weight[(j + window) * embed_size + k] += learning_rate * neu4[(j + window) * embed_size + k] / sum_grad;
                    }
                }
            }
        }
    }

    // buffer should be large enough for computing
    // to avoid time-consuming dynamic allocation in this function, as doc can be very short
    void model::train_vlbl(const std::vector<int> &doc, float *buffer) {
        float *neu1 = buffer,   // size = embed_size
              *neu2 = neu1 + embed_size,  // size = embed_size
              *neu3 = neu2 + embed_size,  // size = embed_size
              *neu4 = neu3 + embed_size;  // size = (2 * window + 1) * embed_size

        for (int i = 0; i < doc.size(); ++i) {
            // Compute sum of input vectors with weight
            for (int k = 0; k < embed_size; ++k)
                neu1[k] = neu2[k] = neu3[k] = 0;
            for (int j = -window; j <= window; ++j) {
                if (j == 0 || i + j < 0 || i + j >= doc.size())
                    continue;
                for (int k = 0; k < embed_size; ++k) {
                    neu1[k] += v_in[doc[i + j] * embed_size + k] * weight[(window + j) * embed_size + k];
                    neu4[(j + window) * embed_size + k] = 0;
                }
            }
            
            // n = 0 for training sample, n > 0 for noise sample
            for (int n = 0; n < option_nce + 1; ++n) {
                float s = 0;
                int current = n > 0 ? sample_table[rand() % sample_table.size()] : doc[i];

                for (int k = 0; k < embed_size; ++k) {
                    s += v_out[current * embed_size + k] * neu1[k];
                }

                float coef;
                
                // for computation stability, e^x can be extremely large
                if (s < 0) {
                    float exp_s = exp(s);
                    coef = n > 0 ? (- exp_s / (option_nce * pn[current] + exp_s)) : (option_nce * pn[current] / (option_nce * pn[current] + exp_s));
                } else {
                    float exp_s = exp(-s);
                    coef = n > 0 ? (- 1 / (option_nce * pn[current] * exp_s + 1)) : (option_nce * pn[current] * exp_s / (option_nce * pn[current] * exp_s + 1));
                }
                
                for (int k = 0; k < embed_size; ++k) {
                    neu2[k] += v_out[current * embed_size + k] * coef;
                }
                
                if (option_dependent) {
                    for (int j = -window; j <= window; ++j) {
                        if (j == 0 || i + j < 0 || i + j >= doc.size())
                            continue;
                        for (int k = 0; k < embed_size; ++k) {
                            neu4[(j + window) * embed_size + k] += coef * v_in[doc[i + j] * embed_size + k] * v_out[current * embed_size + k];
                        }
                    }
                }

                float g1 = 0, sum_grad = 0;
                for (int k = 0; k < embed_size; ++k) {
                    neu3[k] = coef * neu1[k];
                    g1 += neu3[k] * neu3[k];
                }
                g_out[current] += g1 / embed_size;
                sum_grad = sqrtf(g_out[current]);
                for (int k = 0; k < embed_size; ++k) {
                    v_out[current * embed_size + k] += learning_rate * neu3[k] / sum_grad;
                }
            }
            
            // input vectors
            for (int j = -window; j <= window; ++j) {
                if (j == 0 || i + j < 0 || i + j >= doc.size())
                    continue;
                float g3 = 0, sum_grad = 0;
                for (int k = 0; k < embed_size; ++k) {
                    neu3[k] = neu2[k] * weight[(window + j) * embed_size + k];
                    g3 += neu3[k] * neu3[k];
                }
                g_in[doc[i + j]] += g3 / embed_size;
                sum_grad = sqrtf(g_in[doc[i + j]]);
                for (int k = 0; k < embed_size; ++k) {
                    v_in[doc[i + j] * embed_size + k] += learning_rate * neu3[k] / sum_grad;
                }
            }
            
            // Compute the gradient of weights
            if (option_dependent) {
                for (int j = -window; j <= window; ++j) {
                    if (j == 0 || i + j < 0 || i + j >= doc.size())
                        continue;
                    float g2 = 0, sum_grad;
                    for (int k = 0; k < embed_size; ++k) {
                        g2 += neu4[(j + window) * embed_size + k] * neu4[(j + window) * embed_size + k];
                    }
                    g_weight[j + window] += g2 / embed_size;
                    sum_grad = sqrtf(g_weight[j + window]);
                    for (int k = 0; k < embed_size; ++k) {
                        weight[(j + window) * embed_size + k] += learning_rate * neu4[(j + window) * embed_size + k] / sum_grad;
                    }
                }
            }
        }
    }
    
    void model::train_thread(const std::string &filename, int num_threads, int tid) {
        std::ifstream fin(filename.c_str(), std::ios::in);
        fin.seekg(0, std::ios_base::end);
        long long file_size = fin.tellg();
        long long beg = file_size / num_threads * tid, end = file_size / num_threads * (tid + 1);
        if (end > file_size)
            end = file_size;
        fin.seekg(beg, std::ios::beg);
        
        std::vector<int> doc;
        float *buffer = new float[embed_size * (2 * window + 1) * 5];
        
        int no_more_words = 0;
        clock_t tbeg = clock(), tend;
        while (!no_more_words) {
            doc.clear();
            no_more_words = read_words(doc, fin, end);
            if (option_ivlbl) {
                train_ivlbl(doc, buffer);
            } else {
                train_vlbl(doc, buffer);
            }
            trained_words += doc.size();
            if (tid == 0 && trained_words - trained_words_before >= max_doc_len) {
                tend = clock();
                
                printf("\rprogress %.2f%%, speed %.2f k/s", 100.0 * trained_words / total_words, (trained_words - trained_words_before) / ((tend - tbeg) * 1000.0 / num_threads / CLOCKS_PER_SEC));
                fflush(stdout);
                trained_words_before = trained_words;
                tbeg = tend;
            }
        }
        
        delete []buffer;
        fin.close();
    }
    
    int model::train(const std::string &filename, int num_threads) {
        if (word2id.empty()) {
            std::cerr << "No vocabulary yet, call learn_vocab method" << std::endl;
            return -1;
        }
        if (option_nce <= 0) {
            std::cerr << "nce should be positive" << std::endl;
            return -1;
        }
        
        std::ifstream fin(filename.c_str(), std::ios::in);
        if (!fin) {
            std::cerr << "Failed to open training file " << filename << std::endl;
            return -1;
        }
        fin.close();
        
        std::cout << "Start Training..." << std::endl;
        for (int iter = 0; iter < option_iter_num; ++iter) {
            std::cout << "iter " << iter << std::endl;
            std::vector<std::thread> threads;
            for(int i = 0; i < num_threads; ++i) {
                threads.push_back(std::thread(&model::train_thread, this, filename, num_threads, i));
            }
            for(int i = 0; i < num_threads; ++i) {
                threads[i].join();
            }
            std::cout << std::endl;
            trained_words = trained_words_before = 0;
        }
        return 0;
    }

    int model::save(const std::string &filename, int binary) {
        std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
        if (!fout) {
            std::cerr << "Open file " << filename << " failed." << std::endl;
            return -1;
        }
        fout << word2id.size() << " " << embed_size << std::endl;
        for (int i = 0; i < id2word.size(); ++i) {
            fout << id2word[i];
            if (!binary) {
                for (int j = i * embed_size; j < (i + 1) * embed_size; ++j) {
                    fout << " " << v_in[j];
                }
            } else {
                fout << " ";
                fout.write((char*)(v_in + i * embed_size), embed_size * sizeof(float));
            }
            fout << std::endl;
            
            if (!fout) {
                std::cerr << "Write file " << filename << " failed." << std::endl;
                return -1;
            }
        }
        fout.close();
        return 0;
    }
}
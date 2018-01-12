//
//  textReader.hpp
//  multi
//
//  Created by 任磊达 on 2017/12/15.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#ifndef textReader_hpp
#define textReader_hpp
#include "MUL_DEFINE.h"
using namespace std;

class textReader{
    // for BKDHash
    const static int SEED = 131313, MOD = 1000003, AND = 0x7FFFFFFF;
    int BKDhash(const char *str){
        int hashv = 0;
        while(*str) hashv = hashv * SEED + (*str++);
        return (hashv & AND) % MOD;
    }
    int _word_counter_file[MOD]; 
    int _trainLen[77777], _testLen[10000];
    int div_id; // id for <sss>
    
    // temp variable for reader
    FEATURE _x;
    char _word[1024];
    int _idx;
    
    
    void initialize_class(); // gen <sss> hash value
    void readTrain();
    void readTest();
    void genTFIDF();
    void genTrainValid();
    set<int> stopwords;
    void initStopwords(){
        ifstream words(stop_words.c_str());
        string w;
        while(words >> w){
            stopwords.insert(BKDhash(w.c_str()));
        }
    }
    bool isStopwords(int id){
        return stopwords.count(id);
    }
public:
    textReader();
    vector<int> validID;
    vector<FEATURE> orix;
    vector<FEATURE> trainx, validx, testx;
    vector<LABEL> trainy, validy;
    vector<LABEL> oriy;
};


#endif /* textReader_hpp */

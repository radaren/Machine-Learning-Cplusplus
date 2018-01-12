//
//  textReader.cpp
//  multi
//
//  Created by 任磊达 on 2017/12/15.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#include "textReader.hpp"
using namespace std;

textReader::textReader(){
    initStopwords();
    initialize_class();
    readTrain();
    readTest();
    genTrainValid();
};

void textReader::genTFIDF(){
    double TOT = orix.size() + testx.size();
    for(int i = 0; i < orix.size(); ++i){
        for(mit it = orix[i].begin(); it != orix[i].end(); ++it){
            it->second = (it->second / _trainLen[i]) // TF
            * (TOT / _word_counter_file[it->first]);
        }
    }
    for(int i = 0; i < testx.size(); ++i){
        for(mit it = testx[i].begin(); it != testx[i].end(); ++it){
            it->second = (it->second / _testLen[i]) // TF
            * (TOT / _word_counter_file[it->first]);
        }
    }
}

void textReader::genTrainValid(){
    trainx.clear();trainy.clear();
    validx.clear();validy.clear();
    validID.clear();
    for (int idx = 0; idx < orix.size(); ++idx) {
        if((rand()*1.0/RAND_MAX) > 0.2){
            trainx.push_back(orix[idx]);
            trainy.push_back(oriy[idx]);
        }else{
            validx.push_back(orix[idx]);
            validy.push_back(oriy[idx]);
            validID.push_back(idx);
        }
    }
}

void textReader::initialize_class()
{
    div_id = BKDhash(div_for_sentece.c_str());
}


void textReader::readTrain(){
    // read train data
    freopen(train_data.c_str(), "r", stdin);
    char c = '\0'; // reader
    for(int Line = 0; c != EOF; ++Line)
    {
        _x.clear();
        
        _idx = 0;
        c = getchar();
        LABEL y = (c == 'L') ? LOW : (c == 'M' ? MID : HIGH);
        c = getchar(); c = getchar(); // OW/ID/IG
        if((c = getchar()) == EOF) return; // tab or EOF
        c = getchar();// tab
        while((c = getchar()) != EOF)
        {
            if(c == '\n') break;
            if(c == ' '){
                _word[_idx] = 0;
                int id = BKDhash(_word);
                _idx = 0;
                if(id == div_id){
                    continue; // split symbol for different sentence.
                }
                if(!isStopwords(id))
                    _x[id] = _x[id] + 1;
            }else{
                _word[_idx] = c;
                ++_idx;
            }
        }
        orix.push_back(_x);
        oriy.push_back(y);
    }
}

void textReader::readTest(){
    // read train data
    freopen(test_data.c_str(), "r", stdin);
    char c = '\0'; // reader
    for(int Line = 0; c != EOF; ++Line)
    {
        _x.clear();
        
        _idx = 0;
        c = getchar(); // ?
        if((c = getchar()) == EOF) return; // tab or EOF
        c = getchar();// tab
        while((c = getchar()) != EOF)
        {
            if(c == '\n') break;
            if(c == ' '){
                _word[_idx] = 0;
                if(_word[0] >= 'a' && _word[0] <= 'z'){
                    int id = BKDhash(_word);
                    if(id == div_id){
                        continue; // split symbol for different sentence.
                    }
                    if(!isStopwords(id))
                        _x[id] = _x[id] + 1;
                }
                _idx = 0;
            }else{
                _word[_idx] = c;
                ++_idx;
            }
        }
        testx.push_back(_x);
    }
}

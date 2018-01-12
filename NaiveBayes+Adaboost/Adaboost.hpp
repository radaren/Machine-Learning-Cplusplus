//
//  Adaboost.hpp
//  multi
//
//  Created by 任磊达 on 2018/1/7.
//  Copyright © 2018年 任磊达. All rights reserved.
//

#ifndef Adaboost_hpp
#define Adaboost_hpp

#include "MUL_DEFINE.h"
using namespace std;

struct bayes{
    bayes(){idx.clear();x.clear();y.clear();D = 0;};
    vector<int> idx; // of Adaboost
    vector<double> w;
    vector<FEATURE> x;
    vector<LABEL>y;
    // train model
    unordered_map<int, double> NW;
    unordered_map<int, double> NEW[MAX_LABEL];
    double NE[MAX_LABEL];
    double D, err, upd;
    
    void calcuD();
    void gen_ne_new();
    double predict(FEATURE& x, LABEL y);
    LABEL predict(FEATURE& x);
    void err_calcu();
    vector<bool> pred;
};
class Adaboost{
    friend class bayes;
private:
    vector<bayes> classifiers;
    vector<FEATURE> trainx;
    vector<LABEL> trainy;
    vector<double> globalWeight;
public:
    void outputweightLabel(string str, vector<FEATURE> testx, vector<LABEL> testy){
        ofstream of(str);
        for(int i = 0; i < testx.size(); ++i){
            double sum[3] = {0,0,0};
            for(int idx = 0; idx < classifiers.size(); ++idx){
                sum[LOW] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], LOW);
                sum[MID] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], MID);
                sum[HIGH] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], HIGH);
            }
            of << sum[LOW] << "," << sum[MID] << "," << sum[HIGH] << "," << testy[i] << endl;
        }
        of.close();
    }
    void outputweight(string str, vector<FEATURE> testx){
        ofstream of(str);
        for(int i = 0; i < testx.size(); ++i){
            double sum[3] = {0,0,0};
            for(int idx = 0; idx < classifiers.size(); ++idx){
                sum[LOW] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], LOW);
                sum[MID] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], MID);
                sum[HIGH] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], HIGH);
            }
            of << sum[LOW] << "," << sum[MID] << "," << sum[HIGH] << endl;
        }
        of.close();
    }
public:
    void fit(vector<FEATURE> trainx, vector<LABEL> trainy
             , int totC = 20, double sampleRate = 0.3);
    vector<LABEL> predict(vector<FEATURE> testx);
};

#endif /* Adaboost_hpp */

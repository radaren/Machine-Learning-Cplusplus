//
//  NBAB.cpp
//  multi
//
//  Created by 任磊达 on 2017/12/16.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#include "NBAB.hpp"

void NBAB::fit(vector<FEATURE> trainx, vector<LABEL> trainy
               , int C, int iter_step){
    
    classifiers.clear();
    //double D = 10;
    //double sampleRate = C > D ? D / C : 0.2;
    double sampleRate = 0.25;
    
    cerr << "At initial " << ":\n";
    for(int i = 0; i < C; ++i){
        vector<double> weight;
        vector<FEATURE> x;
        vector<LABEL> y;
        for(int i = 0; i < trainx.size(); ++i){
            if(rand() * 1.0 / RAND_MAX < sampleRate){
                weight.push_back(2.0 * (rand()*1.0/RAND_MAX));
                x.push_back(trainx[i]);
                y.push_back(trainy[i]);
            }
        }
        classifiers.push_back(nb(x, y, weight));
#ifdef SHOW_DETAIL_ERR
        cerr << classifiers.size()-1
            << "\terr = " << classifiers[classifiers.size()-1].err
            << "\tupd = "<< classifiers[classifiers.size()-1].upd << endl;
#endif
    }
#ifdef SHOW_DETAIL_ERR
    cerr << endl;
#endif
    // training
    for(int iter = 0; iter < iter_step; ++iter){
        //cerr << "At step " << iter << ":\n";
        for(int idx = 0; idx < classifiers.size(); ++idx){
            classifiers[idx].train_step();
#ifdef SHOW_DETAIL_ERR
            cerr << idx
            << "\terr = " << classifiers[idx].err
            << "\tupd = " << classifiers[idx].upd << endl;
#endif
        }
#ifdef SHOW_DETAIL_ERR
        cerr << endl;
#endif
    }
    // bagging, recover the best state
#ifdef SHOW_DETAIL_ERR
    cerr << "Bagging, recover best" << endl;
#endif
    for(int idx = 0; idx < classifiers.size(); ++idx){
        classifiers[idx].recoverBest();
#ifdef SHOW_DETAIL_ERR
        cerr << idx << "\terr = " << classifiers[idx].getBestErr() << endl;
#endif
    }
}


vector<LABEL> NBAB::predict(vector<FEATURE> testx){
    vector<double> low, mid, high;
    vector<LABEL> predict;
    for(int i = 0; i < testx.size(); ++i){
        double sum[3] = {0,0,0};
        for(int idx = 0; idx < classifiers.size(); ++idx){
            sum[LOW] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], LOW);
            sum[MID] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], MID);
            sum[HIGH] += log(classifiers[idx].upd) * classifiers[idx].predict(testx[i], HIGH);
        }
        predict.push_back((sum[LOW] >= sum[MID] && sum[LOW] >= sum[HIGH])? LOW :
                          (sum[MID] >= sum[HIGH])? MID :HIGH);
    }
    return predict;
}


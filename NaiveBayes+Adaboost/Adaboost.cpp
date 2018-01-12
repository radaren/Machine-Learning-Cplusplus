//
//  Adaboost.cpp
//  multi
//
//  Created by 任磊达 on 2018/1/7.
//  Copyright © 2018年 任磊达. All rights reserved.
//

#include "Adaboost.hpp"
void bayes::gen_ne_new(){
    // NE & NEW
    NE[LOW] = NE[MID] = NE[HIGH] = 0;
    NEW[LOW].clear(); NEW[MID].clear(); NEW[HIGH].clear();
    for(int i = 0; i < x.size(); ++i){
        for(mit j = x[i].begin(); j != x[i].end(); ++j){
            NEW[y[i]][j->first] += j->second * w[i];
            NW[j->first] += j->second * w[i];
        }
        NE[y[i]] += w[i];
    }
}
void bayes::calcuD(){
    set<int> FEATUREID;
    for (int i = 0; i < x.size(); ++i) {
        for(mit j = x[i].begin(); j != x[i].end(); ++j){
            FEATUREID.insert(j->first);
        }
    }
    D = FEATUREID.size();
}
void bayes::err_calcu(){
    // get pred vector, err, s
    double tot = 0;
    err = 0;
    pred.clear();
    for (int i = 0; i < x.size(); ++i) {
        LABEL p = predict(x[i]);
        err += w[i] * (p != y[i]);
        tot += w[i];
        pred.push_back(p == y[i]);
    }
    err /= tot;
    upd = ((1 - err) / err) * (2);
}
double bayes::predict(FEATURE& x, LABEL y){
    // calcu log(NE/D) - tlog[N(E) + alpha * D] + \sum[log(N(E, W) + alpha)]
    double pred = log(NE[y] / D);
    for (mit i = x.begin(); i != x.end(); ++i) {
        pred += i->second * (log(NEW[y][i->first] + 1e-5) - log(NW[i->first] + 1e-5 * D) );
    }
    return pred;
}
LABEL bayes::predict(FEATURE& x){
    double lowPos = predict(x, LOW);
    double midPos = predict(x, MID);
    double highPos= predict(x, HIGH);
    
    return (lowPos >= midPos && lowPos >= highPos) ? LOW :
    midPos >= highPos ? MID : HIGH;
}

void Adaboost::fit(vector<FEATURE> xx, vector<LABEL> yy, int iter_num, double sampleRate){
    trainx = xx;
    trainy = yy;
    globalWeight.resize(xx.size(), 1.0);
    classifiers.clear();
    for (int I = 0; I < iter_num; ++I) {
        bayes CF;
        // sample datas
        for(int i = 0; i < trainx.size(); ++i){
            if(rand()*1.0/RAND_MAX < sampleRate){
                CF.idx.push_back(i);
                CF.w.push_back(globalWeight[i]); // current global weight
                CF.x.push_back(trainx[i]);
                CF.y.push_back(trainy[i]);
            }
        }
        // calcu error
        CF.calcuD();
        CF.gen_ne_new();
        CF.err_calcu();
        // update global weight
        for (int i = 0; i < CF.idx.size(); ++i) {
            int now = CF.idx[i];
            if(CF.pred[i]){
                globalWeight[now] /= CF.upd;
            }else{
                globalWeight[now] *= CF.upd;
            }
        }
        // normalize w
        double avg = 0, std = 0;
        for (int i = 0; i < globalWeight.size(); ++i) {
            avg += globalWeight[i];
        }
        avg /= globalWeight.size();
        for (int i = 0; i < globalWeight.size(); ++i) {
            std += (globalWeight[i] - avg) * (globalWeight[i] - avg);
        }
        std = sqrt(std / globalWeight.size());
        for (int i = 0; i < globalWeight.size(); ++i) {
            globalWeight[i] = globalWeight[i] / std;
        }
        // over training step
        classifiers.push_back(CF);
        printf("Iter %d, Error %lf\n", I, CF.err);
    }
}
vector<LABEL> Adaboost::predict(vector<FEATURE> testx){
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


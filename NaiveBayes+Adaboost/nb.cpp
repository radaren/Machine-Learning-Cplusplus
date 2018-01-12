//
//  nb.cpp
//  multi
//
//  Created by 任磊达 on 2017/12/16.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#include "nb.hpp"
void nb::update_weight(){
    double avgw = 0, stdw = 0;
    for (int i = 0; i < weight.size(); ++i) {
        if(pred[i] == y[i]){
           
        }else{
            weight[i] *= upd;
        }
        avgw += weight[i];
    }
    // renormalization w
    avgw /= weight.size();
    for (int i = 0; i < weight.size(); ++i) {
        stdw += (weight[i] - avgw) * (weight[i] - avgw);
    }
    stdw = sqrt(stdw / weight.size());
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] /= stdw;
    }
}
void nb::gen_ne_new(){
    // NE & NEW
    NE[LOW] = NE[MID] = NE[HIGH] = 0;
    NEW[LOW].clear(); NEW[MID].clear(); NEW[HIGH].clear();
    for(int i = 0; i < x.size(); ++i){
        for(mit j = x[i].begin(); j != x[i].end(); ++j){
            NEW[y[i]][j->first] += j->second * weight[i];
        }
        NE[y[i]] += weight[i];
        D += NE[y[i]];
    }
}

void nb::err_calcu(){
    // get pred vector, err, s
    double tot = 0;
    err = 0;
    pred.clear();
    for (int i = 0; i < x.size(); ++i) {
        LABEL p = predict(x[i]);
        err += weight[i] * (p != y[i]);
        tot += weight[i];
        pred.push_back(p);
    }
    err /= tot;
    upd = (1 - err) / err;
    if(err < bestErr){
        bestErr = err;
        bestWeight = weight;
    }
}

void nb::train_step(){
    update_weight();
    gen_ne_new();
    err_calcu();
}

void nb::recoverBest(){
    weight = bestWeight;
    gen_ne_new();
    err_calcu();
}
double nb::predict(FEATURE& x, LABEL y){
    // calcu log(NE/D) - tlog[N(E) + alpha * D] + \sum[log(N(E, W) + alpha)]
    double pred = log(NE[y] / D);
    for (mit i = x.begin(); i != x.end(); ++i) {
        pred += i->second * (log(NEW[y][i->first] + alpha) - log(NE[y] + alpha * D));
    }
    return pred;
}
LABEL nb::predict(FEATURE& x){
    double lowPos = predict(x, LOW);
    double midPos = predict(x, MID);
    double highPos= predict(x, HIGH);
    
    return (lowPos >= midPos && lowPos >= highPos) ? LOW :
    midPos >= highPos ? MID : HIGH;
}



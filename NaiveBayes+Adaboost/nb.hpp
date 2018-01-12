//  sub class of NBAB
//  nb.hpp
//  multi
//
//  Created by 任磊达 on 2017/12/16.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#ifndef nb_hpp
#define nb_hpp

#include "MUL_DEFINE.h"
using namespace std;

class nb{
    
private:
    vector<double> weight;
    vector<FEATURE> x;
    vector<LABEL> y, pred;
    double alpha;// not adaboost, alpha for nb
    
    // train model
    unordered_map<int, double> NW;
    unordered_map<int, double> NEW[MAX_LABEL];
    double NE[MAX_LABEL];
    double D;
    
   
    // funciton to update
    void update_weight();
    void gen_ne_new();
    void err_calcu();
    
    // record the best weight
    double bestErr;
    vector<double> bestWeight;
public:
    double err, upd; // s = sqrt((1-e)/e)
    nb(vector<FEATURE> x, vector<LABEL> y, vector<double> weight
            ,double alpha = 1e-5)
    :x(x), y(y), weight(weight), alpha(alpha){
        bestErr = 1;
        gen_ne_new();
        err_calcu();
    };
    void train_step();
    
    void recoverBest();
    double predict(FEATURE& x, LABEL);
    LABEL predict(FEATURE& x);
    double getBestErr(){return bestErr;};
};
#endif /* nb_hpp */

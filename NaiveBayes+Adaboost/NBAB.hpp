//
//  NBAB.hpp
//  multi
//
//  Created by 任磊达 on 2017/12/16.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#ifndef NBAB_hpp
#define NBAB_hpp

#include "nb.hpp"
#include "MUL_DEFINE.h"
using namespace std;

class NBAB{
private:
    vector<nb> classifiers;
public:
    void fit(vector<FEATURE> trainx, vector<LABEL> trainy
             , int totC = 20, int iter_step = 10);
    vector<LABEL> predict(vector<FEATURE> testx);
};
#endif /* NBAB_hpp */

//
//  MUL_DEFINE.h
//  multi
//
//  Created by 任磊达 on 2017/12/15.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#ifndef MUL_DEFINE_h
#define MUL_DEFINE_h
#include <set>
#include <unordered_map> // speed us 1s in read
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;
#define SHOW_DETAIL_ERR
const string stop_words ="/Users/radar/Desktop/Project/multi-classification/sw.txt";
const string train_data  = "/Users/radar/Desktop/Project/multi-classification/MulLabelTrain.ss";
const string test_data   = "/Users/radar/Desktop/Project/multi-classification/MulLabelTest.ss";
const string div_for_sentece = "<sssss>";

const double e = exp(1);

enum LABEL{
    LOW = 0,
    MID = 1,
    HIGH = 2
};
const int MAX_LABEL = 3;

typedef unordered_map<int, double> FEATURE;
typedef unordered_map<int , double>::iterator mit;
#endif /* MUL_DEFINE_h */

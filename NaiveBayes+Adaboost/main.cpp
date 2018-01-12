// TODO1: KMeans
// TODO2: NB



//  main.cpp
//  multi
//
//  Created by 任磊达 on 2017/12/9.
//  Copyright © 2017年 任磊达. All rights reserved.
//

#include "textReader.hpp"
#include "NBAB.hpp"
#include "Adaboost.hpp"
using namespace std;
// timer
static clock_t beginTime, lastTime, nowTime;
void startProgramTimer()
{
    beginTime = clock();
    lastTime = beginTime;
}

void timeCounter(const char* pid)
{
    nowTime = clock();
    cerr << "Period [" << pid << "],\ttime cost "
    << double(nowTime - lastTime)/ CLOCKS_PER_SEC << "s,\ttot time cost "
    << double(nowTime - beginTime)/ CLOCKS_PER_SEC << "s\n" << endl;
    lastTime = nowTime;
}

string result = "/Users/radar/Desktop/Project/multi-classification/58_v2.csv";

void validation(textReader& text){
    NBAB classifier;
    
    classifier.fit(text.trainx, text.trainy, 35, 5);
    timeCounter("Train Data");
    vector<LABEL> predict = classifier.predict(text.validx);
    double accuracy = 0;
#ifdef SHOW_DETAIL_ERR
    int totLOW = 0;
    int totMID = 0;
    int totHIG = 0;
    //freopen(err_class.c_str(), "w", stdout);
    printf("predict, accurate, SentenceID\n");
#endif
    for(int i = 0; i < predict.size(); ++i){
        accuracy += predict[i] == text.validy[i];
#ifdef SHOW_DETAIL_ERR
        totLOW += predict[i] == LOW;
        totMID += predict[i] == MID;
        totHIG += predict[i] == HIGH;
        if(predict[i] != text.validy[i]){
            if(predict[i] == LOW) printf("LOW,");
            else if(predict[i] == MID) printf("MID,");
            else if(predict[i] == HIGH) printf("HIG,");
            if(text.validy[i] == LOW) printf("LOW,");
            else if(text.validy[i] == MID) printf("MID,");
            else if(text.validy[i] == HIGH) printf("HIG,");
            printf("%d\n", text.validID[i]);
        }
#endif
    }
    accuracy /= predict.size();
    cerr << "Accuracy in valid = " << accuracy << endl;
#ifdef SHOW_DETAIL_ERR
    cerr << totLOW << "/" << totMID << "/" << totHIG << endl;
    timeCounter("Validation Data");
    predict = classifier.predict(text.testx);
    freopen(result.c_str(), "w", stdout);
    for(int i = 0; i < predict.size(); ++i){
        if(predict[i] == LOW) printf("LOW\n");
        else if(predict[i] == MID) printf("MID\n");
        else if(predict[i] == HIGH) printf("HIG\n");
    }
#endif
}


void test(textReader& text){
    NBAB classifier;
    classifier.fit(text.orix, text.oriy, 100, 10);
    vector<LABEL> predict = classifier.predict(text.testx);
    freopen(result.c_str(), "w", stdout);
    for(int i = 0; i < predict.size(); ++i){
        if(predict[i] == LOW) printf("LOW\n");
        else if(predict[i] == MID) printf("MID\n");
        else if(predict[i] == HIGH) printf("HIG\n");
    }
}

void calIndex(const vector<LABEL>& predict, const vector<LABEL> real)
{
    double accuracy = 0;
    int totLOW = 0;
    int totMID = 0;
    int totHIG = 0;
    for(int i = 0; i < predict.size(); ++i){
        accuracy += predict[i] == real[i];
        totLOW += predict[i] == LOW;
        totMID += predict[i] == MID;
        totHIG += predict[i] == HIGH;
    }
    accuracy /= predict.size();
    cerr << "Accuracy in valid = " << accuracy << endl;
    cerr << totLOW << "/" << totMID << "/" << totHIG << endl;
    timeCounter("Validation Data");
}
void Adaboost_(textReader& text){
    Adaboost ad;
    ad.fit(text.trainx, text.trainy, 128, 0.25);
    timeCounter("Train Data");
    vector<LABEL> predict = ad.predict(text.validx);
    calIndex(predict, text.validy);
    
    predict = ad.predict(text.testx);
    freopen(result.c_str(), "w", stdout);
    for(int i = 0; i < predict.size(); ++i){
        if(predict[i] == LOW) printf("LOW\n");
        else if(predict[i] == MID) printf("MID\n");
        else if(predict[i] == HIGH) printf("HIG\n");
    }
    timeCounter("Test Data");
    
}
void Adaboost_output(textReader& text){
    Adaboost ad;
    ad.fit(text.trainx, text.trainy, 128, 0.21);
    timeCounter("Train Data");
    vector<LABEL> predict = ad.predict(text.validx);
    calIndex(predict, text.validy);
    
    predict = ad.predict(text.testx);
    // todo: add sqrt, delete one of */
    freopen(result.c_str(), "w", stdout);
    for(int i = 0; i < predict.size(); ++i){
        if(predict[i] == LOW) printf("LOW\n");
        else if(predict[i] == MID) printf("MID\n");
        else if(predict[i] == HIGH) printf("HIG\n");
    }
    timeCounter("Test Data");
}
int main(int argc, const char * argv[]) {
    startProgramTimer();
    textReader text;
    timeCounter("Read Data");
    //validation(text);
    Adaboost_output(text);
    //timeCounter("End of Program");
    return 0;
}

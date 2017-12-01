//
//  main.cpp
//  LR
//
//  Created by 任磊达 on 2017/11/19.
//  Copyright © 2017年 任磊达. All rights reserved.
//
#include <set>
#include <map>
#include <cmath>
#include <bitset>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

// 文件地址
const string train_data     = "/Users/radar/desktop/Lab5_LR/train.csv";
const string test_data      = "/Users/radar/desktop/Lab5_LR/test.csv";
const string result         = "/Users/radar/desktop/Lab5_LR/result.csv";
const string drawer         = "/Users/radar/desktop/drawer.csv";

typedef vector<double>  FEATURE, WEIGHT;
typedef int             LABEL;

// 回归类
class LogisticRegression
{
public:
    LogisticRegression()
    {
        oriReadFlag=testReadFlag=false;
        genFlag=false; // 在test阶段，初始化时候把genFlag设为true
        // [TODO: w 初始化]
        bestAccu = 0;
    };
    
    void    train(int OptimizerFlag=0);
    double  validate();
    void  test();
    
private:
    
    vector<pair<FEATURE, LABEL> > oridata, trainData,validData, testdata;
    bool  oriReadFlag, testReadFlag, genFlag;
    void  ReadOriginalData();
    void  ReadTestData();
    bool  ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin);
    bool  ReadDataWithQuery(FEATURE &feature, ifstream & fin);
    
    void  GenTrain_Valid(double alpha = 0.2);
    int   predict(const vector<double> & data);
    void  StandalizeData();
    
    double bestAccu;
    WEIGHT w, delta, bestw, avg, sr;
    void  initZeroWeight(WEIGHT &w);
    void  initRandWeight();
};
enum{
    Bagging = 0,
    SGD = 1,
    Standalize = 2,
    Normalize  = 3, // L2 Normalize item
    DynamicLearnRate = 4, // Adam solution
};
ofstream dr(drawer);

int main(int argc, const char * argv[]) {
    LogisticRegression LR;
    int OptimizerFlag = 0;
    OptimizerFlag |= (1<<Bagging);
    //OptimizerFlag |= (1<<SGD);
    //OptimizerFlag |= (1<<Standalize);
    //OptimizerFlag |= (1 << Normalize);
    OptimizerFlag |= (1<<DynamicLearnRate);
    LR.train(OptimizerFlag);
    LR.test();
    return 0;
}


void LogisticRegression::train(int OptimizerFlag)
{
    if(oriReadFlag == false) ReadOriginalData();
    if(testReadFlag== false) ReadTestData();
    if(genFlag == false)     GenTrain_Valid();
    
    //标准化
    if(OptimizerFlag & (1 << Standalize)) StandalizeData();
    
    initZeroWeight(w);       // 初始化方式
    double alpha = 0.00001; // 学习率
    double theta = bool(OptimizerFlag & (1 << Normalize))? 1e-4 : 0;        // 正则化项系数
    double u = 0.9, v = 0.999, m = 0, n = 0, m_ = 0, n_ = 0, ep = 1e-8; // adam系数
    dr << "iteration,Accuracy,Precision,Recall,F1,loss" << endl;
    for (int iter = 1; iter <= 100; ++iter) {
        initZeroWeight(delta);
        // Normalizer
        double loss_normal= 0;
        if(OptimizerFlag & (1 << Normalize))
            for(int wi = 0; wi < w.size(); ++wi)
            {
                loss_normal += theta * w[wi] * w[wi];
            }
        double loss = loss_normal;
        // 分两步为普通梯度下降，一个for为SGD
        for(int F = 0; F < trainData.size(); ++F)
        {
            double wtxi = 0;
            for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
                wtxi += w[idx] * trainData[F].first[idx];
            }
            double hxi = 1.0 / (1 + exp(-wtxi));
            if(hxi < 1 && hxi > 0)
            loss -= trainData[F].second * log(hxi) + (1 - trainData[F].second) * log(1-hxi);
            if(OptimizerFlag & (1 << SGD))
            {
                // 随机梯度下降
                for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
                    w[idx] -= alpha * (exp(wtxi)/ (1 + exp(wtxi)) - trainData[F].second) * trainData[F].first[idx];
                }
            }
            else
            {
                for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
                    delta[idx] += (exp(wtxi)/ (1 + exp(wtxi)) - trainData[F].second) * trainData[F].first[idx];
                }
            }
        }
        if((OptimizerFlag & (1 << SGD)) == 0)
        {
            if(OptimizerFlag & (1 << DynamicLearnRate))
            {
                m = u * m + (1 - u) * loss;
                n = v * n + (1 - v) * loss * loss;
                m_ = m / (1 - pow(u, iter));
                n_ = n / (1 - pow(v, iter));
                double delta_ = (m_ / (sqrt(n_) + ep)) * alpha;
                cerr << delta_ << endl;
                for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
                    w[idx] -= delta_ * delta[idx] - 2 * theta * w[idx];
                }
            }
            else{
                for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
                    w[idx] -= alpha * delta[idx] - 2 * theta * w[idx];
                }
            }
        }
        
        dr << iter << ",";
        cerr << "ITER " << iter << ": (" ;
        for(int i = 0; i < w.size(); ++i) cerr << w[i] << ","; cerr <<")";
        double Accuracy = validate();
        dr << loss << endl;
        // TODO : 绘制参数图。
        // bagging
        if((OptimizerFlag&(1<<Bagging)) && Accuracy > bestAccu)
        {
            bestAccu = Accuracy;
            bestw = w;
        }
    }
}
// return Accuracy
double LogisticRegression::validate()
{
    double Accuracy, Precision, Recall, F1;
    int TP = 0, FN = 0, TN = 0, FP = 0;
    for(int i = 0; i < validData.size(); ++i)
    {
        int pred = predict(validData[i].first);
        int label= validData[i].second;
        if(pred == 1 && label == 1) TP++;
        else if(pred == 1 && label == 0) FP++;
        else if(pred == 0 && label == 0) TN++;
        else FN++;// care for zero situation.
    }
    Accuracy  = (TP + TN) * 1.0 / (TP + FP + TN + FN);
    Precision = (TP + FP) ? TP * 1.0 / (TP + FP):0;
    Recall    = (TP + FN) ? TP * 1.0 / (TP + FN):0;
    F1        = (Precision + Recall) ? 2 * Precision * Recall / (Precision + Recall):0;
    printf("\n\nvalidate result(infinity depth) :\nTP(%d),TN(%d),FP(%d),FN(%d),\naccu(%f), prec(%f), recall(%f), f1(%f)\n\n\n",
           TP,TN,FP,FN,Accuracy, Precision, Recall, F1);
    dr << Accuracy << "," << Precision << "," << Recall << "," << F1 <<",";
    return Accuracy;
}
void LogisticRegression::initZeroWeight(WEIGHT &w)
{
    w.clear();
    for(int i = 0; i < oridata[0].first.size(); ++i)
        w.push_back(0);
}
void LogisticRegression::initRandWeight()
{
    w.clear();
    for(int i = 0; i < oridata[0].first.size(); ++i)
        w.push_back(rand()*1.0/RAND_MAX);
}
void LogisticRegression::test()
{
    if(bestw.size()) w = bestw;
    ofstream re(result);
    for(int i = 0; i < testdata.size(); ++i)
    {
        re << predict(testdata[i].first) << endl;
    }
}
int LogisticRegression::predict(const FEATURE & data)
{
    if(w.size()==0) return 0;
    double sum = 0;
    for(int idx = 0; idx < data.size(); ++idx)
    {
        sum += w[idx] * data[idx];
    }
    return sum > 0.5; // thread
}
void LogisticRegression::GenTrain_Valid(double alpha)
{
    // new dataset.
    trainData.clear();
    validData.clear();
    // 20% percent to valid
    for(int dat = 0; dat < (int)oridata.size(); ++dat)
    {
        if(rand()*1.0/RAND_MAX < alpha)
        {
            validData.push_back(oridata[dat]);
        }
        else
        {
            trainData.push_back(oridata[dat]);
        }
    }
    cout << "Divide train(" << trainData.size() << "), valid(" << validData.size() << ")\n";
}

void LogisticRegression::StandalizeData()
{
    // 使用avg向量统计训练集上平均值，sr统计方差
    initZeroWeight(avg);
    initZeroWeight(sr);
    for(int F = 0; F < trainData.size(); ++F)
    {
        for (int idx = 1; idx < oridata[0].first.size(); ++idx) {
            avg[idx] += trainData[F].first[idx] / trainData.size();
        }
    }
    for(int F = 0; F < trainData.size(); ++F)
    {
        for (int idx = 1; idx < oridata[0].first.size(); ++idx) {
            sr[idx] += (trainData[F].first[idx] - avg[idx])
            *(trainData[F].first[idx] - avg[idx]) / trainData.size();
        }
    }
    for (int idx = 0; idx < oridata[0].first.size(); ++idx) {
        sr[idx] = sqrt(sr[idx]);
    }
    for(int F = 0; F < trainData.size(); ++F)
    {
        for (int idx = 1; idx < oridata[0].first.size(); ++idx) {
            if(sr[idx])
                trainData[F].first[idx] /= sr[idx];
        }
    }
    for(int F = 0; F < validData.size(); ++F)
    {
        for (int idx = 1; idx < oridata[0].first.size(); ++idx) {
            if(sr[idx])
                validData[F].first[idx] /= sr[idx];
        }
    }
    for(int F = 0; F < testdata.size(); ++F)
    {
        for (int idx = 1; idx < oridata[0].first.size(); ++idx) {
            if(sr[idx])
                testdata[F].first[idx] /= sr[idx];
        }
    }
}
/* readers */
void  LogisticRegression::ReadOriginalData()
{
    oriReadFlag = true;
    FEATURE Fe;
    LABEL   La;
    ifstream traini(train_data.c_str(), std::ifstream::in);
    oridata.clear();
    while(ReadDataWithLabel(Fe, La, traini))
    {
        oridata.push_back(make_pair(Fe,La));
    }
    cout << "Reading(ori):" << oridata.size() << "*" << oridata[0].first.size() << endl;
}
void LogisticRegression::ReadTestData()
{
    testReadFlag = true;
    FEATURE Fe;
    ifstream testi(test_data.c_str(), std::ifstream::in);
    testdata.clear();
    while(ReadDataWithQuery(Fe, testi))
    {
        testdata.push_back(make_pair(Fe, -1));
    }
    cout << "Reading(Test):" << testdata.size() << "*" << testdata[0].first.size() << endl;
    
}

bool LogisticRegression::ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin)
{
    string TestLine;
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;
        
        feature.clear();
        feature.push_back(1); // x0
        
        Ldir = Rdir = 0;
        
        while(Rdir < TestLine.size())
        {
            if(TestLine[Rdir] == ',')
            {
                Fea = atof(TestLine.substr(Ldir, Rdir - Ldir).c_str());
                feature.push_back(Fea);
                Ldir = Rdir + 1;
            }
            ++Rdir;
        }
        label = atof(TestLine.substr(Ldir, Rdir - Ldir).c_str());
        return true;
    }
    else
        return false;// read the end of document
}

bool LogisticRegression::ReadDataWithQuery(FEATURE &feature, ifstream & fin)
{
    string TestLine;
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;
        
        feature.clear();
        feature.push_back(1); // x0
        
        Ldir = Rdir = 0;
        while(Rdir < TestLine.size())
        {
            if(TestLine[Rdir] == ',')
            {
                Fea = atof(TestLine.substr(Ldir, Rdir - Ldir).c_str());
                feature.push_back(Fea);
                Ldir = Rdir + 1;
            }
            ++Rdir;
        }
        // with ? add '//'
        // Fea = atof(TestLine.substr(Ldir, Rdir - Ldir).c_str());
        // feature.push_back(Fea);
        return true;
    }
    else
        return false;// read the end of document
}

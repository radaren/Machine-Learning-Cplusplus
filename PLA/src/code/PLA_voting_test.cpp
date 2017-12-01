/**
 * created by Ren Leida
 * Copyright (c) 2017, leidar100@gmail.com All Rights Reserved.
 * *                #                                                   #
 * #                       _oo0oo_                     #
 * #                      o8888888o                    #
 * #                      88" . "88                    #
 * #                      (| -_- |)                    #
 * #                      0\  =  /0                    #
 * #                    ___/`---'\___                  #
 * #                  .' \\|     |# '.                 #
 * #                 / \\|||  :  |||# \                #
 * #                / _||||| -:- |||||- \              #
 * #               |   | \\\  -  #/ |   |              #
 * #               | \_|  ''\---/''  |_/ |             #
 * #               \  .-\__  '-'  ___/-. /             #
 * #             ___'. .'  /--.--\  `. .'___           #
 * #          ."" '<  `.___\_<|>_/___.' >' "".         #
 * #         | | :  `- \`.;`\ _ /`;.`/ - ` : | |       #
 * #         \  \ `_.   \_ __\ /__ _/   .-` /  /       #
 * #     =====`-.____`.___ \_____/___.-`___.-'=====    #
 * #                       `=---='                     #
 * #     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   #
 * #                                                   #
 * #               佛祖保佑         永无BUG              #
 * #                                                   #
 */
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;

// input
const string train_data = "../data/train.csv";
const string vali_data = "../data/val.csv";
const string test_data = "../data/test.csv";
// output
const string test_output = "../test_ans.csv";

typedef vector<double>  FEATURE;
typedef int             LABEL; 


class PLA{
public:
    void                voting();
    void                init();
private:
    vector<FEATURE>     TrainFeature;
    vector<FEATURE>     TestFeature;
    
    vector<double>      Weight, MaxWeight;

    vector<LABEL>       TrainLabel;

    int                 trainStep(double alpha = 1); 
   
    void                initAvgWeight();
    void                generateFeature(FEATURE &feature);
    bool                ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin);
    bool                ReadDataWithQuery(FEATURE &feature, ifstream & fin);

    void                ReadTrainData();
    void                ReadTestData();

    string              TestLine;
}MyPLA;

int main(int argc, char** argv)
{
    MyPLA.init();
    MyPLA.voting();
    return 0;
}

void PLA::init()
{
    TrainFeature.clear();
    TestFeature.clear();
    TrainLabel.clear();
}

void PLA::voting()
{
    ReadTrainData();
    double Accuracy, Precision, Recall, F1;
    ReadTestData();
    ofstream v("test.csv",std::ofstream::out);
    vector<double> voting_pool;
    for (int i = 0; i < TestFeature.size(); ++i)
    {
        voting_pool.push_back(0);
    }
    for(int cell = 1; cell <= 65; ++cell)
    {
        initAvgWeight();
        int iter = 0;
        for(iter = 0; iter < 100; ++iter)
            trainStep(1.0);
        for(int val = 0;val < TestFeature.size(); ++val)
        {
            double sum = 0;
            for(int k = 0; k < TestFeature[val].size(); ++k)
            {
                sum += TestFeature[val][k] * Weight[k];
            }
            voting_pool[val] += (sum>0)?2:-1;
        }
    }
    for(int val = 0;val < TestFeature.size(); ++val)
    {
        v << (voting_pool[val] > 0 ? 1 : -1) << endl;
    }
}

int PLA::trainStep(double alpha)
{
    int ErrNum = 0;
    for(int i = 0; i < TrainFeature.size(); ++i)
    {
        double sum = 0;
        for(int k = 0; k < TrainFeature[i].size(); ++k)
        {
            sum += TrainFeature[i][k] * Weight[k];
        }
        if(sum * TrainLabel[i] <= 0)
        {
            for(int k = 0; k < TrainFeature[i].size(); ++k)
            {
                // dropout
                if(rand()*1.0/RAND_MAX< 0.5)
                    Weight[k] += alpha * TrainLabel[i] * TrainFeature[i][k];
            }
            ++ErrNum;
        }// incorrect sample, update.
    }
    return ErrNum;
}

void PLA::initAvgWeight()
{
    Weight.clear();
    for(int fea = 0; fea < TrainFeature[0].size(); ++fea)
    {
        double avg_sum = 0, delta_sum = 0;
        for(int data = 0; data < TrainFeature.size(); ++data)
        {
            avg_sum += TrainFeature[data][fea];
        }
        avg_sum /= TrainFeature.size();
        Weight.push_back(avg_sum + rand() * 10.0 / RAND_MAX);
    }
}

double fun(double x)
{
    return x * x * x;
}
void PLA::generateFeature(FEATURE &feature)
{
    int baiscFeature = feature.size();
    for(int fea = 1; fea < baiscFeature; ++fea)
    {
        for(int bea = fea + 1; bea < baiscFeature; ++bea)
            feature.push_back(feature[fea] * feature[bea]);
    }
}
/*

READERs

*/
void  PLA::ReadTrainData()
{
    FEATURE Fe;
    LABEL   La;
    ifstream traini(train_data.c_str(), std::ifstream::in);
    TrainFeature.clear();
    TrainLabel.clear();
    while(ReadDataWithLabel(Fe, La, traini))
    {
        //generateFeature(Fe);
        TrainFeature.push_back(Fe);
        TrainLabel.push_back(La);
    }
    cout << "Reading(train):" << TrainFeature.size() << "*" << TrainFeature[0].size() << endl;
}



void PLA::ReadTestData()
{
    FEATURE Fe;
    ifstream testi(test_data.c_str(), std::ifstream::in);
   
    TestFeature.clear();
    while(ReadDataWithQuery(Fe, testi))
    {
        TestFeature.push_back(Fe);
    }
    cout << "Reading(Test):" << TestFeature.size() << "*" << TestFeature[0].size() << endl;

}



bool PLA::ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin)
{
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;

        feature.clear();
        feature.push_back(1);
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

bool PLA::ReadDataWithQuery(FEATURE &feature, ifstream & fin)
{
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;

        feature.clear();
        feature.push_back(1);
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
        return true;
    }
    else
        return false;// read the end of document
}

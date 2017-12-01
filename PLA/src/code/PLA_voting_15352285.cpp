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
    void                train();
    void                valid();
    void                test();
private:
    vector<FEATURE>     TrainFeature;
    vector<FEATURE>     ValidFeature;
    vector<FEATURE>     TestFeature;
    
    vector<double>      Weight, MaxWeight;

    vector<LABEL>       TrainLabel;
    vector<LABEL>       ValidLabel;

    int                 trainStep(double alpha = 1);
    
    // calcu function only for validation data.
    void                Calcu(double& Accuracy,double& Precision,double& Recall,double& F1,const vector<FEATURE>& Fes,const vector<LABEL>& Lbs);
    double              Accuracy_C(const vector<FEATURE>& Fes,const vector<LABEL>& Lbs);

    void                initZeroWeight();
    void                initAvgWeight();
    void                generateFeature(FEATURE &feature);
    bool                ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin);
    bool                ReadDataWithQuery(FEATURE &feature, ifstream & fin);

    void                ReadTrainData();
    void                ReadValidData();
    void                ReadTestData();

    void                normalize();
    string              TestLine;
}MyPLA;

int main(int argc, char** argv)
{
    MyPLA.init();
    MyPLA.voting();
    MyPLA.valid();
    MyPLA.test();
    return 0;
}

void PLA::init()
{
    TrainFeature.clear();ValidFeature.clear();TestFeature.clear();
    TrainLabel.clear();ValidLabel.clear();
}

void PLA::voting()
{
    //626 Positive Sample
    ReadTrainData();
    double Accuracy, Precision, Recall, F1;
    //normalize();
    //160 Positive Label
    ReadValidData();
    ofstream v("valid_better_drop.csv",std::ofstream::out);
    vector<double> voting_pool;
    for (int i = 0; i < ValidFeature.size(); ++i)
    {
        voting_pool.push_back(0);
    }
    double maxAccu = 0, maxP = 0;
    
    v << "cell,nowAccu,Accuracy,Precision,Recall,F1," << endl;

    for(int cell = 1; cell <= 200; ++cell)
    {
        initAvgWeight();
        int iter = 0, cnt_now = 0;
        double Accuracy = 0, Precision = 0,
                Recall = 0, F1 = 0;
        for(iter = 0; iter < 100; ++iter)
            trainStep(1.0);
        for(int val = 0;val < ValidFeature.size(); ++val)
        {
            double sum = 0;
            for(int k = 0; k < ValidFeature[val].size(); ++k)
            {
                sum += ValidFeature[val][k] * Weight[k];
            }
            voting_pool[val] += (sum>0)?2:-1;
            cnt_now += sum * ValidLabel[val] > 0;
        }

        double cnt = 0;
        int TP = 0, TN = 0, FP = 0, FN = 0;
        for(int val = 0;val < ValidLabel.size(); ++val)
        {
            if(voting_pool[val] > 0 && ValidLabel[val] > 0) TP++;
            else if(voting_pool[val] > 0 && ValidLabel[val] < 0) FP++;
            else if(voting_pool[val] < 0 && ValidLabel[val] < 0) TN++;
            else FN++;// care for zero situation.
        }
        Accuracy  = (TP + TN) * 1.0 / (TP + FP + TN + FN);
        Precision = (TP + FP) ? TP * 1.0 / (TP + FP):0;
        Recall    = (TP + FN) ? TP * 1.0 / (TP + FN):0;
        F1        = (Precision + Recall) ? 2 * Precision * Recall / (Precision + Recall):0;

        v << cell << "," 
            << cnt_now * 1.0 / ValidLabel.size() << ","
            << Accuracy << ","
            << Precision << ","
            << Recall << ","
            << F1 << ","
            << endl;
        if(Accuracy > maxAccu)
        {
            maxAccu = Accuracy;
            maxP = cell;
        }
    }
    cerr << maxAccu << "/" << maxP << endl;
}
/*

public methods

*/
void PLA::train()
{
    //626 Positive Sample
    ReadTrainData();
    double Accuracy, Precision, Recall, F1;
    //normalize();
    //160 Positive Label
    ReadValidData();
    ofstream v("valid_pocket1.csv",std::ofstream::out);

    initZeroWeight();
    double maxAccu = 0, maxP = 0;
    double alpha = 1;
    
    v << "packnum,Accuracy,Precision,Recall,F1,maxAccu" << endl;
    for(int bag = 1; bag < 100; ++bag)
    {
        initAvgWeight();
        int iter = 0;
        for(iter = 0; iter < 10; ++iter)
            trainStep(alpha);
        if(Accuracy_C(ValidFeature, ValidLabel) > maxAccu)
        {
            maxAccu = Accuracy_C(ValidFeature, ValidLabel);
            maxP = bag;
            MaxWeight = Weight;
        }
        Calcu( Accuracy, Precision, Recall, F1, ValidFeature, ValidLabel);
        if(Accuracy > 0.84 )
        {
            cerr << "#Test bag = " << bag << " : " << Accuracy << endl;
            v << bag << "," << Accuracy << "," << Precision << "," << Recall << "," << F1 << "," << maxAccu << endl;
        }
        if(bag % 10 == 0)
            cerr << "#Test bag = " << bag << " : "<< maxAccu << "/" << maxP << endl;
    }
    cerr << maxAccu << "/" << maxP << endl;
    Weight = MaxWeight;
    // for(int i = 0; i < Weight.size();++i)
    //     cerr << Weight[i] << ",";
    // cerr << endl;
}

void  PLA::valid()
{
    //ReadValidData();
    cerr << "VALID:" << Accuracy_C(ValidFeature, ValidLabel) << endl;
}
void  PLA::test()
{
    ReadTestData();
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

double PLA::Accuracy_C(const vector<FEATURE>& Fes,const vector<LABEL>& Lbs)
{
    int totAccurate = 0;
    for(int i = 0; i < Fes.size(); ++i)
    {
        double sum = 0;
        for(int k = 0; k < Fes[i].size(); ++k)
        {
            sum += Fes[i][k] * Weight[k];
        }
        if(sum * Lbs[i] > 0) ++totAccurate;
    }
    return totAccurate * 1.0 / Fes.size();
}
void PLA::Calcu(double& Accuracy,double& Precision,double& Recall,double& F1, const vector<FEATURE>& Fes,const vector<LABEL>& Lbs)
{
    int TP = 0, FN = 0, TN = 0, FP = 0;
    for(int i = 0; i < Fes.size(); ++i)
    {
        double sum = 0;
        for(int k = 0; k < Fes[i].size(); ++k)
        {
            sum += Fes[i][k] * Weight[k];
        }
        if(sum > 0 && Lbs[i] > 0) TP++;
        else if(sum > 0 && Lbs[i] < 0) FP++;
        else if(sum < 0 && Lbs[i] < 0) TN++;
        else FN++;// care for zero situation.
    }
    Accuracy  = (TP + TN) * 1.0 / (TP + FP + TN + FN);
    Precision = (TP + FP) ? TP * 1.0 / (TP + FP):0;
    Recall    = (TP + FN) ? TP * 1.0 / (TP + FN):0;
    F1        = (Precision + Recall) ? 2 * Precision * Recall / (Precision + Recall):0;
}



void PLA::initZeroWeight()
{
    Weight.clear();
    for(int i = 0; i < TrainFeature[0].size(); ++i)
        Weight.push_back(0);
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
void PLA::normalize()
{
    for(int fea = 1; fea < TrainFeature[0].size(); ++fea)
    {
        double avg_sum = 0, delta_sum = 0;
        for(int data = 0; data < TrainFeature.size(); ++data)
        {
            avg_sum += TrainFeature[data][fea];
        }
        avg_sum /= TrainFeature.size();
        for(int data = 0; data < TrainFeature.size(); ++data)
        {
            delta_sum += (TrainFeature[data][fea] - avg_sum)*(TrainFeature[data][fea] - avg_sum);
        }
        delta_sum = sqrt(delta_sum);
        if(delta_sum != 0)
        {
            for(int data = 0; data < TrainFeature.size(); ++data)
            {
                TrainFeature[data][fea] = (TrainFeature[data][fea]-avg_sum)/delta_sum;
            }
        }
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

void PLA::ReadValidData()
{
    FEATURE Fe;
    LABEL   La;
    ifstream valii(vali_data.c_str(), std::ifstream::in);
    ValidFeature.clear();
    ValidLabel.clear();
    while(ReadDataWithLabel(Fe, La, valii))
    {
        //generateFeature(Fe);
        ValidFeature.push_back(Fe);
        ValidLabel.push_back(La);
    }
    cout << "Reading(validation):" << ValidFeature.size() << "*" << ValidFeature[0].size() << endl;
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

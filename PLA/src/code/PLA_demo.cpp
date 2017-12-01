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
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;

// input
const string train_data = "train.csv";
const string test_data = "test.csv";
// output

typedef vector<double>  FEATURE;
typedef int             LABEL; 


class PLA{
public:
    void                init();
    void                train();
    void                valid();
    void                test();
private:
    vector<FEATURE>     TrainFeature;
    vector<FEATURE>     TestFeature;
    
    vector<double>      Weight;

    vector<LABEL>       TrainLabel;

    int                 trainStep(double alpha = 1);
    
    void                initZeroWeight();
    
    bool                ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin);
    bool                ReadDataWithQuery(FEATURE &feature, ifstream & fin);

    void                ReadTrainData();
    void                ReadTestData();

    string              TestLine;
}MyPLA;

int main(int argc, char** argv)
{
    MyPLA.train();
    MyPLA.test();
    return 0;
}

/*

public methods

*/
void PLA::train()
{
    //626 Positive Sample
    ReadTrainData();
    double Accuracy, Precision, Recall, F1;
    
    initZeroWeight();
    
    double alpha = 1;
    cerr << "initial Weight : ";
    for (int i = 0; i < Weight.size(); ++i)
    {
        cerr << Weight[i] << " ";
    }
    cerr << endl;
    trainStep(alpha);
    cerr << "after 1 train : ";
    
    for (int i = 0; i < Weight.size(); ++i)
    {
        cerr << Weight[i] << " ";
    }
    cerr << endl;
}

void  PLA::test()
{
    ReadTestData();
    double sum = 0;
    for (int i = 0; i < TestFeature[0].size(); ++i)
    {
        sum += TestFeature[0][i] * Weight[i];
    }
    cerr << "sum = " << sum << endl;
    cerr << "label(predict) = " << ((sum > 0) ? 1 : -1) << endl;

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
                Weight[k] += TrainLabel[i] * TrainFeature[i][k];
            }
            ++ErrNum;
        }// incorrect sample, update.
    }
    return ErrNum;
}




void PLA::initZeroWeight()
{
    Weight.clear();
    for(int i = 0; i < TrainFeature[0].size(); ++i)
        Weight.push_back(0);
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
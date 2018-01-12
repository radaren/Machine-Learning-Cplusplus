//  MultiLayer Perception
//
//  Created by 任磊达 on 2017/12/4.
//  Copyright © 2017年 任磊达. All rights reserved.
//
#include <map>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
// timer
clock_t beginTime, lastTime, nowTime;
inline void startProgramTimer()
{
    beginTime = clock();
    lastTime = beginTime;
}
inline void timeCounter(const char* pid)
{
    nowTime = clock();
    cerr << "Period [" << pid << "],\ttime cost "
    << double(nowTime - lastTime)/ CLOCKS_PER_SEC << "s,\ttot time cost "
    << double(nowTime - beginTime)/ CLOCKS_PER_SEC << "s\n" << endl;
    lastTime = nowTime;
}

// 文件地址
const string train_data     = "/Users/radar/Desktop/BPNN_Dataset/train_.csv";
const string test_data      = "/Users/radar/Desktop/BPNN_Dataset/test_.csv";
const string result         = "/Users/radar/desktop/15352285_renleida_BPNN.txt";

// use for handle data for mlp dataset
class Data{
private:
    void readHeader(ifstream& fin, bool isTrain = false)
    {
        string Header;
        getline(fin, Header, '\n');
        string name = "";
        for(int idx = 0; idx < Header.size(); ++idx)
        {
            if(Header[idx] == ',')
            {
                if(isTrain)
                    colName[name] = colID++;
                name = "";
            }
            else name = name + Header[idx];
        }
        if(name.size() && isTrain)
            colName[name] = colID++;
    }
    bool readData(ifstream &fin, vector<double>& vd){
        vd.clear();
        string Line;
        if(getline(fin, Line, '\n'))
        {
            string name = "";
            for(int idx = 0; idx < Line.size(); ++idx)
            {
                if(Line[idx] == ',')
                {
                    vd.push_back(atof(name.c_str()));
                    name = "";
                }
                else name = name + Line[idx];
            }
            if(name.size())
                vd.push_back(atof(name.c_str()));
            return true;
        }
        return false;
    }
    void readtrain()
    {
        ifstream train_i(train_data.c_str());
        readHeader(train_i,true);
        vector<double> vec;
        while(readData(train_i, vec))
        {
            oridata.push_back(vec);
        }
    }
    void readtest()
    {
        ifstream test_i(test_data.c_str());
        readHeader(test_i);
        vector<double> vec;
        while(readData(test_i, vec))
        {
            testdata.push_back(vec);
        }
    }
    vector<vector<double> > oridata;
    vector<vector<double> > testdata;
    void genTrainValidTest(){
        trainx.clear(); trainy.clear();
        validx.clear(); validy.clear();
        testx.clear(); testy.clear();
        // gen train&valid
        for(int d = 0; d < oridata.size(); ++d)
        {
            vector<double> vd;
            for (int i = 0; i < oridata[0].size(); ++i) {
                if(i == ID) continue;
                else if(i == Y || i == MNTH) continue;
                else{
                    vd.push_back(oridata[d][i]);
                }
            }
            if(oridata[d][MNTH] > 11.5) //december
            {
                validx.push_back(vd);
                validy.push_back(oridata[d][Y]);
            }
            else{
                trainx.push_back(vd);
                trainy.push_back(oridata[d][Y]);
            }
        }
        
        // gen testdata
        for(int d = 0; d < testdata.size(); ++d)
        {
            vector<double> vd;
            for (int i = 0; i < testdata[0].size(); ++i) {
                if(i == ID) continue;
                else if(i == Y || i == MNTH) continue;
                else{
                    vd.push_back(testdata[d][i]);
                }
            }
            testx.push_back(vd);
        }
    }
    map<string, int> colName;
    int colID;
    int ID, Y, MNTH;
public:
    vector<vector<double> > trainx, validx, testx;
    vector<double> trainy, validy, testy;
    unsigned long xLen;
    Data(){
        colID = 0;
        readtrain();
        readtest();
        ID = colName["instant"];
        Y = colName["cnt"];
        MNTH = colName["mnth"];
        // here use December as valid.
        genTrainValidTest();
        xLen = trainx[0].size();
    }
};

enum PARTIAL_TYPE{
    SIGMOID = '1',
    TANH = '2',
    RELU = '3',
    NONE = '4'
};

// use for input layer, output layer
class placeHolder{
public:
    vector<double> value; // output value(when fw)
    placeHolder* nextLayer, *lastLayer;
    
    placeHolder(unsigned long Len)
    {
        value.resize(Len,0);
        lastLayer = nextLayer = NULL;
    }
};

// inherited from placeholder
class MLPLayer: public placeHolder{
public:
    vector<vector<double> > weight, updw;
    vector<double> bias, err, updb; // err(when bp)
    MLPLayer(unsigned long Len, unsigned long lastLen):placeHolder(Len)
    {
        weight.clear(); bias.clear();
        for(int item = 0; item < Len; ++item)
        {
            bias.push_back((2.0*rand())/RAND_MAX - 1);
        }
        updb.resize(Len, 0);
        for(int row = 0; row < Len; ++row)
        {
            vector<double> rdnow;
            for(int item = 0; item < lastLen; ++item)
            {
                rdnow.push_back((2.0*rand())/RAND_MAX - 1);
            }
            weight.push_back(rdnow);
            updw.push_back(vector<double>(Len,0));
        }
        err.resize(Len, 0);
    }
};

class MultilayerPerception{
    
public:
    void IHOmodel(int hidden_len = 3, int max_iter = 3, double alpha = 1e-5, PARTIAL_TYPE pt = SIGMOID, bool NORMALGD=false);
    
    void MLPmodel(int hdLNum = 3, int hidden_len = 128, int max_iter = 300, double alpha = 1e-7, PARTIAL_TYPE pt = SIGMOID);
    double forward(PARTIAL_TYPE pt, PARTIAL_TYPE opt = NONE);
    void backward(double label, double alpha, double lambda, PARTIAL_TYPE pt, PARTIAL_TYPE bkpt = NONE);
    
    /*
     normal gd(3 steps):
        1. initialization all updw and updb as 0
        2. add err in updw and updb
        3. update w matrix
     */
    void backward0(){
        MLPLayer *initl = (MLPLayer*) InputLayer->nextLayer;
        while(initl != NULL)
        {
            initl->updb.resize(initl->updb.size(), 0);
            for(int i = 0; i < initl->updw.size(); ++i)
            {
                initl->updw[i].resize(initl->updw[i].size(), 0);
            }
            initl = (MLPLayer*)initl->nextLayer;
        }
    }
    void backward1(double label, PARTIAL_TYPE pt, PARTIAL_TYPE bkpt){
        // calcu error
        // output layer(with active)
        MLPLayer *nextL = OutputLayer;
        nextL->err[0] = (label-nextL->value[0]) * partial(nextL->value[0], bkpt)/data.testx.size();
        // hidden layer
        placeHolder *nowL = nextL->lastLayer;
        // calcu -err here, so the next is add err('-err')
        while(nowL->lastLayer != NULL) //input layer
        {
            MLPLayer *NL = (MLPLayer*) nowL;
            for(int w = 0; w < NL->value.size(); ++w)
            {
                NL->err[w] = 0;
                for(int i = 0; i < NL->nextLayer->value.size(); ++i)
                {
                    NL->err[w] += nextL->err[i] * nextL->weight[i][w];
                }
                NL->err[w] *= partial(NL->value[w], pt);
            }
            nowL = nowL->lastLayer;
            nextL = (MLPLayer*) nextL->lastLayer;
        }
        // update updw and updb
        nowL = OutputLayer;
        while(nowL->lastLayer != NULL)//input layer
        {
            MLPLayer *NL = (MLPLayer*) nowL;
            for(int w = 0; w < NL->err.size(); ++w)
            {
                NL->updb[w] += NL->err[w];
                for(int nw = 0; nw < NL->lastLayer->value.size(); ++nw)
                {
                    NL->updw[w][nw] += NL->err[w] * NL->lastLayer->value[nw];
                    NL->updw[w][nw] = 0;
                }
            }
            nowL = nowL->lastLayer;
        }
    }
    void backward2(double alpha,double lambda)
    {
        // update w and b
        placeHolder* nowL = OutputLayer;
        while(nowL->lastLayer != NULL)//not input layer
        {
            MLPLayer *NL = (MLPLayer*) nowL;
            for(int w = 0; w < NL->err.size(); ++w)
            {
                if(!isnan(NL->updb[w]))
                    NL->bias[w] += alpha * NL->updb[w];
                for(int nw = 0; nw < NL->lastLayer->value.size(); ++nw)
                {
                    if(!isnan(NL->updw[w][nw]))
                        NL->weight[w][nw] += alpha * NL->updw[w][nw];
                }
            }
            nowL = nowL->lastLayer;
        }
    }

    void December()
    {
        InputLayer = new placeHolder(data.xLen);
        MLPLayer *Hidden;
        
        Hidden = new MLPLayer(5, InputLayer->value.size());
        InputLayer->nextLayer = Hidden;
        Hidden->lastLayer = InputLayer;
        
        OutputLayer = new MLPLayer(1, 5);
        Hidden->nextLayer = OutputLayer;
        OutputLayer->lastLayer = Hidden;
        double lasterr = 1e7, nowerr = 0;
        double alpha = 1e-7;
        for(int iter = 1; iter <= 3000; ++iter)
        {
            for(int dt = 0; dt < data.trainx.size(); ++dt)
            {
                InputLayer->value = data.trainx[dt];
                forward(TANH);
                backward(data.trainy[dt], alpha, 1e-7, TANH);
            }
            nowerr = validate(iter, TANH);
            if(nowerr < lasterr)
            {
                if(nowerr < 6276)
                {
                    ofstream csv("/Users/radar/Desktop/BPNN_Dataset/December1.csv");
                    csv << "dt,real,predict" << endl;
                    for(int dt = 0; dt < data.validx.size(); ++dt)
                    {
                        InputLayer->value = data.validx[dt];
                        csv << dt << "," << data.validy[dt] << "," << forward(TANH) << endl;
                    }
                    csv.close();
                }
                lasterr = nowerr;
            }
        }
        // delete for potential memory leak
        MLPLayer *w = (MLPLayer*)InputLayer->nextLayer, *h;
        delete InputLayer;
        while(w)
        {
            h =  (MLPLayer*)w->nextLayer;
            delete w;
            w = h;
        }
    }
private:
    placeHolder* InputLayer;
    MLPLayer* OutputLayer;
    Data data;
    double fx(double x, PARTIAL_TYPE type)
    {
        if(type == SIGMOID) return 1.0 / (1 + exp(-x));
        if(type == TANH) return (exp(2 * x) - 1.0) / (exp(2 * x) + 1.0);
        if(type == RELU) return x > 0 ? x : 0;
        return x; // type == NONE
    }
    double partial(double fx, PARTIAL_TYPE type){
        if(type == SIGMOID) return fx * (1 - fx);
        if(type == TANH) return 1 - fx * fx;
        if(type == RELU) return fx > 0 ? 1 : 0;
        return 1; // type == NONE
    }
    double validate(int id, PARTIAL_TYPE pt){
        char num[30];
        double mse = 0;
        for(int dt = 0; dt < data.validx.size(); ++dt)
        {
            InputLayer->value = data.validx[dt];
            double predict = forward(pt);
            //if(!isnan(predict))
            mse += (predict - data.validy[dt]) * (predict - data.validy[dt]);
        }
        if(id % 10 == 0)
        {
            sprintf(num, "valid iteration[%d],MSE:%lf",id, mse/data.validy.size());
            timeCounter(num);
        }
        return mse / data.validx.size();
    }
    double trainerr(int id, PARTIAL_TYPE pt){
        char num[30];
        double mse = 0;
        for(int dt = 0; dt < data.trainx.size(); ++dt)
        {
            InputLayer->value = data.trainx[dt];
            double predict = forward(pt);
            //if(!isnan(predict))
            mse += (predict - data.trainy[dt]) * (predict - data.trainy[dt]);
        }
        if(id % 10 == 0)
        {
            sprintf(num, "train iteration[%d],MSE:%lf",id, mse/data.trainy.size());
            timeCounter(num);
        }
        return mse / data.trainy.size();
    }
};

int main(int argc, const char * argv[]) {
    startProgramTimer();
    MultilayerPerception MLP;
    timeCounter("Read and Handle Data");
    
    MLP.December();
    //IHOmodel(int hidden_len, int max_iter, double alpha, PARTIAL_TYPE pt,bool NORMALGD=false){
    //MLP.IHOmodel(4, 3000, 1e-4, SIGMOID, true);
    //MLP.IHOmodel(5, 3000, 1e-4, TANH,true);
    //MLP.IHOmodel(10, 3000, 1e-7, RELU,true);
    // sgd(when normalgd==flase)
    //MLP.IHOmodel(4, 3000, 1e-7, SIGMOID);
    //MLP.IHOmodel(5, 3000, 1e-7, TANH);
    //MLP.IHOmodel(10, 3000, 1e-7, RELU);
    
    // MLPmodel(int hdLNum = 3, int hidden_len = 128, int max_iter = 300, double alpha = 1e-7, PARTIAL_TYPE pt = SIGMOID){
    // MLP.MLPmodel(10,3,10000,1e-7,SIGMOID);
    return 0;
}





// implementation of mlp class
double MultilayerPerception::forward(PARTIAL_TYPE pt, PARTIAL_TYPE opt)
{
    placeHolder *iteration = InputLayer;
    MLPLayer *next;
    
    while (iteration->nextLayer != NULL) {
        next = (MLPLayer*)iteration->nextLayer;
        for(int v = 0; v < next->value.size(); ++v)
        {
            if(isnan(next->bias[v])) next->bias[v] = 0;
            next->value[v] = next->bias[v];
            for(int r = 0; r < iteration->value.size(); ++r)
            {
                if(isnan(next->weight[v][r]))next->weight[v][r]=0;
                next->value[v] += next->weight[v][r] * iteration->value[r];
            }
            // last layer do not need active function
            if(next->nextLayer != NULL)
                next->value[v] = fx(next->value[v], pt);
            else // output layer
                next->value[v] = fx(next->value[v], opt);
        }
        iteration = next;
    }
    return iteration->value[0];
}
void MultilayerPerception::backward(double label, double alpha,double lambda, PARTIAL_TYPE pt, PARTIAL_TYPE bkpt)
{
    // calcu error
    // output layer(with active)
    MLPLayer *nextL = OutputLayer;
    nextL->err[0] = (label-nextL->value[0]) * partial(nextL->value[0], bkpt);
    // hidden layer
    placeHolder *nowL = nextL->lastLayer;
    // calcu -err here, so the next is add err('-err')
    while(nowL->lastLayer != NULL) //input layer
    {
        MLPLayer *NL = (MLPLayer*) nowL;
        for(int w = 0; w < NL->value.size(); ++w)
        {
            NL->err[w] = 0;
            for(int i = 0; i < NL->nextLayer->value.size(); ++i)
            {
                NL->err[w] += nextL->err[i] * nextL->weight[i][w];
            }
            NL->err[w] *= partial(NL->value[w], pt);
        }
        nowL = nowL->lastLayer;
        nextL = (MLPLayer*) nextL->lastLayer;
    }
    // update w and b
    nowL = OutputLayer;
    while(nowL->lastLayer != NULL)//input layer
    {
        MLPLayer *NL = (MLPLayer*) nowL;
        for(int w = 0; w < NL->err.size(); ++w)
        {
            NL->bias[w] += alpha * NL->err[w];
            for(int nw = 0; nw < NL->lastLayer->value.size(); ++nw)
            {
                NL->weight[w][nw] *= (1 - lambda);// regulazation element
                NL->weight[w][nw] += alpha * NL->err[w] * NL->lastLayer->value[nw];
            }
        }
        nowL = nowL->lastLayer;
    }
}

void MultilayerPerception::IHOmodel(int hidden_len, int max_iter, double alpha, PARTIAL_TYPE pt,bool NORMALGD){
    InputLayer = new placeHolder(data.xLen);
    MLPLayer *Hidden;
    
    Hidden = new MLPLayer(hidden_len, InputLayer->value.size());
    InputLayer->nextLayer = Hidden;
    Hidden->lastLayer = InputLayer;
    
    OutputLayer = new MLPLayer(1, hidden_len);
    Hidden->nextLayer = OutputLayer;
    OutputLayer->lastLayer = Hidden;
    ofstream csv("/Users/radar/Desktop/BPNN_Dataset/gd_tanh.csv");
    csv << "iter,train,test" << endl;
    double lasterr = 1e-7, nowerr = 0;
    for(int iter = 1; iter <= max_iter; ++iter)
    {
        if(NORMALGD)
            backward0();
        for(int dt = 0; dt < data.trainx.size(); ++dt)
        {
            InputLayer->value = data.trainx[dt];
            forward(pt);
            if (NORMALGD) {
                backward1(data.trainy[dt], pt,NONE);
            }else
            backward(data.trainy[dt], alpha, 1e-6, pt);
        }
        if(NORMALGD)
            backward2(alpha, 0);
        nowerr = validate(iter, pt);
        //if(nowerr > lasterr + 5)
        //    alpha *= 0.9;
        lasterr = nowerr;
        csv << iter << ",";
        csv << trainerr(iter, pt) << ",";
        csv << nowerr << endl;
    }
    // delete for potential memory leak
    MLPLayer *w = (MLPLayer*)InputLayer->nextLayer, *h;
    delete InputLayer;
    while(w)
    {
        h =  (MLPLayer*)w->nextLayer;
        delete w;
        w = h;
    }
}

// decrease learning rate depend on the performance on validation
void MultilayerPerception::MLPmodel(int hdLNum , int hidden_len , int max_iter , double alpha , PARTIAL_TYPE pt
                                    ){
    // construct
    InputLayer = new placeHolder(data.xLen);
    placeHolder * lastLayer = InputLayer;
    for(int l = 0; l < hdLNum; ++l)
    {
        MLPLayer *Hidden = new MLPLayer(hidden_len, lastLayer->value.size());
        lastLayer->nextLayer = Hidden;
        Hidden->lastLayer = lastLayer;
        lastLayer = Hidden;
    }
    double lastrev = 1e7, rev;
    OutputLayer = new MLPLayer(1, hidden_len);
    lastLayer->nextLayer = OutputLayer;
    OutputLayer->lastLayer = lastLayer;
    
    for(int iter = 1; iter <= max_iter; ++iter)
    {
        // train
        for(int dt = 0; dt < data.trainx.size(); ++dt)
        {
            InputLayer->value = data.trainx[dt];
            forward(pt);
            
            backward(data.trainy[dt], alpha, 0, pt);
        }
        
        // validate
        if(iter % 10 == 0)
        {
            rev = validate(iter, pt);
            if(rev > lastrev)
                alpha *= 0.8;
            lastrev = rev;
        }
    }
    MLPLayer *w = (MLPLayer*)InputLayer->nextLayer, *h;
    delete InputLayer;
    while(w)
    {
        h =  (MLPLayer*)w->nextLayer;
        delete w;
        w = h;
    }
}

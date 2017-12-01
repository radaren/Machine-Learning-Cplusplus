//
//  main.cpp
//  决策树
//
//  Created by 任磊达 on 2017/11/1.
//  Copyright © 2017年 任磊达. All rights reserved.
//

/*
 1. 实现ID3决策树，验证集（随机）划分，暴力限制层数剪枝。
 2. 实验要求：第一层划分（基于划分验证集之后的训练集）：
     ·First Feature 111111111 div by Fea0(20.5)30/613·
 3. 实现基础ID3随机森林，F1指标较决策树优秀。
 */
#include <set>
#include <map>
#include <cmath>
#include <bitset>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

const int MAX_DATA = 1024;
const int MAX_FEATURE = 9;
const double inf = 1e10;
// 文件地址
const string train_data     = "/Users/radar/desktop/lab4_Decision_Tree/train.csv";
const string test_data      = "/Users/radar/desktop/lab4_Decision_Tree/test.csv";
const string result         = "/Users/radar/desktop/lab4_Decision_Tree/result.csv";
typedef vector<double>  FEATURE;
typedef int             LABEL;

vector<pair<FEATURE, LABEL> > oridata, trainData,validData, testdata;
// first : feature
// second: label

enum{
    ID3 = 'I',
    C4_5= 'C',
    GINI= 'G'
};
class DecisionNode{
public:
    DecisionNode();
    DecisionNode(const bitset<MAX_DATA>& GivenData, const bitset<MAX_FEATURE>& FilledFeature);
    void GenNode(int depth, int maxDepth, int NodeType=ID3);// 剪枝参数
    
    int Dpredict(const vector<double> & data)
    {
        if(Lcnt == 0) return 1;
        if(Rcnt == 0) return -1;
        if(DSubTree.count(data[DivideFeatureId]) == false) // leaf node
        {
            if(Lcnt < Rcnt) return 1; // 众数比较。
            else return -1;
        }
        else return DSubTree[data[DivideFeatureId]]->Dpredict(data);
    }
    int predict(const vector<double> & data)
    {
        if(Lcnt == 0) return 1;
        if(Rcnt == 0) return -1;
        if(LTree == NULL || RTree == NULL
           || DivideValue < 0) // leaf node
        {
            if(Lcnt < Rcnt) return 1; // 众数比较。
            else return -1;
        }
        if(data[DivideFeatureId] < DivideValue) return LTree->predict(data);
        else return RTree->predict(data);
    }
private:
    int NodeID;
    int DivideFeatureId, Lcnt, Rcnt;
    double DivideValue;
    map<int, int> VotingMap; // Feature, cnt
    bitset<MAX_FEATURE> FilledFeatureId;
    bitset<MAX_DATA> DataToDivide;
    DecisionNode *LTree, *RTree;
    
    map<int, DecisionNode*> DSubTree;
    double calcuEntropy(double p)
    {
        if(p <= 1e-7) return 0;
        return p * log2(1.0/p);
    }
    double cal_upd_Entropy(int fea, double & divideThread, double & info, double &ginisum);
};

map<int, DecisionNode*> leaves;
int totNode = 0;

bool oriReadFlag = false;
bool testReadFlag = false;
class DecisionTree{
public:
    void  ReadOriginalData();
    void  ReadTestData();
 
    void  GenTrain_Valid(double alpha = 0.2);
    
    void  Tree(int Type);
    void  Forest(int TreeNum, int maxDepth, double sampleData, double sampleFeature, int Type);
    
    void  Test();
private:
    bool ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin);
    bool ReadDataWithQuery(FEATURE &feature, ifstream & fin);
    bitset<MAX_DATA> _trainData, _selectData;
    bitset<MAX_FEATURE> _allFeature, _selectFeature;
    DecisionNode* _tree;
};
double calcuEntropy(double p)
{
    if(p <= 1e-7) return 0;
    return p * log2(1.0/p);
}
int main(int argc, const char * argv[]) {
    double a = calcuEntropy(1.0/4) + calcuEntropy(3.0/4);
    double b = calcuEntropy(1.0/2) ;
    cerr << a-b;
//    srand(0);
//    DecisionTree tree;
//    tree.Tree(ID3); // infinite
//    tree.Forest(15, 3, 0.1, 0.5, ID3);//TreeNum, maxDepth, sampleData, sampleFeature
//
//    tree.Tree(C4_5);
//    tree.Forest(15, 3, 0.1, 0.5, C4_5);//TreeNum, maxDepth, sampleData, sampleFeature
//
//    tree.Tree(GINI);
//    tree.Forest(15, 3, 0.1, 0.5, GINI);//TreeNum, maxDepth, sampleData, sampleFeature
//
//    tree.Test();
    return 0;
}

// use ID3 model to predict.
void DecisionTree::Test()
{
    if(oriReadFlag == false) ReadOriginalData();
    if(testReadFlag == false)ReadTestData();
    // without cut branch, use all data
    GenTrain_Valid(0);
    _tree = new DecisionNode(_trainData, _allFeature);
    _tree->GenNode(1, -1, ID3);
    ofstream re(result);
    for(int i = 0; i < testdata.size(); ++i)
    {
        re << _tree->predict(testdata[i].first) << endl;
    }
    
}

/* decision tree,
   this function use only once in speific model
 */
void DecisionTree::GenTrain_Valid(double alpha)
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
    _allFeature= 0;
    _trainData = 0;
    for(int i = 0; i < trainData.size(); ++i)
    {
        _trainData[i] = 1;
    }
    for(int i = 0; i < trainData[0].first.size(); ++i)
    {
        _allFeature[i] = 1;
    }
    cout << "Divide train(" << trainData.size() << "), valid(" << validData.size() << ")\n";
}
void  DecisionTree::Tree(int Type)
{
    if(oriReadFlag == false) ReadOriginalData();
    if(testReadFlag == false)ReadTestData();
    GenTrain_Valid(); // random divide the tot data
    // train
    _tree = new DecisionNode(_trainData, _allFeature);
    _tree->GenNode(1, -1, Type);
    // validate
    double Accuracy, Precision, Recall, F1;
    int TP = 0, FN = 0, TN = 0, FP = 0;
    for(int i = 0; i < validData.size(); ++i)
    {
        int pred = _tree->predict(validData[i].first);
        int label= validData[i].second;
        if(pred > 0 && label > 0) TP++;
        else if(pred > 0 && label < 0) FP++;
        else if(pred < 0 && label < 0) TN++;
        else FN++;// care for zero situation.
    }
    Accuracy  = (TP + TN) * 1.0 / (TP + FP + TN + FN);
    Precision = (TP + FP) ? TP * 1.0 / (TP + FP):0;
    Recall    = (TP + FN) ? TP * 1.0 / (TP + FN):0;
    F1        = (Precision + Recall) ? 2 * Precision * Recall / (Precision + Recall):0;
    printf("\n\nvalidate result(infinity depth) :\nTP(%d),TN(%d),FP(%d),FN(%d),\naccu(%f), prec(%f), recall(%f), f1(%f)\n\n\n",
           TP,TN,FP,FN,Accuracy, Precision, Recall, F1);
}

void DecisionTree::Forest(int TreeNum, int maxDepth, double sampleDataRate, double sampleFeatureRate, int Type)
{
    if(oriReadFlag == false) ReadOriginalData();
    if(testReadFlag == false)ReadTestData();
    GenTrain_Valid(); // random divide the tot data
    
    vector<int> voting_poll;
    for(int i = 0; i < validData.size(); ++i)
    {
        voting_poll.push_back(0);
    }
    double maxAccu = 0, pos = 0;
    double Accuracy_ = 0, Precision_ = 0, Recall_ = 0, F1_ = 0;
    int TP_ = 0, FN_ = 0, TN_ = 0, FP_ = 0, trees_ = 0;
    
    for(int trees = 0; trees < TreeNum; ++trees)
    {
        _selectData = 0;
        _selectFeature = 0;
        // sample data
        while (_selectData.count() == 0) {
            for(int i = 0; i < trainData.size(); ++i)
            {
                if(rand()*1.0/RAND_MAX<sampleDataRate)
                    _selectData[i] = 1;
            }
        }
        // sample feature
        while(_selectFeature.count() == 0){
            for(int i = 0; i < trainData[0].first.size(); ++i)
            {
                if(rand()*1.0/RAND_MAX<sampleFeatureRate)
                    _selectFeature[i] = 1;
            }
        }
        _tree = new DecisionNode(_selectData, _selectFeature);
        _tree->GenNode(1, maxDepth, Type);
        for(int i = 0; i < validData.size(); ++i)
        {
            if(_tree->predict(validData[i].first) > 0)
                voting_poll[i]++;
            else
                voting_poll[i]--;
        }
        // validate
        // TODO: 统计当前单个数据的性质
        double Accuracy, Precision, Recall, F1;
        int TP = 0, FN = 0, TN = 0, FP = 0;
        for(int i = 0; i < validData.size(); ++i)
        {
            int pred = voting_poll[i] > 0? 1 : -1;
            int label= validData[i].second;
            if(pred > 0 && label > 0) TP++;
            else if(pred > 0 && label < 0) FP++;
            else if(pred < 0 && label < 0) TN++;
            else FN++;// care for zero situation.
        }
        Accuracy  = (TP + TN) * 1.0 / (TP + FP + TN + FN);
        Precision = (TP + FP) ? TP * 1.0 / (TP + FP):0;
        Recall    = (TP + FN) ? TP * 1.0 / (TP + FN):0;
        F1        = (Precision + Recall) ? 2 * Precision * Recall / (Precision + Recall):0;
        //cerr << TP << "/" << FN << "/" << TN << "/" << FP << endl;
        if(Accuracy > maxAccu)
        {
            maxAccu = Accuracy;
            pos = trees;
            Accuracy_ = Accuracy;Precision_ = Precision;
            Recall_ = Recall;F1_ = F1;
            TP_ = TP; FN_ = FN; TN_ = TN; FP_ = FP; trees_ = trees;
        }
    }
    printf("validate result(random forest %d) :\nTP(%d),TN(%d),FP(%d),FN(%d),\naccu(%f), prec(%f), recall(%f), f1(%f)\n",
           trees_,TP_,TN_,FP_,FN_,Accuracy_, Precision_, Recall_, F1_);
}




/* Decision Node */
DecisionNode::DecisionNode()
{
    DivideValue = -1;
    DivideFeatureId = -1;
    FilledFeatureId = 0;
    Lcnt = Rcnt = 0;
    LTree = RTree = NULL;
    NodeID = ++totNode;
}
DecisionNode::DecisionNode(const bitset<MAX_DATA>& GivenData, const bitset<MAX_FEATURE>& FilledFeature)
{
    Lcnt = Rcnt = 0;
    LTree = RTree = NULL;
    DivideValue = -1;
    DivideFeatureId = -1;
    DataToDivide = GivenData; // inherited from father
    FilledFeatureId = FilledFeature;
    NodeID = ++totNode;
}
void DecisionNode::GenNode(int depth, int maxDepth, int Type)
{
    //初始化: 创建根结点，它拥有全部数据集和全部特征。
    LTree = RTree = NULL;
    Lcnt = Rcnt = 0;
    
    double tot = 0;
    for(int i = 0; i < trainData.size(); ++i)
    {
        if(DataToDivide[i] == 1)
        {
            Lcnt += trainData[i].second < 0;
            Rcnt += trainData[i].second > 0;
            ++tot;
        }
    }
    double EntropyOri = (calcuEntropy(Lcnt/tot) + calcuEntropy(Rcnt/tot));
    //选择特征: 遍历当前结点的数据集和特征，根据某种原则，选择一个特征。
    double Gain = 0, minGini = inf;
    for(int fea = 0; fea < trainData[0].first.size(); ++fea)
    {
        if(FilledFeatureId[fea] == 1) // 该特征还未被分类
        {
            double divideThread = 0, info = 0, ginisum = 0;
            // 计算当前划分中最优划分阈值， 存在divideThread，阈值对应的熵存在NowEntropy
            double NowEntropy = cal_upd_Entropy(fea, divideThread, info, ginisum);
            // 更新结构体
            if(Type == ID3)
            {
                if(EntropyOri - NowEntropy > Gain)
                {
                    Gain = EntropyOri-NowEntropy;
                    DivideFeatureId = fea;
                    DivideValue = divideThread;
                }
            }
            else if(Type == C4_5)
            {
                // 在连续值方面认作是只有两个分类
                if((EntropyOri - NowEntropy) / info > Gain)
                {
                    Gain = (EntropyOri - NowEntropy) / info;
                    DivideFeatureId = fea;
                    DivideValue = divideThread;
                }
            }
            else if(Type == GINI)
            {
                if(ginisum < minGini)
                {
                    minGini = ginisum;
                    DivideFeatureId = fea;
                    DivideValue = divideThread;
                }
            } // GINI
        }
    }
    //划分数据: 根据这个特征的取值，将当前数据集划分为若干个子数据集。
    bitset<MAX_FEATURE> NextFeature = FilledFeatureId;
    bitset<MAX_DATA> NextLData, NextRData;
    NextLData = 0; NextRData = 0;
    NextFeature[DivideFeatureId] = 0;
    for(int data = 0; data < trainData.size(); ++data)
    {
        if(DataToDivide[data] == 1)
        {
            if(trainData[data].first[DivideFeatureId] < DivideValue)
            {
                NextLData[data] = 1;
            }
            else
            {
                NextRData[data] = 1;
            }
        }
    }
//    if(maxDepth == -1 &&  FilledFeatureId.count() == MAX_FEATURE)
//    cerr<< "First Feature " << FilledFeatureId
//        << " div by Fea" << DivideFeatureId <<"(" << DivideValue << ")"
//        << NextLData.count() << "/" << NextRData.count() << endl;
    if(NextFeature.count() // 还有特征需要划分
       && NextLData.count() && NextRData.count() // 子树没有被完全划分
       && (maxDepth == -1 || depth < maxDepth)) // 层数剪枝
    {
        //创建结点: 为每个子数据集创建一个子结点，并删去刚刚选中的特征。
        //递归建树: 对每个子结点，回到第2步。直到达到边界条件，则回溯。
        if(NextLData.count())
        {
            LTree = new DecisionNode(NextLData, NextFeature);
            LTree->GenNode(depth+1, maxDepth, Type);
        }
        if(NextRData.count())
        {
            RTree = new DecisionNode(NextRData, NextFeature);
            RTree->GenNode(depth+1, maxDepth, Type);
        }
        //完成建树: 叶子结点采用多数投票的方式判定自身的类别
    }
    else
    {
        leaves[NodeID] = this;
        //printf("Leaf[%d](%d) : Lcnt(%d) Rcnt(%d)\n",NodeID, depth, this->Lcnt , this->Rcnt);
    }// leaf operation
}

double DecisionNode::cal_upd_Entropy(int fea, double & divideThread, double & info, double & ginisum)
{
    double NowEntropy = inf;
    int totData = (int) DataToDivide.count();// 二进制中为1的个数
    // 计算划分标准, 处理连续型，寻求最大熵增划分
    set<int> FeatureValue;
    for(int i = 0;i < trainData.size(); ++i)
    {
        if(DataToDivide[i] == 1)// 确定是当前划分数据集中的属性。
        {
            FeatureValue.insert(trainData[i].first[fea]);
        }
    }
    set<int>::iterator it = FeatureValue.begin();
    double oldValue = *it;
    for(++it; it != FeatureValue.end(); ++it) // (a[i] + a[i+1])/2
    {
        double nowDiv = (oldValue + (*it)) / 2;
        double nowEntropyDiv = 0;
        double pos_0 = 0, pos_1 = 0, neg_0 = 0, neg_1 = 0, tot_0 = 0, tot_1 = 0;
        for(int i = 0; i < trainData.size(); ++i)
        {
            if(DataToDivide[i] == 1)
            {
                if(trainData[i].first[fea] < nowDiv)
                {
                    pos_0 += trainData[i].second > 0;
                    neg_0 += trainData[i].second < 0;
                    ++tot_0;
                }// in the left branch
                else
                {
                    pos_1 += trainData[i].second > 0;
                    neg_1 += trainData[i].second < 0;
                    ++tot_1;
                }// in the right branch
            }
        }
        nowEntropyDiv = tot_0 / totData * (calcuEntropy(pos_0/tot_0) + calcuEntropy(neg_0/tot_0)) // calcu left entropy
        + tot_1 / totData * (calcuEntropy(pos_1/tot_1) + calcuEntropy(neg_1 / tot_1)); // calcu right entropy
        if(nowEntropyDiv < NowEntropy)
        {
            NowEntropy = nowEntropyDiv;
            divideThread = nowDiv;
            info = calcuEntropy(tot_0/totData) + calcuEntropy(tot_1/totData);
            ginisum = tot_0/totData * (1-pos_0*pos_0-neg_0*neg_0)
            + tot_1/totData * (1-pos_1*pos_1-neg_1*neg_1);
        }
        oldValue = *it;
    }
    return NowEntropy;
}





/* readers */
void  DecisionTree::ReadOriginalData()
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
void DecisionTree::ReadTestData()
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

bool DecisionTree::ReadDataWithLabel(FEATURE &feature, LABEL &label, ifstream & fin)
{
    string TestLine;
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;
        
        feature.clear();
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

bool DecisionTree::ReadDataWithQuery(FEATURE &feature, ifstream & fin)
{
    string TestLine;
    if(getline(fin, TestLine))
    {
        double Fea;
        size_t Ldir, Rdir;
        
        feature.clear();
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
        //Fea = atof(TestLine.substr(Ldir, Rdir - Ldir).c_str());
        // feature.push_back(Fea);
        return true;
    }
    else
        return false;// read the end of document
}

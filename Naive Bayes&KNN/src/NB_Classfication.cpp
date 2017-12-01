/* author : 任磊达 */
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <queue>
#include <map>
#include <set>
#include <cstring>
#include <fstream>
#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
using namespace std;
#define HINT_1

#define LINUX

typedef map<int,int>::iterator mit;
/*
basic functions
- timeCounter
- stdout recovery
*/
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
    printf("Period [%s],\ttime cost %.6fs,\ttot time cost %.4fs\n"
    , pid, double(nowTime - lastTime)/ CLOCKS_PER_SEC
    , double(nowTime - beginTime)/ CLOCKS_PER_SEC);
    lastTime = nowTime;
}

void recoverySTDOUT()
{
#ifdef WINDOWS
    freopen("CON","w",stdout);
#endif

#ifdef LINUX
    freopen("/dev/tty","w",stdout);
#endif
}
// dictionary 
// dict of word(string->id)
map<string, int> dict;
map<int, string> rdict;
const int MAX_EMOTION = 7; // 6
const int MAX_WORD = 1e7;
int wordCount[MAX_WORD];
int wordEmotionCount[MAX_WORD][MAX_EMOTION];
int totID = 0; // make id from 1

inline int wid(string word)
{
#ifdef HINT_1
    if(word.size() < 4 || word == "with") return 0;
#endif
    if (dict.count(word)) {
        return dict[word];
    }else{
        dict[word] = ++totID;
        rdict[totID] = word;
        wordCount[totID] = 0;
        return totID;
    }
}
// dict of emotion
map<string, int> emotionDict;
map<int, string> emotionrDict;
int emotionID = 0;
inline int eid(const string word)
{
    if (emotionDict.count(word)) {
        return emotionDict[word];
    }else{
        emotionDict[word] = ++emotionID;
        emotionrDict[emotionID] = word;
        return emotionID;
    }
}
// begin programming



struct rowData
{
    // ID, count
    map<int, int> IDs;
    int totWords, Emotion;
    //float anger,disgust,fear,joy,sad,surprise;
    
    inline void insert(const int strID)
    {
        // 统计词出现个数
        if(IDs.count(strID))
        {
            IDs[strID]++;
        }
        else
            IDs[strID] = 1;
      	// 统计总词数
        ++totWords;
    }
    inline void init()
    {
        IDs.clear();
        totWords = 0;
    }
}rD, rV;

bool readDataTorD(bool train = false)
{
    char emotion[10];
    rD.init();
    int wordID = 0;
    char reader = getchar();
    string stread = "";
    while(reader != EOF && reader == '\n') reader = getchar();
    if(reader == EOF) return false;
    while(1)
    {
        if(reader == ' ' || reader == ',')
        {
            wordID = wid(stread);
            if(train && rD.IDs.count(wordID) == 0)
                wordCount[wordID]++;
            if(wordID)
                rD.insert(wordID);
            stread = "";
        }
        else
            stread = stread + reader;
        if(reader == ',') break;
        
        reader = getchar();
    }
    scanf("%s", emotion);
    rD.Emotion = eid(emotion);
    return (reader = getchar()) != EOF;
}
vector<rowData> Library, vali;

double emotionP[MAX_EMOTION];
set<int> nck[MAX_EMOTION];
int TrainSize;

void train()
{
    cerr << "Enter train" << endl;
    freopen("./DATA/classification_dataset/train_set.csv", "r", stdin);
    char reader = '\0';
    while((reader = getchar()) != '\n');
    while(readDataTorD(true)) Library.push_back(rD);
    Library.push_back(rD);
    cerr << "Finish reading" << endl;
    for(int i = 0; i < Library.size(); ++i)
    {
        emotionP[Library[i].Emotion]++;
        for(mit w = Library[i].IDs.begin();
            w != Library[i].IDs.end(); ++w)
            {
                wordEmotionCount[w->first][Library[i].Emotion]
                    += w->second;
                nck[Library[i].Emotion].insert(w->first);
            }
    }
    TrainSize = totID; // 训练集维数
    cerr << "out of train" << endl;
    for(int i = 1; i < MAX_EMOTION; ++i)
    {
        emotionP[i] /= Library.size();
    }
}

// 使用了统计词频的矩阵，比OneHot多了一个出现次数

const double eps = 1e-10;
int NB(const double &alpha, rowData & words)
{
    double TestPossibility;

    int ans = 1;    // 标签
    double maxV = -1000000;  // 值
    for(int i = 1; i <= emotionID; ++i)
    {
        // calcu TestPossibility for emotion[i]
        TestPossibility = log(emotionP[i]) - words.totWords * log(nck[i].size() + alpha * TrainSize);
        for(mit iz = words.IDs.begin(); iz != words.IDs.end(); ++iz)
        {
            TestPossibility += iz->second 
                        * log(wordEmotionCount[iz->first][i] + alpha);
        }


        // TestPossibility = words.totWords * (log(emotionP[i]) - log(TrainSize) + log(emotionP[i] * Library.size()));
        // for(mit iz = words.IDs.begin(); iz != words.IDs.end(); ++iz)
        // {
        //     TestPossibility += iz->second 
        //                 * log((wordEmotionCount[iz->first][i] + alpha* TrainSize ) / (wordCount[i] + alpha ));
        // }

        //cout << TestPossibility << "\t";
        if(TestPossibility > maxV)
        {
            maxV = TestPossibility;
            ans = i;
        }
    }
//    cout << endl;
    return ans;
}
void validation()
{
    freopen("./DATA/classification_dataset/validation_set.csv", "r", stdin);   
    char reader = '\0';
    vali.clear();
    while((reader = getchar()) != '\n');
    while(readDataTorD()) vali.push_back(rD);
    vali.push_back(rD);
    timeCounter("read validation");
    freopen("./nb_cf.csv", "w", stdout);
    int idx = 0;
    double maxalpha, maxcorrect;
    maxalpha = 0, maxcorrect = 0;
    double step = 1e-100;
    
    for(double alpha = 0.0001; alpha < 10.00005; alpha += 0.0001)
    //double alpha = 1;
    {
        idx++;
        int correctSentence = 0;
        for(int j = 0; j < vali.size(); ++j)
        {
            correctSentence += (NB(alpha, vali[j]) == vali[j].Emotion);
        }
        double correct_rate = (1.0 * correctSentence) / vali.size();
        cout << alpha << "," << correct_rate << endl;
        if(correct_rate > maxcorrect){
            maxcorrect = correct_rate;
            maxalpha = alpha;
        }
        //if(idx % 1000 == 0)
        cerr << "at idx" << idx << " maxcor = " << maxcorrect << " at alpha = " << maxalpha << endl;    
    }
}
double alphaC(double alpha)
{
    int correctSentence = 0;
    for(int j = 0; j < vali.size(); ++j)
    {
        correctSentence += (NB(alpha, vali[j]) == vali[j].Emotion);
    }
    return (1.0 * correctSentence) / vali.size();
}
// 利用观察得到达凹凸性得到最优值点2.7013717421
void validation1()
{
    freopen("./DATA/classification_dataset/validation_set.csv", "r", stdin);   
    char reader = '\0';
    vali.clear();
    while((reader = getchar()) != '\n');
    while(readDataTorD()) vali.push_back(rD);
    vali.push_back(rD);
    recoverySTDOUT();
    int idx = 0;
    double maxalpha, maxcorrect;
    maxalpha = 0, maxcorrect = 0;
    
    double left = 0.1;
    double right = 10;
    double p1, p2, v1, v2;
    while(right-left>1e-16)
    {
        p1 = left + (right - left) / 3;
        p2 = left + 2 * (right - left) / 3;
        v1 = alphaC(p1);
        v2 = alphaC(p2);
        if(v1 > maxcorrect)
        {
            maxcorrect = v1;
            maxalpha = p1;
        }
        if(v2 > maxcorrect)
        {
            maxcorrect = v2;
            maxalpha = p2;
        }
        if(v1 <= v2)
            left = p1;
        else
            right = p2;
        idx++;
        printf("idx=%d,(%.10lf,%.10lf), maxv=%.10lf, maxp=%.10lf\n",idx,left,right,maxcorrect,maxalpha);
    }
}
bool readTestDataTorD()
{
    char emotion[10];
    rD.init();
    int wordID = 0;
    char reader = getchar();
    string stread = "";
    while(reader != EOF && reader != ',') reader = getchar();
    if(reader == EOF) return false;
    reader = getchar();
    while(1)
    {
        if(reader == ' ' || reader == ',')
        {
            wordID = wid(stread);
            if(wordID)
                rD.insert(wordID);
            stread = "";
        }
        else
            stread = stread + reader;
        if(reader == ',') break;
        
        reader = getchar();
    }
    scanf("%s", emotion);
    return (reader = getchar()) != EOF;
}
void test()
{
    freopen("./DATA/classification_dataset/test_set.csv", "r", stdin);   
    char reader = '\0';
    vali.clear();
    while((reader = getchar()) != '\n');
    while(readTestDataTorD()) vali.push_back(rD);
    vali.push_back(rD);

    freopen("./cv1.csv", "w", stdout);
    double alpha = 2.7013717421;
    for(int j = 0; j < vali.size(); ++j)
    {
        int ans = NB(alpha, vali[j]);
        cout << emotionrDict[ans] << endl;        
    }
}
/*
main function.
*/
int main(int argc, char** argv)
{
    srand((unsigned int)(time(NULL)));
    startProgramTimer();
    train();
    timeCounter("training");
    test();
    //validation1();
    recoverySTDOUT();
    timeCounter("predict");
    return 0;
}


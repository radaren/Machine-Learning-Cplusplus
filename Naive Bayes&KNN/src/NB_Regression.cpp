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
#include <assert.h>
using namespace std;

string test_addr = "./DATA/regression_dataset/test_set.csv";
string train_addr = "./DATA/regression_dataset/train_set.csv";
string validate_addr = "./DATA/regression_dataset/validation_set.csv";
const double eps = 1e-7;
const int MAX_WORD = 1e5;

const int MAX_EMOTION = 6;
int wordCount[MAX_WORD];
double wordEmotionCount[MAX_WORD][MAX_EMOTION];


double maxe[MAX_EMOTION];
double emotionP[MAX_EMOTION];
set<int> nck[MAX_EMOTION];
int TrainSize;


#define HINT

#define LINUX
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
    cerr << "Period [" << pid << "],\ttime cost "
     << double(nowTime - lastTime)/ CLOCKS_PER_SEC
     << "s,\ttot time cost " << double(nowTime - beginTime)/ CLOCKS_PER_SEC << "s\n";
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

inline double readFloatFromStdin()
{
    double a = 0, b = 0, power;
    int indot = false;
    char ch;
    while((ch = getchar()) != EOF) if(ch >= '0' && ch <= '9') break;
    if(ch != EOF) a = ch - '0';
    while((ch = getchar()) != EOF)
    {
        if(ch == '.')
        {
            indot = true;
            power = 0.1;
        }
        else if(ch >= '0' && ch <= '9')
        {
            if(indot == false)
                a = a * 10 + ch - '0';
            else
                {
                    b = b + power * (ch - '0');
                    power /= 10;
                }
        }
        else
            break;
    }
    return a + b;
}

typedef map<int, int>::iterator mit;
// dict of word(string->id)
map<string, int> dict;
map<int, string> rdict;
int totID = 0; // make id from 1

inline int wid(string word)
{
#ifdef HINT
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

struct rowData
{
    // ID, count
    map<int, int> IDs;
    int totWords;
    double Emotion[6];
    //float anger,disgust,fear,joy,sad,surprise;
    map<int, int>::iterator iter;
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
    inline void showEmotion()
    {
        for(int i = 0; i < 6; ++i) printf("%f\t", Emotion[i]);
        putchar('\n');
    }
}rD, rV;

vector<rowData> Library, standard, predict;

bool readDataTorD(bool train = false)
{
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
    scanf("%lf,%lf,%lf,%lf,%lf,%lf", &rD.Emotion[0],
    &rD.Emotion[1], &rD.Emotion[2], &rD.Emotion[3],
    &rD.Emotion[4], &rD.Emotion[5]);
    return (reader = getchar()) != EOF;
}
void NB(const double &alpha, rowData & words)
{
    double emotion[6];

    for(int emo = 0; emo < MAX_EMOTION; ++emo)
    {
        emotion[emo] = log(emotionP[emo])
        - words.totWords * log(alpha * nck[emo].size() + TrainSize);
        for(mit iz = words.IDs.begin(); iz != words.IDs.end(); ++iz)
        {
            emotion[emo] += iz->second 
                        * log(wordEmotionCount[iz->first][emo] + alpha);
        }
    }
    
    
    // standard score
    // 归一化， 这里最后一步没有减去均值u，防止出现负数
    double u, d, sum = 0;
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        sum += emotion[i];
    }
    u = sum / MAX_EMOTION;
    sum = 0;
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        sum += (emotion[i] - u) * (emotion[i] - u);
    }
    d = sqrt(sum / MAX_EMOTION);
    for(int i = 0; i < MAX_EMOTION; ++i)
        emotion[i] = (emotion[i] ) / d;

    // 尺度放缩
    double maxV = emotion[0], minV = emotion[0]; // delete negative
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        minV = min(minV, emotion[i]);
        maxV = max(maxV, emotion[i]);     
    }
    for(int i = 0; i < MAX_EMOTION; ++i)
    {    
        emotion[i] = (emotion[i] - minV) / (maxV - minV);
    }
    // change sum to 1:
    sum = 0;
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        sum += emotion[i];
    }
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        emotion[i] /= sum;
    }

    // 存储，用于计算相关系数
    for(int i = 0; i < MAX_EMOTION; i++)
    {
        if(i < MAX_EMOTION - 1)
            printf("%.6f,", emotion[i]);
        else
            printf("%.6f\n", emotion[i]);
        rV.Emotion[i] = emotion[i];
    }

}


void train()
{
    freopen(train_addr.c_str(), "r", stdin);
    char reader = '\0';
    while((reader = getchar()) != '\n');
    while(readDataTorD())
    {
        Library.push_back(rD);
    }
    Library.push_back(rD);
    for(int i = 0; i < Library.size(); ++i)
    {
        for(int emo = 0; emo < MAX_EMOTION; ++emo)
        {
            emotionP[emo] += Library[i].Emotion[emo];
            for(mit w = Library[i].IDs.begin();
                w != Library[i].IDs.end(); ++w)
                {
                    wordEmotionCount[w->first][emo]
                        += w->second * Library[i].Emotion[emo];
                    
                    if(Library[i].Emotion[emo] > 0.3)
                        nck[emo].insert(w->first);
                }
        }
    }
    #ifdef HINT
    double sum;
    // 归一化一下 wordEmotionCount
    for(int emo = 0; emo < MAX_EMOTION; ++emo)
    {
        sum = 0;
        for(int i = 0; i < MAX_WORD; ++i)
            sum += wordEmotionCount[i][emo]*wordEmotionCount[i][emo];
        sum = sqrt(sum);
        for(int i  =0; i < MAX_WORD; ++i)
            wordEmotionCount[i][emo] /= sum;
    }
    #endif
    TrainSize = totID; // 训练集维数
    // sum of all emotion of a sentence is 1
    for(int i = 0; i < MAX_EMOTION; ++i)
    {
        emotionP[i] /= Library.size();
    }

    freopen(validate_addr.c_str(), "r", stdin);
    while((reader = getchar()) != '\n');
    standard.clear();
    while(readDataTorD())
    {
        standard.push_back(rD);
    }
    standard.push_back(rD);
}
// calcu corr between rD.Emotion & rV.Emotion
double corr(const int &c)
{
    double sum1, sum2, sum3;
    double ass, asp;
    double x_x, y_y;
    
    sum1 = sum2 = sum3 = 0;
    ass = asp = 0;

    for(int r = 0; r < standard.size(); ++r)
    {
        ass += standard[r].Emotion[c];
        if(!isnan(predict[r].Emotion[c]))
            asp += predict[r].Emotion[c];
    }
    ass /= standard.size();
    asp /= standard.size(); // mean

    for(int r = 0; r < standard.size(); ++r)
    {
        x_x = standard[r].Emotion[c] - ass;
        y_y = predict[r].Emotion[c] - asp;
        if(isnan(y_y)) y_y = x_x;
        sum1 += x_x * y_y;
        sum2 += x_x * x_x;
        sum3 += y_y * y_y;
    }
    sum2 = max(eps, sum2);
    sum3 = max(eps, sum3);
    return sum1 / sqrt(sum2 * sum3);
}
void validation()
{
    double ans = 0;
    double anspos = 0;
    int k = 0;
    double step = 0.00002;
    //for(double alpha = step; alpha < 0.02; alpha += step)
    double alpha = 0.013889;
    {
        //++k;
        predict.clear();
        for(int i = 0; i < standard.size(); ++i)
        {
            NB(alpha, standard[i]);
            predict.push_back(rV);
        }   
        double cor[MAX_EMOTION];
        double corsum;
        corsum = 0;
        memset(cor, 0, sizeof(cor));
        for(int i = 0; i < MAX_EMOTION; ++i)
        {
            cor[i] = corr(i);
            corsum += cor[i];    
        }
        if(corsum > ans)
        {
            ans = corsum;
            anspos = alpha;
        }
        if(k % 1000 == 0)
        {
            cerr << "At iteratoration " << k << " maxAns = " << ans / MAX_EMOTION
            << " maxAnsPos = " << anspos << endl;
            if(k % 100000 == 0)
            {
                timeCounter("PAUSE");
            }
        }
        // printf("%f,%lf", alpha, corsum / MAX_EMOTION);
        // for(int i = 0; i < MAX_EMOTION ; ++i) printf(",%lf", cor[i]);
        // puts("");
    }
}
bool readTestDataTorD()
{
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
            if(rD.IDs.count(wordID) == 0)
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
    
    while(reader == ',' || reader == '?') reader = getchar();
    
    return (reader = getchar()) != EOF;
}
void test()
{
    
    char reader;
    freopen(test_addr.c_str(), "r", stdin);
    while((reader = getchar()) != '\n');
    standard.clear();
    while(readTestDataTorD())
    {
        standard.push_back(rD);
    }
    standard.push_back(rD);
    double alpha = 0.013889;
    for(int i = 0; i < standard.size(); ++i)
    {
        NB(alpha, standard[i]);
    }    
}
int main(int argc, char** argv)
{
    srand((unsigned int)(time(NULL)));
    startProgramTimer();
    train();
    timeCounter("training");
    freopen("./test1.csv", "w", stdout);
    test();
    //validation();
    

    recoverySTDOUT();
    timeCounter("validation");
    return 0;
}


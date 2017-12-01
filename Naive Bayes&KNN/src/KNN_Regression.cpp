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

#define HINT
#define LINUX
/*
basic functions
- timeCounter
- stdout recovery
*/
// timer
const double eps = 1e-7;
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
const int MAX_WORD = 1e7;
int wordCount[MAX_WORD];
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
    friend ostream& operator<<(ostream& out, rowData& s);
}rD, rV;

ostream& operator<<(ostream& out, rowData& s)
{
    out << "{";
    s.iter = s.IDs.begin();
    out << rdict[s.iter->first] << "(" << s.iter->second << ")";
    ++s.iter;
    for(; s.iter != s.IDs.end(); ++s.iter)
    {
        out << " | " << rdict[s.iter->first] << "(" << s.iter->second << ")";
    }
    out << "}";
    return out;
}


vector<rowData> Library, standard, predict;

bool readDataTorD()
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
    scanf("%lf,%lf,%lf,%lf,%lf,%lf", &rD.Emotion[0],
    &rD.Emotion[1], &rD.Emotion[2], &rD.Emotion[3],
    &rD.Emotion[4], &rD.Emotion[5]);
    return (reader = getchar()) != EOF;
}
// 使用了统计词频的矩阵，比OneHot多了一个出现次数
inline double cosDistance(map<int, int> v1, int size1, map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double same = 0;
    while(i1 != v1.end() && i2 != v2.end())
    {
        if(i1->first == i2->first)
        {
            same +=  i1->second * i2->second; // 相同个数是乘积
            i1++; i2++;
        }
        else if(i1->first < i2->first)
            i1++;
        else
            i2++;
    }
    return same; 
}
inline double cosDistance_tfidf(map<int, int> v1, int size1, map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double same = 0, tfidf1, tfidf2;
    double sum1 = 0, sum2 = 0;
    for(; i1 != v1.end(); ++i1)
    {
        tfidf1 += wordCount[i1->first] ? i1->second * 1.0 / wordCount[i1->first] : i1->second;
        sum1 += tfidf1 * tfidf1;
    }   
    for(; i2 != v2.end(); ++i2)
    {
        tfidf2 += i2->second * 1.0 / wordCount[i2->first];
        sum2 += tfidf2 * tfidf2;
    } 
    sum1 = sqrt(sum1);
    sum2 = sqrt(sum2);

    i1 = v1.begin();
    i2 = v2.begin();
    while(i1 != v1.end() && i2 != v2.end())
    {
        tfidf1 = wordCount[i1->first] ? i1->second * 1.0 / wordCount[i1->first] : i1->second;
        tfidf2 = i2->second * 1.0 / wordCount[i2->first];
        tfidf1 /= sum1;
        tfidf2 /= sum2;
        if(i1->first == i2->first)
        {
            same +=  tfidf1 * tfidf2; // 相同个数是乘积
            i1++; i2++;
        }
        else if(i1->first < i2->first)
            i1++;
        else
            i2++;
    }
    return same; 
}
inline double one_hotDistance(map<int, int> v1, int size1,map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double diff = 0;
    while(i1 != v1.end() && i2 != v2.end())
    {
        if(i1->first == i2->first)
        {
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            diff ++;
            i1++;
        }
        else
        {
            diff ++;
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        diff++;
        i1++;
    }
    while(i2 != v2.end())
    {
        diff++;
        i2++;
    }
    return diff ; 
}
inline double l1Distance(map<int, int> v1, int size1,map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double diff = 0;
    while(i1 != v1.end() && i2 != v2.end())
    {
        if(i1->first == i2->first)
        {
            diff += abs(i1->second - i2->second) ;
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            if(wordCount[i1->first])
                diff += i1->second;
            i1++;
        }
        else
        {
            if(wordCount[i2->first] < 10)
                diff += i2->second;
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        if(wordCount[i1->first])
            diff += i1->second;
        i1++;
    }
    while(i2 != v2.end())
    {
        if(wordCount[i2->first] < 10)
            diff += i2->second;
        i2++;
    }
    return diff ; 
}
inline double l1Distance_tfidf(map<int, int> v1, int size1, map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double diff = 0, tfidf1, tfidf2;
    while(i1 != v1.end() && i2 != v2.end())
    {
        tfidf1 = wordCount[i1->first] ? i1->second * 1.0 / wordCount[i1->first] : i1->second;
        tfidf2 = i2->second * 1.0 / wordCount[i2->first];
        if(i1->first == i2->first)
        {
            diff += abs(tfidf1 - tfidf2);
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            if(wordCount[i1->first])
                diff += tfidf1;
            i1++;
        }
        else
        {
            if(wordCount[i2->first] < 10)
                diff += tfidf2;
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        if(wordCount[i1->first])
            diff += tfidf1;
        i1++;
    }
    while(i2 != v2.end())
    {
        if(wordCount[i2->first] < 10)
            diff += tfidf2;
        i2++;
    }
    return diff ; 
}
inline double l2Distance(map<int, int> v1, int size1, map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double diff = 0;
    while(i1 != v1.end() && i2 != v2.end())
    {
        if(i1->first == i2->first)
        {
            diff += (i1->second - i2->second)*(i1->second - i2->second) * 1.0 / wordCount[i1->first];
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            if(wordCount[i1->first])
                diff += i1->second * i1->second * 1.0/ wordCount[i1->first];
            i1++;
        }
        else
        {
            if(wordCount[i2->first] < 10)
                diff += i2->second * i2->second * 1.0/ wordCount[i2->first];
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        if(wordCount[i1->first])
            diff += i1->second * i1->second * 1.0/ wordCount[i1->first];
        i1++;
    }
    while(i2 != v2.end())
    {
        if(wordCount[i2->first] < 10)
            diff += i2->second * i2->second * 1.0/ wordCount[i2->first];
        i2++;
    }
    return sqrt(diff/size1/size2); 
}
inline double l2Distance_tfidf(map<int, int> v1, int size1, map<int, int> v2, int size2)
{
    mit i1 = v1.begin();
    mit i2 = v2.begin();
    double diff = 0, tfidf1, tfidf2;
    while(i1 != v1.end() && i2 != v2.end())
    {
        tfidf1 = wordCount[i1->first] ? i1->second * 1.0 / wordCount[i1->first] : i1->second;
        tfidf2 = i2->second * 1.0 / wordCount[i2->first];
        if(i1->first == i2->first)
        {
            diff += (tfidf1 - tfidf2) * (tfidf1 - tfidf2);
            //diff += (i1->second - i2->second)*(i1->second - i2->second) * 1.0 / wordCount[i1->first];
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            if(wordCount[i1->first])
                diff += tfidf1 * tfidf1;
                //diff += i1->second * i1->second * 1.0/ wordCount[i1->first];
            i1++;
        }
        else
        {
            if(wordCount[i2->first] < 10)
                diff += tfidf2 * tfidf2;
                //diff += i2->second * i2->second * 1.0/ wordCount[i2->first];
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        if(wordCount[i1->first])
            diff += tfidf1 * tfidf1;
            //diff += i1->second * i1->second * 1.0/ wordCount[i1->first];
        i1++;
    }
    while(i2 != v2.end())
    {
        if(wordCount[i2->first] < 10)
            diff += tfidf2 * tfidf2;
            //diff += i2->second * i2->second * 1.0/ wordCount[i2->first];
        i2++;
    }
    return diff; 
}
const int TOTEMOTION = 6;

void KNN(int k, const rowData & rD, int type)
{
    priority_queue<pair<double, int> > maxK;
    while(maxK.size()) maxK.pop();

    double emotion[6];
    for(int i = 0; i < TOTEMOTION; ++i) emotion[i] = 0;

    for(int st = 0; st < Library.size(); ++st)
    {
        switch(type)
        {
            case 1 : maxK.push(make_pair(eps + cosDistance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 2 : maxK.push(make_pair(eps + cosDistance_tfidf(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 3 : maxK.push(make_pair(eps + 1.0/l1Distance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 4 : maxK.push(make_pair(eps + l1Distance_tfidf(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 5 : maxK.push(make_pair(eps + 1.0/l2Distance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 6 : maxK.push(make_pair(eps + l2Distance_tfidf(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break;
            case 7 : maxK.push(make_pair(eps + one_hotDistance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords),st));
                break; 
            default: cerr << "FAULT TYPE" << endl;
        }
    }
    for(int i = 0; i < k; ++i)
    {
        if(maxK.size())
        {
            double samebility = maxK.top().first;
            int idx = maxK.top().second;
            maxK.pop();
            for(int i = 0; i < 6; i++)
            {
                emotion[i] += Library[idx].Emotion[i] * samebility;// * distance when use cos
            }
        }
    }

    // standard score
    // 归一化， 这里最后一步没有减去均值u，防止出现负数
    double u, d, sum = 0;
    for(int i = 0; i < TOTEMOTION; ++i)
    {
        sum += emotion[i];
    }
    u = sum / TOTEMOTION;
    sum = 0;
    for(int i = 0; i < TOTEMOTION; ++i)
    {
        sum += (emotion[i] - u) * (emotion[i] - u);
    }
    d = sqrt(sum / TOTEMOTION);
    for(int i = 0; i < TOTEMOTION; ++i)
        emotion[i] = (emotion[i] ) / d;
    // 尺度放缩
    double maxV = emotion[0], minV = emotion[0]; // delete negative
    for(int i = 0; i < TOTEMOTION; ++i)
    {
        minV = min(minV, emotion[i]);
        maxV = max(maxV, emotion[i]);     
    }
    for(int i = 0; i < TOTEMOTION; ++i)
    {    
        emotion[i] = (emotion[i] - minV) / (maxV - minV);
    }
    // change sum to 1:
    sum = 0;
    for(int i = 0; i < TOTEMOTION; ++i)
    {
        sum += emotion[i];
    }
    for(int i = 0; i < TOTEMOTION; ++i)
    {
        emotion[i] /= sum;
    }
    // 存储，用于计算相关系数
    for(int i = 0; i < TOTEMOTION; i++)
    {
        if(i==0)
            printf("%f",emotion[i]);
        else
            printf(",%f",emotion[i]);
        rV.Emotion[i] = emotion[i];
    }
    puts("");
}

double maxe[TOTEMOTION];
void train()
{
    freopen(train_addr.c_str(), "r", stdin);
    char reader = '\0';
    while((reader = getchar()) != '\n');
    while(readDataTorD())
    {
        #ifdef HINT
        if(rD.Emotion[0] > 0.4
            || rD.Emotion[1] > 0.25
            || rD.Emotion[2] > 0.6
            || rD.Emotion[4] > 0.9)
            Library.push_back(rD);
        #endif
        Library.push_back(rD);
    }
    Library.push_back(rD);
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
void validation(int type)
{
    double ans = 0;
    int anspos = 0;
    
    for(int k = 1; k <= 200; ++k)
    {
        predict.clear();
        for(int i = 0; i < standard.size(); ++i)
        {
            KNN(k, standard[i], type);
            predict.push_back(rV);
        }   
        double cor[TOTEMOTION];
        double corsum;
        corsum = 0;
        memset(cor, 0, sizeof(cor));
        for(int i = 0; i < TOTEMOTION; ++i)
        {
            cor[i] = corr(i);
            corsum += cor[i];    
        }
        if(corsum > ans)
        {
            ans = corsum;
            anspos = k;
        }
        if(k % 10 == 0)
        {
            cerr << "At iteratoration " << k << " maxAns = " << ans / TOTEMOTION
            << " maxAnsPos = " << anspos << endl;
        }
        // output each line correction
        printf("%d,%lf", k, corsum / TOTEMOTION);
        for(int i = 0; i < TOTEMOTION ; ++i) printf(",%lf", cor[i]);
        puts("");
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
    freopen("test.csv","w",stdout);
    int k = 211;
    char reader;
    freopen(test_addr.c_str(), "r", stdin);
    while((reader = getchar()) != '\n');
    standard.clear();
    while(readTestDataTorD())
    {
        standard.push_back(rD);
    }
    standard.push_back(rD);
    for(int i = 0; i < standard.size(); ++i)
    {
        KNN(k, standard[i], 2);
    } 
}
int main(int argc, char** argv)
{
    srand((unsigned int)(time(NULL)));
    startProgramTimer();
    train();
    timeCounter("training");
    test();
    timeCounter("Testing");
    /*
    freopen("./cos.csv", "w", stdout);
    validation(1);
    timeCounter("cos");

    freopen("./cos_tfidf.csv", "w", stdout);
    validation(2);
    timeCounter("cos-tfidf");

    freopen("./l1.csv", "w", stdout);
    validation(3);
    timeCounter("l1");

    freopen("./l1_tfidf.csv", "w", stdout);
    validation(4);
    timeCounter("l1tfidf");

    freopen("./l2.csv", "w", stdout);
    validation(5);
    timeCounter("l2");

    freopen("./l2_tfidf.csv", "w", stdout);
    validation(6);
    timeCounter("l2tfidf");

    freopen("./one_hot.csv", "w", stdout);
    validation(7);
    timeCounter("one_hot");

    recoverySTDOUT();

    timeCounter("validation");
    */
    return 0;
}


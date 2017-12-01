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
//#define HINT_2
#define HINT_3
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
const int MAX_WORD = 1e7;
int wordCount[MAX_WORD];
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


void train()
{
    freopen("./DATA/classification_dataset/train_set.csv", "r", stdin);
    char reader = '\0';
    while((reader = getchar()) != '\n');
    while(readDataTorD(true)) 
    {
        Library.push_back(rD);
    }
    Library.push_back(rD);
}

typedef map<int,int>::iterator mit;
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
inline double ont_hotDistance(map<int, int> v1, int size1,map<int, int> v2, int size2)
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
            //diff += abs(i1->second - i2->second) * 1.0 / wordCount[i1->first];
            i1++; i2++;
        }
        else if(i1->first < i2->first)
        {
            if(wordCount[i1->first])
                diff += 1.0/ wordCount[i1->first];
            i1++;
        }
        else
        {
            if(wordCount[i2->first] < 10)
                diff += 1.0/ wordCount[i2->first];
            i2++;
        }
    }
    while(i1 != v1.end())
    {
        if(wordCount[i1->first])
            diff += 1.0/ wordCount[i1->first];
        i1++;
    }
    while(i2 != v2.end())
    {
        if(wordCount[i2->first] < 10)
            diff += 1.0/ wordCount[i2->first];
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
const double eps = 1e-10;
int KNN(const int &k, const rowData & rD)
{
    priority_queue<pair<double, int> > maxK; // 使用优先队列，得到前k个值
    while(maxK.size()) maxK.pop();

    // counting distance， 距离计算，使用OneHot×IDF方法
    for(int st = 0; st < Library.size(); ++st)
    {
        
        // 放入队列中，由于最初的队列是大顶堆，取负数
        maxK.push(make_pair(
            //cosDistance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords)
            1.0/(eps+l1Distance(rD.IDs, rD.totWords, Library[st].IDs, Library[st].totWords))
            , Library[st].Emotion));
    }
    // voting， 相当于取众数
    vector<double> voting_pool(emotionID + 1, 0);
    for(int i = 0; i < k; ++i)
    {
        if(maxK.size())
        {
            voting_pool[maxK.top().second]+=maxK.top().first;
            maxK.pop();
        }
    }

    int ans = 1;    // 众数标签
    double maxV = -1, minV = voting_pool[1];  // 众数值
    for(int i = 1; i <= emotionID; ++i)
    {
        if(voting_pool[i] > maxV)
        {
            maxV = voting_pool[i];
            ans = i;
        }
    }
    return ans;
}
void validation()
{
    freopen("./DATA/classification_dataset/validation_set.csv", "r", stdin);   
    char reader = '\0';
    vali.clear();
    while((reader = getchar()) != '\n');
    while(readDataTorD()) vali.push_back(rD);
    timeCounter("read validation");
    freopen("./cos_cf.csv", "w", stdout);
    for(int k = 1; k < 100; k ++)
    {
        int correctSentence = 0;
        for(int j = 0; j < vali.size(); ++j)
        {
            correctSentence += (KNN(k, vali[j]) == vali[j].Emotion);
        }
        cout << k << "," << (1.0 * correctSentence) / vali.size() << endl;
        cerr << "test k = " << k << " correct rate = " << (1.0 * correctSentence) / vali.size() << endl;
        
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

    freopen("./cv.csv", "w", stdout);
    int k = 19;
    for(int j = 0; j < vali.size(); ++j)
    {
        int ans = KNN(k, vali[j]);
        cout << emotionrDict[ans] << endl;        
    }
}

/*
main function.
*/
int main(int argc, char** argv)
{
    srand((unsigned int)(time(NULL)));
    timeCounter("begining");
    startProgramTimer();
    train();
    timeCounter("training");
    // validation();
    // recoverySTDOUT();
    // timeCounter("predict");
    test();
    recoverySTDOUT();
    timeCounter("test");
    
    return 0;
}


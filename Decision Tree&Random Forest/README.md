| ![](./sysu.png) | 人工智能实验 |
| --------------- | ------ |
|                 |        |

##中山大学数据科学与计算机学院移动信息工程专业

##人工智能本科生实验报告

###（2017-2018学年秋季学期）
课程名称：**Artificial Intelligence**

| 教学班级 | **周五5-6** | 专业（方向） | 移动互联网 |
| ---- | --------- | ------ | ----- |
| 学号   | 15352285  | 姓名     | 任磊达   |



[TOC]

## 一、 实验题目

> 决策树

## 二、 实验内容

### 算法原理

#### 简介

决策树是一种树形决策模型，由于每一次通过一个具体特征进行决策，使得模型效率极高，可以在线性时间完成决策。但是由于树上分叉多，叶子节点以（随特征数目）指数级别增长，也就是模型参数指数级别增长，因此极易过拟合。

另外，我认为决策树是一种对于KNN的扩展，KNN是直接定义两个特征之间的距离，而剪枝之后的决策树是通过最优的几个特征是否相同来判断类别，并且最后的结果产生方式都是**多数投票**。

#### 训练（建树）

构建一棵决策树首先需要明白树的组成部分：节点的数据结构

1. 当前节点数据集，对于非叶子节点并不是必须存储，而是需要使用，以辅助生成子节点的数据集以及剪枝。对于叶子节点而言节点数据集是必须要用的，以为判断依据是在叶子节点取该节点数据集的众数。
2. 该节点是否为叶子节点，如果不是，所关联的所有叶子节点。
3. 该节点对于叶子节点的映射，即划分依据。

于是在训练过程中，我们采用递归的方法，从根节点向叶子节点进行构建：

1. 初始化当前节点，从父节点继承训练集的一个子集，以及剩余的可分类的特征。
2. 从当前待分类的特征当中，选择**最优**的特征进行数据**划分**。
3. 如果**存在**划分，对于每个数据划分创建子节点，递归建树。

上述步骤中关键的三个词为最优，划分，存在；下面会较为详细的讲述这三个点：

##### 划分依据

- ID3方法：

  - 我们通过信息熵定义样本集合纯度，对于当前样本集合D中第k类样本（标签）所占比例为P_k,定义信息熵为：
    $$
    H(D) = -\sum_{k=1}^{|Y|}p_klog_2p_k
    $$

  - 当使用了a特征之后，可以计算如果通过a划分之后集合的条件熵H(D|A)：
    $$
    H(D|A)=\sum_{a\in A} p(a)H(D|A=a)
    $$

  - 由于H(D|A=a)<=H(D),那么可论证的，H(D|A)<=H(D),于是对于每一种划分有一个非负的信息增益Gain(D, A) = H(D) - H(D|A)

  - 使用信息增益最大的点对于数据进行划分，特别的，当H(D|A)为0的时候，说明那一个种分类可以将数据完全划分为不同标签的多个类别。

- C4.5方法：

  - 由于ID3最后一点强调的，当一种类别标签数目较多的时候，会对于信息增益有一个增益，于是这里使用C4.5方法对于标签类目进行一个制衡，使用信息增益率最大的特征作为划分依据：

  - 信息增益率gRatio(D,A)定义为：
    $$
    gRatio(D,A)=\frac{Gain(D,A)}{SplitInfo(D,A)}
    \\
    SplitInfo(D,A)=-\sum_{j=1}^{v}\frac{|D_j|}{|D|}log_2\frac{|D_j|}{|D|}
    $$

  - 值得注意的是，可以[Quilan, 1993]结合这两种方法，仅仅对于信息增益大于平均值的特征计算增益率。

- CART方法：

  - 使用Gini系数`gini(D,A)`衡量特征选择后的不确定性：
    $$
    gini(D,A)=\sum_{j=1}^{v}p(A_j)gini(D_j|A=A_j)\\
    gini(D_j|A=A_j)=\sum_{i=1}^np_i(1-p_i)=1-\sum_{i=1}^np_i^2
    $$

  - Gini的具体含义是指从当前数据集里面抽两个数，两个数**不同**的概率。因此Gini系数越小，特征越划分之后数据集纯度提升越大。

##### 叶子节点终止标志

- 预剪枝，继续扩展在验证集上没有性能提升。
- 当前节点全部**数据**属于同一类别，不能再划分。这样的话不会出现空集。
- 没有进一步划分的特征，也就是当前给定的**特征集合**已经使用完。

##### 连续特征的二划分

- 排序所有特征值，记排序后的结果为`a0, a1, …, an`。
- 对于k=1~n的所有ak，使用$\frac{a_k + a_{k-1}}{2}$ 作为阈值来划分。
- 取最优划分的结果， 作为实际的划分边界及划分结果。

####预测（深度优先搜索）

从根节点开始递归搜索，直到叶子节点为止：

- 如果当前为叶子节点，停止搜索，返回当前节点数据集中的众数。
- 否则依据当前节点的划分方法，递归进入下一个节点，返回下一个节点返回的结果。

### 伪代码

由于代码当中实现的是认作连续值，在伪代码当中也实现连续值划分的逻辑。

代码使用`C++`完成，伪代码使用类`python`格式展示。

```python
# calcu ID3, C4.5, CART best divide feature
def CALCU_ID3(D, A):
	NowEntropy 	:= -SUM(for label in D: p(label) * log2(p(label)))
    NowGain 	:= 0
    NowDiv		:= -1 # meaningless initial value
   	FeaValue	:= ObtainValue(D, A)
    FeaValue.sort()
    for div in FeaValue:
        midV 	:= (div+Lastdiv) / 2
        D1 		:= data.feature(A) < midV
        D2		:= data.feature(B) >=midV
        thisEntropy 	
        :=  -SUM(for label in D1: p(label) * log2(p(label)))
            -SUM(for label in D2: p(label) * log2(p(label)))
        if NowEntropy - thisEntropy > Gain:
            NowGain :=  NowEntropy - thisEntropy
            NowDiv  :=  midV
    return NowGain, NowDiv
    
# generate decision tree recursively
# D: data, F: feature set, Type: ID3, C4.5 or CART
def GenTree(D, F, Type):
    # save choose feature, gain, gain ratio, and gini.
    Csfeature := NULL
    CsGain 	  := 0
    CsGainR   := 0
    CsGini    := inf
    CsDiv     := -1 # meaningless initial value
    # choose best feature
    for fea in F:
        if Type == 'ID3':
            # calcu gain
            NowGain, NowDiv := CALCU_ID3(D, fea)
            # update
            if NowGain > CsGain:
                Csfeature 	:= fea
                CsGain 	  	:= NowGain
                CsDiv  		:= NowDiv
        elif Type == 'C4.5':
            # calcu gain
            NowGainR, NowDiv := CALCU_C4_5(D, fea)
            # update
            if NowGainR > CsGainR:
                Csfeature 	:= fea
                CsGainR 	:= NowGainR
                CsDiv  		:= NowDiv
        else # for CART, calcu Gini
        	# calcu gain
            NowGini, NowDiv := CALCU_CART(D, fea)
            # update
            if CsGini > NowGini:
                Csfeature 	:= fea
                CsGini	 	:= NowGini
                CsDiv  		:= NowDiv
   	# for leaf node
    Lcnt := 0
    Rcnt := 0
    # for subTree
    LDiv := Empty Data Set
    RDiv := Empty Data Set
    NextF:= F - CsFeature
	if Csfeature != NULL:
    	for d in Data:
            # count data label, save for leavf dicision
            if Label[d] == 1: 
                Rcnt := Rcnt + 1
            else:
                Lcnt := Lcnt + 1
            # count subTree dataset
            if Data[d].Feature(Csfeature) < NowDiv:
                LDiv.insert(Data[d])
            else:
                RDiv.insert(Data[d])
    if isEmpty(LDiv) or isEmpty(RDiv) or isEmpty(NextF):
        return Tree(Lcnt, Rcnt) # leaf node
    else:
        Tree.lTree := GenTree(LDiv, NextF, Type):
        Tree.rTree := GenTree(RDiv, NextF, Type):
        return Tree.setFeature(CsFeature).setDivide(Cs)
    
# predict data recursively, a method in Tree Class
def predict(D):
    if isLeaf(this):
        return Lcnt > Rcnt ? -1 : 1
    else
    	if D.Feature(CsFeature) < NowDiv:
            return LTree.predict(D)
        else: 
        	return RTree.predict(D)
    
# train and predict, a example of ID3
def main():
    DecisionTree := GenTree(Data, ALLCharator, 'ID3');
    DecisionTree.predict(Data1);
```

### 关键代码

由于决策树构造部分在伪代码中描述的较为详细，下面关键代码中补充其他的部分

####指标计算

由于指标计算的相似性，以及较为繁复的划分代码，在实际编程当中，可以一次同时计算多个指标：

```c++
double DecisionNode::cal_upd_Entropy
		(int fea, double & divideThread, double & info, double & ginisum)
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
```

####随机森林

整个C++实现当中，使用`bitset`进行数据及特征的标签，于是可以很好的进行对于数据和特征进行随机下采样：

```c++
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

```

####生成验证集

由于给定数据只给出了训练集和验证集，于是需要手动生成验证集对于模型进行验证：

以及进一步的剪枝过程是通过模型再验证集上的表现而决定：

```c++
void DecisionTree::GenTrain_Valid(double alpha)
{
    // new dataset.
    trainData.clear();
    validData.clear();
    // sample validation rate = alpha
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
}
```

### 创新点&优化

1. 数据结构的优化，通过一次读入划分数据，使用索引操作进行建树以及预测，大大降低空间复杂度。
2. 尝试使用随机森林进行`bagging`，增强模型的鲁棒性，**但是**在没有剪枝的情况下，随机森林表现不如单棵树，原因在于单棵树以及对于模型过拟合了，随机森林数目很少的时候就会对于集合过拟合。 

## 三、 实验结果及分析

### 实验结果展示示例

使用实验要求样例：

| 长鼻子  | 大耳朵  | 是否为大象 |
| ---- | ---- | ----- |
| 1    | 1    | 1     |
| 0    | 1    | 0     |
| 1    | 0    | 0     |
| 0    | 0    | 0     |

#### ID3

经验熵
$$
H(D)=-\frac{1}{4}log(\frac{1}{4})-\frac{3}{4}log(\frac{3}{4})=0.811278\\

H(D|A=长鼻子)= \frac{2}{4}(-\frac{1}{2}log(\frac{1}{2})-\frac{1}{2}log(\frac{1}{2}))+\frac{2}{4}(-0*log(\frac{1}{2})-\frac{2}{2}*0))=0.5
\\
Gain(D, A=长鼻子)=H(D)-H(D|A=长鼻子)=0.311278
\\
H(D|A=大耳朵)= \frac{2}{4}(-0*log(\frac{1}{2})-\frac{2}{2}*0))+\frac{2}{4}(-\frac{1}{2}log(\frac{1}{2})-\frac{1}{2}log(\frac{1}{2}))=0.5
\\
Gain(D, A=大耳朵)=H(D)-H(D|A=大耳朵)=0.311278
$$
故实际上这两种划分信息增益一致，选取信息增益最大的特征作为决策点的原则，选取其中任意一个即可。

#### C4.5

计算数据集关于两个特征的熵：
$$
SplitInfo(D, A=长鼻子)=-\frac{2}{4}log\frac{2}{4}-\frac{2}{4}log\frac{2}{4}=1\\
SplitInfo(D, A=大耳朵)=-\frac{2}{4}log\frac{2}{4}-\frac{2}{4}log\frac{2}{4}=1
$$
对应的信息增益率为：
$$
gainRatio(D, A=长鼻子)=\frac{Gain(D, A=长鼻子)}{SplitInfo(D, A=长鼻子)}=0.311278\\
gainRatio(D, A=大耳朵)=\frac{Gain(D, A=大耳朵)}{SplitInfo(D, A=大耳朵)}=0.311278
$$
依据C4.5方法，选取对应信息增益率最高的特征进行划分，由于上述两个特征增益率相等，于是可以任意选一个。

####CART

计算每个指标的Gini系数
$$
Gini(D, A=长鼻子)=\frac{1}{2}Gini(D_{长鼻子=0}|A=长鼻子)+\frac{1}{2}Gini(D_{长鼻子=1}|A=长鼻子)\\
=\frac{1}{2}(1-(\frac{1}{2})^2-(\frac{1}{2})^2)+\frac{1}{2}(1-(1)^2-(0)^2)
\\=\frac{1}{4}
$$
由于数据的对称性，不再赘述`大耳朵`特征的计算过程，结果也是1/4.由于CART是选取**最小**的特征进行划分依据，从其中任意选择一个即可。



于是对于这样三个模型而言，给定的数据集在两个特征上划分能力相同。但是通过运算公式，我们可以分析验证每个模型的正确性。

### 评测指标展示及分析

使用三种不剪枝的决策树以及对应的随机森林

随机森林限制树高度3层，树数目15棵，数据采样率0.1，特征采样率0.5

| 模型类型        | Accuracy | Precision | Recall   | F1       |
| ----------- | -------- | --------- | -------- | -------- |
| ID3         | 0.645833 | 0.621212  | 0.611940 | 0.616541 |
| ID3 Forest  | 0.602649 | 0.720000  | 0.253521 | 0.375000 |
| C4.5        | 0.632258 | 0.514286  | 0.610169 | 0.558140 |
| C4.5 Forest | 0.606897 | 0.522523  | 0.935484 | 0.670520 |
| Cart        | 0.635135 | 0.542857  | 0.633333 | 0.584615 |
| Cart Forest | 0.601307 | 0.490385  | 0.864407 | 0.625767 |

可以看到，三种模型再没有剪枝的情况下，较为简单直接地ID3反而表现更好；另外对比普通树模型和随机森林模型，树模型有更好的Accuracy和F1值，森林模型有更好的Precision和Recall。


## 四、 思考题

- 决策树有哪些避免过拟合的方法?
    1. 剪枝，分为预剪枝和后剪枝：预剪枝是训练时候判断是否扩展，使用在验证集上表现判断。后剪枝是在建完树之后实行剪枝。
    2. 随机森林，一种对于树的bagging方法，通过随机选择特征和随机选择数据两重随机构建多组不同的子训练集，而后使用每个子训练集训练一颗决策树。最后使用所有决策树结果（sign函数输出）求和求sign，而后得到最终的结果。
    3. 随机森林使用验证集来截取分类效果较好的子树，相当于森林层面上面的剪枝。

- C4.5相比于ID3的优点是什么?
   1.  ID3模型可能对于可取值数目较多的属性有所偏好，而C4.5除以一个IV函数来抑制数目对于增益率的影响，从而平衡了可取值数目较少的属性。
   2.  事实上，在[Quinlan, 1993]论文当中描述的C4.5考虑了对于数目较少属性的一个抑制问题，也就是要求C4.5选择的属性在ID3指标上面高于平均值。

- 如何用决策树来判断特征的重要性?
    1. 显然的，Gain，Gain_ratio，Gini三个指标是对于特征重要性的一种描述，Gain，Gain_ratio越大，在先验层面认为特征越为重要，Gini越小认为特征越为重要。
    2. 引入随机森林了之后我们可以统计每个特征存在对于数据的影响，因为随机森林对于特征进行了一次下采样，不同的树选取的特征是不同的。

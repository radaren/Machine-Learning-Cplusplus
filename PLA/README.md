| ![](http://a3.att.hudong.com/30/34/01100000000000144722342994012_140.jpg) | 人工智能实验 |
| ---------------------------------------- | ------ |
|                                          |        |

##中山大学数据科学与计算机学院移动信息工程专业

##人工智能本科生实验报告

###（2017-2018学年秋季学期）
课程名称：**Artificial Intelligence**

| 教学班级 | **周五5-6** | 专业（方向） | 移动互联网 |
| ---- | --------- | ------ | ----- |
| 学号   | 15352285  | 姓名     | 任磊达   |



[TOC]

##一、 实验题目

> 感知机学习算法


##二、 实验内容

### 算法原理

####二分感知机分类原理

感知机预测的函数是对于一个**x** 向量每一个元素的加权组合，组合的结果大于threshold的时候预测结果为1（正样本），否则预测结果为-1(负样本）：
$$
predict(\vec x) =  
\begin{cases}  
1, &\text{if  $\vec w^T\vec x > threshold$} \\
-1, &\text{else}  
\end{cases}
$$
进一步，可以对于特征向量，添加一个偏置项1，来将`threshold`并入特征部分：
$$
\vec{x_0} = [a_0,a_1,...a_n]^T\\
\vec{x} = [1,a_0,a_1,...a_n]^T
\\
predict(\vec x) =  
\begin{cases}  
1, &\text{if  $\vec w^T\vec x > 0$} \\
-1, &\text{else}  
\end{cases}
$$

### 二分感知机训练原理

感知机是通过迭代的方法训练，对于每一次迭代，使用当前存储的**w**对于每一个（训练集）样本进行预测，当预测结果与训练标签不同的时候，使用当前训练样本，更新**w**中每个项,直到收敛/完全预测正确/达到理想迭代次数，在第t次迭代中：
$$
\vec w_{t+1} =
\begin{cases}  
\vec w_t + y_i \vec x_i, &\text{if  $\vec w^T\vec x_i \neq y_i$} \\
\vec w_{t}, &\text{else}  
\end{cases}
$$
这里使用的损失函数是误分类点到划分超平面的距离：
$$
loss(\vec w) =
\begin{cases}  
-\sum_i y_i(\vec w ^T\vec x_i), &\text{if  $\vec w ^T\vec x_i \neq y_i$} \\
0, &\text{$\vec w ^T\vec x_i = y_i$ }  
\end{cases}
$$
由于更新的是**w**向量，转化为**w**的函数：
$$
f(\vec w) = y_i*predict(\vec x) =y_i \vec w^T \vec x \\ =y_i( w_0x_0 + w_1x_1 + \dots+w_nx_n)\\
$$
因此，预测成功时候为是分别对于**w** 中的每个项求导数，而后对于错误样本，使用一阶偏导数,依据分类错误类型，(对于误分类样本)进行更新：
$$
\vec {w_{t+1} }:= \vec{w_t} + \sum_i \frac{\partial f(\vec w_t)}{\partial w_i}
\\
where :\frac{\partial \vec w}{\partial w_i} =y_i x_i\\
$$

### pocket

在多次迭代中，保存分类结果最好的**w**作为模型：
$$
\vec w={minarg}_{loss(\vec w)} \vec w_i \\for\ w_i \ in\ each\ itreation
$$

### bagging

通过随机初始化，训练之后得到多个**w**, 而后使用多个**w** 分别预测，使用所有预测结果进行最终预测:
$$
predict(\vec x)=sign(\sum_i  \vec w_i^T \vec x)
$$

### dropout

一种防止过拟合的策略，对于错误样本以及错误样本中的每个元素，有一定概率不更新：
$$
\vec w_{t+1} =
\begin{cases}  
\vec w_t + y_i \vec x_i, &\text{if  $\vec w^T\vec x_i \neq y_i$ and $random()>DropOutRate$} \\
\vec w_{t}, &\text{else}  
\end{cases}
$$


### 伪代码

感知机分类算法是一种通过对于特征的线性组合预测值而后与阈值相比较而得到结果的一种算法

其训练过程通过错误分类的样本调整某种特征线性加权权值。

```python
# parameter for data
# including bias 1
FEATURE_LEN   := 65
training_iter := 1000

# variables
w := zeros(1, FEATURE_LEN)

# training
for iter in training_iter:
  for trainx, labelx in train_data, train_label:
    Value := sum(trainx .* w)
    PredictLabel := sign(Value)
    if(PredictLabel != labelx):
      for xi in range(FEATURE_LEN):
        w(xi)` := w(xi) + labelx * trainx(xi) 
          
# predicting:
value  := sum(test .* w)
predict:= sign(value)
```

感知机

`pocket`：通过vali集合测试多个w得到最优值

```python
maxw := zeros(1, FEATURE)
maxv := 0
  
# training Framework:
for rand_i in pocket_size:
  w := rands(1, FEATURE_LEN)
  # PLA for w
  ...
  # test in validation set
  correct := 0
  for valix, valiy in Vali_data, Vali_Label :
    if sign(valix .* valix) * valiy > 0:
      correct := correct + 1
  v := accuracy(w , Vali)
  if v > maxv:
    maxw := w
    maxv := v
      
# predicting:
value  := sum(test .* maxw)
predict:= sign(value)
```

voting

求和多个模型的预测（而不是sum），而后作为预测输出，可以显著提高F1值。

### 关键代码截图

#####训练

```c++
int PLA::trainStep(double alpha)
{
    int ErrNum = 0;
    for(int i = 0; i < TrainFeature.size(); ++i)
    {
        double sum = 0;
        for(int k = 0; k < TrainFeature[i].size(); ++k)
        {
            sum += TrainFeature[i][k] * Weight[k];
        } // 加权求和，得到预测值

        if(sum * TrainLabel[i] <= 0)//省略sign函数，优化效率。
        {
            for(int k = 0; k < TrainFeature[i].size(); ++k)
            {
                Weight[k] += alpha * TrainLabel[i] * TrainFeature[i][k];
            }
            ++ErrNum;
        }// incorrect sample, update.
    }
    return ErrNum;
}
```

##### 随机初始化

```c++
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
        Weight.push_back(avg_sum + rand() * 10.0 / RAND_MAX);// 添入随机因素
    }
}
```

##### bagging方法

```c++

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
    voting_pool[val] += (sum>0)?2:-1;// 由于正样本较少，使用2提高优化率
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
  F1        = (Precision + Recall) ? 2 * Precision * Recall/(Precision + Recall):0;

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
    maxP = cell;// 记录最优bagging值
  }
}
```



### 创新点&优化

1. 设置学习率 $\alpha$  (优化了bagging模型的F1值）
   $$
   w(xi)` := w(xi) + \alpha * labelx * trainx(xi)
   $$

2. 使用随机因子进行pocket

3. 对于参数更新使用dropout（对F1提升有显著效果）

4. 尝试bagging策略。



##三、 实验结果及分析

###实验结果展示示例
> 小数据测试：
>
> 由于pocket算法中填入随机因子，结果重复性较原始算法低，小数据测试仅针对原始算法。

训练集：

-4,-1,1
0,3,-1

测试集：

-2,3,?

在一次迭代后收敛（准确率为1）

![](./demo.png)

###评测指标展示即分析

####测试指标

| 指标        | 含义                    |
| --------- | --------------------- |
| Accuracy  | 判断正确测试样本/全部测试样本       |
| Precision | 判断为T的正确样本/判断为T的样本     |
| Recall    | 判断为T的正确样本/标签为T的样本     |
| F1        | Precision和Recall的调和平均 |

####迭代次数

学习率为1,最优解为**0.843**

![](./init.png)

#### pocket

迭代次数为20,随机因子设为10.0,最优解**0.85**

![](./pock.png)

#### bagging数目

1. 平滑了Accu，提高模型鲁棒性。
2. pocket方法最优F1值为`0.482759`，这里dropout优化后最优为**0.496815**

![](./bag.png)

#### alpha vs 迭代次数（无关）

尝试多个迭代次数，发现与alpha无关。

![](./alpha.png)

##四、 思考题

- 有什么其他的手段可以解决数据集非线性可分的问题？
  - 使用非线性函数
  - 利用多个特征组合生成特征

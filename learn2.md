> Fire 生成命令行接口（Command Line Interfaces, CLIs）

```python
#只有一个函数　python name.py --para val
if __name__ == '__main__':
    fire.Fire(cal_days)
#有两个函数，要指定哪个函数　python name.py train --para val
if __name__=='__main__':
    fire.Fire()
#直接传入一个类，可以执行类的函数 python name.py train --para val
if __name__=='__main__':
    fire.Fire(class)
```

> 隐马尔科夫模型

关于时序的概率模型。由一个隐藏的马尔科夫链产生一个不可观测的状态序列，各个状态又生成一个观测，形成观测序列。

由 初始状态概率向量，状态转移概率矩阵和观测概率矩阵决定。

三个问题:

- 概率计算问题　给定模型和观测序列，计算该序列出现的概率　-->前向算法
  - 前向概率:　给定时刻t，观测子序列为o1-ot，状态为qi的概率 $\alpha_t(i)$ 
  - 递推　$\alpha_{t+1}(i)$ 多了ot+1,和状态i,　状态i可以通过所有状态转移过去，所以要累加，之后再用观测概率
  - 终止　累加所有i在最后一个时刻。
- 学习问题　　给定观测序列，估计模型参数，使该序列出现的概率最大
  - 监督学习　直接极大似然估计
- 预测问题　　给定模型和观测序列，求最有可能的状态序列　　-->维特比算法
  - 动态规划　求最优路径（一个路径代表一个状态序列）（状态序列可以是标记序列）-->最优子结构
  - 同时记录最大概率的值和索引，在最后一代时确定最优路径，回溯得到状态序列。

> 条件随机场 CRF

和隐马尔科夫模型最大的不同，这是一个判别模型（从输入序列到输出序列），还是一个对数线性模型。

条件随机场：给定X下的，Y的马尔科夫随机场　--> 概率无向图模型　-->满足成对马尔科夫性　-->P(Y)因式分解

P(Y) -->最大团上的势函数

一般是线性条件随机场　最大团为相邻两个节点（当前节点和前一个节点）

- 参数化形式
  - 在ｘ下的条件概率　归一化的项Z，势函数由转移特征函数（$t_k$ transform）（和当前状态和前一个状态相关）和状态特征函数（$s_k$ state）（只和当前状态相关）构成。　配上权值。

- 简化形式
  - 写成权值和势函数的内积形式　（对所有位置i求和）

- 矩阵形式
  - 定义T（标记的步数）个M为m阶矩阵（标记的取值个数）元素为状态转移函数（对所有的K（特征函数的个数）求和）

三个问题：

- 给定条件随机场P(Y|X)[模型]，输入序列$x$ [条件]，输出序列$y$ [结果]，计算条件概率P(yi|x)，P(yi-1,yi|x) -->前向算法

### 实现：

https://blog.csdn.net/demm868/article/details/103053500

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/teF4oHzZ4IRsHPDicYtlYNZn1FfaLrr5NoZDbsbicmAuDPOj1938ynGyfgCia7iaFib2clrnZPWNcDKFvPOUhO37iauQ/640?wx_fmt=jpeg)



BiLSTM-CRF的输入是词嵌入向量，输出是预测的最有可能的标签序列。



![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/teF4oHzZ4IRsHPDicYtlYNZn1FfaLrr5NYyJhKhFS6bXbtzFuhqn0UYiaxCINXrqwKkxeUvnG5dDEN95EskkYXkA/640?wx_fmt=jpeg)

BiLSTM层的输出表示该单词对应各个类别的分数。如W0，BiLSTM节点的输出是1.5 (B-Person), 0.9 (I-Person), 0.1 (B-Organization), 0.08 (I-Organization) and 0.05 (O)。这些分数将会是CRF层的输入。

:arrow_forward:如果没有CRF层会是什么样，一样可以工作，直接取最大值就行。

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/teF4oHzZ4IRsHPDicYtlYNZn1FfaLrr5NGb4OJTyl9uj8D0fI5vLHKEPWjcWiaiaN7Qicib1hrPUTjg5Mic2nOSUcOHg/640?wx_fmt=jpeg)

但是这样的分类结果可能不太好。

:arrow_forward: CRF可以做什么　

​	CRF层可以学习到句子的约束条件。比如:句子的开头应该是“B-”或“O”，而不是“I-”。

`CRF 层`

损失函数是什么　CRF层中的损失函数包括两种类型的分数，而理解这两类分数的计算是理解CRF的关键。

- Emission score(发射分数（状态分数）) 这些状态分数来自BiLSTM层的输出

  <img src="https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/teF4oHzZ4IRsHPDicYtlYNZn1FfaLrr5Nicr0IcfWkLQfhD4yqIvxnXanf6O3Nn6kkRUjZwEoZ9JxGKY8AReDWTA/640?wx_fmt=jpeg" alt="img" style="zoom:100%;" />

  记$x_{i,y_j}$ 代表状态分数，i是单词的位置索引，yj是类别的索引$x_{i=1,y_j=2}=0.1$ 表示单词w1被预测为B−Organization的分数是0.1。

- 转移分数

  用$t_{y_iy_j}$来表示转移分数 比如，$t_{0,1}=0.9$ 表示从类别B−Person→I−Person的分数是0.9 因此，我们有一个所有类别间的转移分数矩阵。

  为了使转移分数矩阵更具鲁棒性，我们加上START 和 END两类标签。START代表一个句子的开始（不是句子的第一个单词），END代表一个句子的结束。

  下表是加上START和END标签的转移分数矩阵。

  ![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/teF4oHzZ4IRsHPDicYtlYNZn1FfaLrr5NKG7U3NBFaCRCoGUtp4L6JkYMzh5icuvI8jB8icxXvKJxo0yNSYL8Cd8Q/640?wx_fmt=jpeg)

  实际上，转移矩阵是BiLSTM-CRF模型的一个参数。在训练模型之前，你可以随机初始化转移矩阵的分数。这些分数将随着训练的迭代过程被更新，换句话说，CRF层可以自己学到这些约束条件。

  

> 自定义数据集 pytorch

定义一个MyDataset类，继承自Dataset,重写\__len__()   \__getitem__() 

getitem 返回一个字典，数据和标签 

>huggingface.co/transformers bert

- *class* `transformers.``BertConfig`

  存放 [`BertModel`](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel) 的相关配置。默认值会对应于[bert-base-uncased](https://huggingface.co/bert-base-uncased)，继承自 [`PretrainedConfig`](https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig)　

  参数：(*vocab_size=30522*, *hidden_size=768*, *num_hidden_layers=12*, *num_attention_heads=12*, *intermediate_size=3072*, *hidden_act='gelu'*, *hidden_dropout_prob=0.1*, *attention_probs_dropout_prob=0.1*, *max_position_embeddings=512*, *type_vocab_size=2*, *initializer_range=0.02*, *layer_norm_eps=1e-12*, *pad_token_id=0*, *gradient_checkpointing=False*, ***kwargs*)

- *class* `transformers.``BertTokenizer`

  参数：(*vocab_file*, *do_lower_case=True*, *do_basic_tokenize=True*, *never_split=None*, *unk_token='[UNK]'*, *sep_token='[SEP]'*, *pad_token='[PAD]'*, *cls_token='[CLS]'*, *mask_token='[MASK]'*, *tokenize_chinese_chars=True*, ***kwargs*)

  ```
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  ```

- *class* `transformers.``BertModel`(*config*)

  ```python
  model = BertModel.from_pretrained('bert-base-uncased')
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
  outputs = model(**inputs)
  #'bert-base-uncased' 文件路径
  parameters:
      input_ids (torch.LongTensor of shape (batch_size, sequence_length)) 
  ```
  
  parameters:
  
  - **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
  
    ​	Indices of input sequence tokens in the vocabulary.
  
  - **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, optional, defaults to `None`)
  
    - Mask to avoid performing attention on padding token indices
  
  Returns:
  
  - last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`): Sequence of hidden-states at the output of the last layer of the model.
  - pooler_output (`torch.FloatTensor`: of shape `(batch_size, hidden_size)`):Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pre-training.This output is usually *not* a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
  
  

> LSTM




				HTML


​					
​				
​				
​						
​				
			xxxxxxxxx <details>    <summary>已折叠图片</summary>     <img src="htt
	已折叠图片 


> 折叠 typora 字体颜色 代码块 页内跳转

`<details>   `

`		<summary>点击时的区域标题：点击查看详细内容</summary> `

`  	<p> - 测试 测试测试</p> `  

`  </details>`

```html
<span style=‘color:red‘>This is red</span>
```

代码块　三个小点｀

```markdown
[你是谁](#傻狍子)

### 傻狍子
```

**按住ctrl并点击**才能实现效果，几级标题都可以



>nn.LSTM()

#构建网络模型---输入矩阵特征数input_size、输出矩阵特征数hidden_size、层数num_layers
inputs = torch.randn(5,3,10)   ->(seq_len,batch_size,input_size)
rnn = nn.LSTM(10,20,2)    ->   (input_size,hidden_size,num_layers) 类的实例化
h0 = torch.randn(2,3,20)   ->(num_layers* 1,batch_size,hidden_size)
c0 = torch.randn(2,3,20)   ->(num_layers*1,batch_size,hidden_size) 
num_directions=1 因为是单向LSTM
'''
Outputs: output, (h_n, c_n)
'''
output,(hn,cn) = rnn(inputs,(h0,c0))　调用类对象

> 建立软链接

ln -s [源地址] [目标地址]  源地址必须是绝对路径，是要创建快捷方式的文件（夹）；目标地址是快捷方式准备放置的地址，以及名称。例子：

`sudo ln -s ~/Documents/code_set/NLP/Bert-BiLSTM-CRF-pytorch-master/  ~/Desktop/bert-crf`

删除　rm -rf [软链接地址]　上述指令中，软链接地址最后不能含有“/”，当含有“/”时，删除的是软链接目标目录下的资源，而不是软链接本身

> linux Tips

pwd 显示当前目录

查看cuda版本　cat /usr/local/cuda/version.txt

本地在远程服务器上使用python matplotlib画图

```python
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
===
plt.savefig('./plot/'+random_name+'.png')
```

之后在vscode查看生成的文件就行。



> git

初始化一个Git仓库，使用`git init`命令。

添加文件到Git仓库，分两步：

1. 使用命令`git add `，注意，可反复多次使用，添加多个文件；
2. 使用命令`git commit -m `，完成。

- `HEAD`指向的版本就是当前版本，因此，Git允许我们在版本的历史之间穿梭，使用命令`git reset --hard commit_id`。`HEAD^,HEAD~100`
- 穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。
- 要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。

场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令`git checkout -- file`。

场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令`git reset HEAD `，就回到了场景1，第二步按场景1操作。

场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考[版本回退](https://www.liaoxuefeng.com/wiki/896043488029600/897013573512192)一节，不过前提是没有推送到远程库。

`git rm`用于删除一个文件。如果一个文件已经被提交到版本库，那么你永远不用担心误删，但是要小心，你只能恢复文件到最新版本，你会丢失**最近一次提交后你修改的内容**

>[回溯 leetcode]()

适用模型：遍历决策树

模板：

```python
def backtrack(路径，选择列表): #void
    if 终止条件:
        res.append(path)
    for choice in choices:
        if 剪枝条件：
        	continue
        做选择（添加一个节点到path）
        backtrack(路径，选择列表) 走下一步
        撤销选择
       
```

`#其核心就是 for 循环里面的递归，在递归调用之前「做选择」，在递归调用之后「撤销选择」，特别简单。 `

>leetcode

原地

> [linux 分卷压缩　合并解压](#linux 分卷压缩　合并解压)

**1.使用tar分卷压缩**

```
格式 tar cvzf - filedir | split -d -b 50m - filename
```

样例：

```
tar cvzf - ./picture | split -d -b 10m - picture
将./picture 打包，并切割为 10m 的包输出的文件为 filename00、filename01、filename02 ...
```

假设不加filename，则输出文件为 x00、x01、x02 ...

假设不加參数 -d。则输出aa、ab、ac ...

**2.解压分卷**

首先将分卷包合拼

```
cat x* > myzip.tar.gz
```

然后解压

```
tar xzvf myzip.tar.gz
```

样例：

```
cat picture* > picture.tar.gz
tar xzvf picture.tar.gz
```



>Pandas基本操作

- 一维Series, 二维DataFranme. 可以用Series字典生成DataFrame.

- 显示索引和列名　df.index df.columns 查看统计摘要：df.describe()

- 获取数据，选择单列，产生 `Series`，与 `df.A` 等效： `df['A']`  用 [ ] 切片行：`df[0:3]`

  用标签提取一行数据：`df.loc[dates[0]]`, `df.loc[:, ['A', 'B']]` 用标签切片，包含行与列结束点：`df.loc['20130102':'20130104', ['A', 'B']]` 提取标量值：`df.loc[dates[0], 'A']` 快速访问标量，与上述方法等效`df.at[dates[0], 'A']` 

- 用整数位置选择： `df.iloc[3]`(选择第四行)　`df.iloc[3:5, 0:2]` `df.iloc[[1, 2, 4], [0, 2]]` 显式整行切片： `df.iloc[1:3, :]` 

- 提取值：`df.iloc[1, 1]` `df.iat[1, 1]`

- 布尔索引　用单列的值选择数据：` df[df.A > 0]` (如果列A 的数据> 0，这一行被选出)

  - 选择 DataFrame 里满足条件的值：`df[df > 0]` 该元素满足条件保留，否则NaN。
  - 用 [isin()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html#pandas.Series.isin)[ ](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html#pandas.Series.isin) 筛选：`df2[df2['E'].isin(['two', 'four'])]` (相当于枚举值)

- 赋值　`df.at[dates[0], 'A'] = 0` `df.iat[0, 1] = 0` `df.loc[:, 'D'] = np.array([5] * len(df))`

- 合并　结合（Concat）`pieces = [df[:3], df[3:7], df[7:]]  pd.concat(pieces)`

  - `pd.concat([df1,df2],index=0)` 合并多个df

  - 连接　`left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]}) `

    `	right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})`

    ` 	pd.merge(left, right, on='key')`

  - 追加　`df.append(s, ignore_index=True)`
  
  - 分组　按条件把数据分割成多组　为每组单独应用函数　将处理结果组合成一个数据结构
  
- nan: `np.isnan(names.iloc[1,4])` 判断nan

- 删除指定行、列

  - ![img](https://img2018.cnblogs.com/blog/1588501/201901/1588501-20190124203207898-484233341.png)

  ```python
  df21=df1.drop(labels=[1,3],axis=0)
  ```
  
- 单列可以直接转list  .tolist()

- 遍历行元素　`for i, row in ratings.iterrows():` 

- 根据另一个df 给这个df添加列：

  - ```python
        for i, row in ratings.iterrows():
            u = row['user_id']
            tmp = u_meta[u_meta.user_id ==u]['occp'].item()
            res.append(tmp)
        ratings['occp']=res
    ```

- 去重（取集合）

  `tmp_df.drop_duplicates(subset=['user_id'],keep='first',inplace=False)['user_id']`  

- 排序

>numpy

NumPy的主要对象是**同构多维数组**。它是一个元素表（通常是数字），所有类型都相同，由非负整数**元组**索引。在NumPy维度中称为 *轴* 

NumPy的数组类被调用`ndarray`。

- **ndarray.ndim** - 数组的轴（维度）的个数。
- **darray.shape** - 数组的维度。这是一个整数的元组
- **ndarray.size** - 数组元素的总数。
- **ndarray.dtype** - 一个描述数组中元素类型的对象 例如numpy.int32、numpy.int16和numpy.float64.
- **ndarray.itemsize** - 数组中每个元素的字节大小

### 数组创建

- ```python
  a = np.array([2,3,4])
  一个常见的错误 a = np.array(1,2,3,4)
  ```

  ```python
  np.zeros( (3,4) )  ones
  np.arange( 10, 30, 5 )
  ```


#### 保存读取

- `np.save('name',obj)` 	`a = np.load('name', allow_pickle=True) `可以是字典,之后要加上 `a.item() `
- 

> 剑指offer　

- 有限状态机（20)
  - 根据字符类型和合法数值的特点，先定义状态，再画出状态转移图，最后编写代码即可



> 概率论

条件概率　全概率公式　贝叶斯公式

> KMP算法



暴力求解太慢了，因为一次坏字符模式串较主串只往前移动一位。KMP的主要提速点就在于遇到一个坏字符可以从模式串往前移动若干位。这得益于已匹配的字符。

next数组：i 已匹配前缀的下一个位置，也就是待填充的数组下标

 						j 最长可匹配前缀子串的下一个位置”，也就是待填充的数组元素值。

生成next数组：前两项为０，动态规划。`if pattern[j] ==pattern[i-1]:` `next[i]=next[i-1]+1 `

`else: j = next[j] if patten[j]==patten[i-1]:next[i]=next[j] ...`



>BERT 



>余弦相似度

首先，我们要记住一点，两个特征的余弦相似度计算出来的范围是**[-1,1]**
 其实，对于两个特征，它们的余弦相似度就是两个特征在经过L2归一化之后的矩阵内积。

```
import torch
import torch.nn.functional as F
#假设feature1为N*C*W*H， feature2也为N*C*W*H（基本网络中的tensor都是这样）
feature1 = feature1.view(feature1.shape[0], -1)#将特征转换为N*(C*W*H)，即两维
feature2 = feature2.view(feature2.shape[0], -1)
feature1 = F.normalize(feature1)  #F.normalize只能处理两维的数据，L2归一化
feature2 = F.normalize(feature2)
distance = feature1.mm(feature2.t())#计算余弦相似度
```



> numpy np 

- 添加行，列　使用 `np.c_[]` 和 `np.r_[]` 分别添加行和列　使用 `np.insert`　使用'column_stack'

- 浅拷贝(改变形状)　.view() 深拷贝　.copy()





> transformer 

首个完全抛弃RNN的recurrence，CNN的convolution，仅用attention来做特征抽取的模型。

Attention最早是Bengio在2014年运用在NMT(神经机器翻译)

左半边是Encoder部分，右半边是Decoder部分。Transformer有6层这样的结构

<img src="https://pic2.zhimg.com/v2-91c234f7b659e2774cd2b06b9a016360_r.jpg" alt="preview" style="zoom:45%;" />

> 广播机制，拓展，torch.cat( (A,B),0 )

两个 Tensors 只有在下列情况下才能进行 broadcasting 操作：

- 每个 tensor 至少有一维
- 遍历所有的维度，从尾部维度开始，每个对应的维度大小**要么相同，要么其中一个是 1，要么其中一个不存在**。

###### expand()函数

cat dim 在dim上变长

```python
a=['a','b']
print(''.join(a))
```

>XLNet, AE 自编码器

![img](https://img-blog.csdn.net/20171029173204782?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbjEwMDc1MzAxOTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

*y*=*s*(*W**x*+*b*)　*z*=*s*(*W*′*y*+*b*′)　*W*′=*WT*  *L**H*(*x*,*z*)=−∑*k*=1*n**l**n*[*x**k**l**o**g**z**k*+(1−*x**k*)*l**o**g*(1−*z**k*)] 

　学的是一个相等函数　y可以视为*x*的有损压缩形式

DAE:

在神经网络模型训练阶段开始前，通过Auto-encoder对模型进行预训练可确定编码器W的初始参数值。然而，受模型复杂度、训练集数据量以及数据噪音等问题的影响，通过Auto-encoder得到的初始模型往往存在过拟合的风险。

![img](https://img-blog.csdn.net/20171029173315272?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbjEwMDc1MzAxOTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**自回归语言模型（Autoregressive LM）** 根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，这种类型的LM被称为自回归语言模型.

缺点是只能利用上文或者下文的信息，不能同时利用上文和下文的信息

优点，其实跟下游NLP任务有关，比如生成类NLP任务，比如文本摘要，机器翻译等，在实际生成内容的时候，就是从左向右的，自回归语言模型天然匹配这个过程

**自编码语言模型（Autoencoder LM）**

DAE LM的优缺点正好和自回归LM反过来，它能比较自然地融入双向语言模型

缺点主要在输入侧引入[Mask]标记，导致预训练阶段和Fine-tuning阶段不一致的问题，因为Fine-tuning阶段是看不到[Mask]标记的。

XLNet:把当前点其他的输入点排列组合，这样就可以在上文看到下文的内容了。



>递归leetcode

例题：对称二叉树，有效的二叉搜索树，二叉树的最大深度...

归--> 问题归纳为一个基本（一般）单元的`解决方案`

递-->　这个解决方案可以一直递传下去（`通过函数调用自身`，改变函数参数），直到`终止条件`　　　三要求｀｀

有两个模板：



```python
T1: 
def fun(root):
    return fun(root.left)
T2:
def fun(root):
    def helper(root,x=x,y=y):
        return helper(root.left,X,Y)
    return helper(root)
```

T1: 当问题不需要额外参数（状态）时。

T2: 需要额外参数（状态 x,y）

思考过程：

​	1) 对于某个选定单元

​	2)  带的状态由参数传递

　3) 总结出一般规律

详细模板

```python
def fun(root):
    def helper(root,x,y)
  ①终止条件
		if not root:
        	return False
  ②规律解法
（逻辑　return True 要满足xx和yy）
		if not xx :
        	return False
        if not yy:
            return False
        return True(通关)
```



>leetcode 题目思路

###### 合并两个有序数组　　双指针 / 从前往后　双指针 / 从后往前

###### 爬楼梯　递归，动态规划，　滚动数组，　矩阵快速幂，　通项公式。　

###### 买卖股票的最佳时机 　动态规划

###### 计数素数　排除法

> 动态规划

１．状态定义

２．转移方程

３．初始值

４．输出值

效率优化：　时间上，空间上（滚动数组，或不需要存整个dp数组）

> leetcode 动态规划

- 最大子序和 	动态规划。 注意状态的定义，并不是所有的题目都是直接定义dp数组，dp[i]为i规模下的结果。比如这题。状态定义为dp[i]是以第i个元素结尾的连续子数组的最大和。因为题目要求的结果，就是dp[i]的最大值。结果的子数组一定是以某个数结尾的。原理是一样的，也是把原问题转化成同结构的子问题。（降维的方向不同）之前都是直接按n->i缩小问题规模。 找一个转移方程好得出的方向。
- 最长回文子串 
  - 动态规划　这题的状态是二维的。P[i,j]表示i->j 之间是否构成回文，输出的是，true里最长的。
  - 中心扩散　所有的答案肯定是从一个回文中心（奇，偶。[i,i], [i,i+1] ）扩散得到，记住最长的就行。
  - Manacher 算法  利用先前的信息减少计算。





> 数组与列表

- 双指针	`三数之和`

- 复杂度大于O($N^2$)?->先排序  `三数之和` 

- 置特殊值法  和标记法相似 --> 原地算法,对后续没有用的空间可以利用起来。 `矩阵置0`

- 字符如a-z，集合，赋值，就可以排序了，又回到第二点了。`字母异位词分组` 比较集合

- 哈希表，对每个字母进行映射，计数。（实质上是编码？**编码**出这个对象“tea”的一个**表示**，比较表示相等判断二者相等）`字母异位词分组` 

- 竟然还可以用质数相乘，因为是要有一个唯一的表示，质数是一个很好的工具。`字母异位词分组` 

- `代码` 输出多个列表，默认字典：

  ```python
  class Solution(object):
      def groupAnagrams(self, strs):
          ans = collections.defaultdict(list)
          for s in strs:
              ans[tuple(sorted(s))].append(s)
          return ans.values()
  ```

- 滑动窗口。（就是队列) (双指针)   是否重复-> 哈希表！`无重复字符的最长子串`


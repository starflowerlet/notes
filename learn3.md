>Tips python

- python 多进程共享的ndarray数组是只读的。　解决方案是复制一份到当前进程。

- jupyter 里配置虚拟环境。

  - 先激活虚拟环境，在此环境下，安装　**conda install ipykernel**　
  - 然后　**ipython kernel install --user --name=python34 （python34为你虚拟环境名称）**　
  - 最后在base 环境下启动jupyter即可。

- 解析JSON数据

  ```python
  import json
  with open('data.json', 'w') as f:
      json.dump(data, f)
   
  # 读取数据
  with open('data.json', 'r') as f:
      data = json.load(f)
  ```

  



> NLP资讯

2020.3语言与智能技术竞赛：***机器阅读理解、面向推荐的对话、关系抽取、语义解析和事件抽取\***

**自然语言处理（NLP）方向比较著名的几个会议有：ACL、EMNLP、NACAL、CoNLL、IJCNLP、CoNLL、IJCNLP、COLING、ICLR、AAAI、NLPCC**

> pytorch

- premute(dims) 参数是一系列的整数，代表原来张量的维度。比如三维就有0，1，2这些dimension

  比如图片img的size比如是（28，28，3）就可以利用img.permute(2,0,1)得到一个size为（3，28，28）的tensor。

- transformer包实用指南

  ```python
  import torch
  from transformers import *
  MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
            (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
            (GPT2Model,       GPT2Tokenizer,       'gpt2'),
            (CTRLModel,       CTRLTokenizer,       'ctrl'),
            (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
            (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
            (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
            (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
            (RobertaModel,    RobertaTokenizer,    'roberta-base'),
            (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
           ]
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)
  input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)]) #tensor([[ 101, 6821, 7027, 3221,  671,  763,  704, 3152,  102]])
  默认add_special_tokens＝True，False时，没有[CLS],[SEP]
  with torch.no_grad():
      last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
  ```

  tokenizer.convert_tokens_to_ids()  0-->'[PAD]' 100-->'[UNK]' 101-->'[CLS]' 102-->'[SEP]' 103-->'[MASK]'

  ==注意== tokenizer分数字不是一个数字一个数字分的，几个数字算一个token。一般中文分词可以先把标点数字过滤出来。

- 自己做的数据集也可以torch.save torch.load

- tensor的筛选，A[A>0]

- 显存的占用：１．网络模型自身参数占用的显存。２．模型计算时（包括forward/backward/optimizer）所产生的中间变量或参数也有占用显存。３．编程框架自身一些额外的开销。

- 报错：

  ValueError: Wrong shape for input_ids (shape torch.Size([1])) or attention_mask (shape torch.Size([1]))

  语句：embed_input = self.embedding(cur_input)[0]   

  ​	//self.embedding = BertModel.from_pretrained(bert_config)

  原因：cur_input不能是一维的，

  ​	//dec_input = torch.tensor([ 101 ] * batch_size).view(-1,1).to(device)

> #### vscode

######  vscode 跳转到指定的行数的快捷键: Ctrl+G

###### VScode快速移动光标到行尾和行首: **使用Home键和End键** （在方向键上面）

###### 神操作：　shift+ctrl+方向键



> 流畅的python

#### operator库

- reduce用来作简单的递归，比如求和（sum）,求积。

```python
from functools import reduce
from operator import mul
def fact(n):
	return reduce(mul, range(1, n+1))
```

- `itemgetter` 处理可迭代序列

  - 用处：根据元组的某个字段给元组列表排序

    ```python
    from operator import itemgetter
    for city in sorted(metro_data, key=itemgetter(1)):
    	print(city)
        #metro_data是一个元组的列表，根据元组的第二字段排序
    ```

  - 如果把多个参数传给 itemgetter,它构建的函数会返回提取的值构成的元组

    ```python
    cc_name = itemgetter(1, 0)
    for city in metro_data:
    	print(cc_name(city))
    ```

  - 还有`attrgetter` 和`methodcaller` ,前者是取属性，后者绑定一个方法，支持额外参数(部分应用)。```hiphenate = methodcaller('replace', ' ', '-')``

- `functools.partial`

  - 部分应用某一函数：部分应用是指,基于一个函数创建一个新的可调用对象,把原函数的某些参数固定。使用这个函数可以把接受一个或多个参数的函数改编成需要回调的API,这样参数更少。

    ```python
    from operator import mul
    from functools import partial
    triple = partial(mul, 3)#partial 的第一个参数是一个可调用对象,后面跟着任意个要绑定的定
    						#位参数和关键字参数。
    triple(7)
    list(map(triple, range(1, 10)))
    [6, 9, 12, 15, 18, 21, 24, 27]
    #因为map是接受一个参数，所以使用partial冻结某个参数，使其可以回调
    ```

    


>#### 10.12

### logging

- 将日志写入文件：初始化logger，设置等级，创建一个文件句柄，设置等级，指定并设置格式，添加句柄。

```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")
```

- 同时输出到控制台：再添加一个流句柄。

  ```python
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logger.addHandler(console)
  ```

### tqdm

```python
from tqdm import tqdm
for i in tqdm(range(1000)):
```


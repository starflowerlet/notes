>  编译cpp 文件

g++ <name>.cpp

./a.out

### python tips

#### {}.format 字符串格式化

1.直接一一对应传入
	data = ['HH',24]
	'my name is{},age is {}'.format(data[0],data[1])	'my name is HH,age is 24'
2.可以传入索引**（多次）**
	'my name is{1},age is{0}{0}'.format(data[0],data[1])	'my name is 24,age is HHHH'
3.字典传入
	data2 = {'name':'hh','age':24}
	'my name is{name},age is {age}'.format(*data2)		 'my name is hh,age is 24'
4.参数名/关键字参数
	'my name is{name},age is {age}'.format(name='hh',age=24) 'my name is hh,age is 24'
5.数组
	data = ['hh',24]
	'my name is {0}, {0} age is {1}'.format(data)		  'my name is hh,hh age is 24'

#### future

在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。

### python知识

- Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。 // dict.get(key, default=None)
  可以把不同的算法的函数指针放到一个字典里，这样就可以用字符串来选择算法。
  def fun():return 1	def fun1():return 2 fun_dict={"fun":fun,"fun1":fun1} 
  a = fun_dict.get("fun",None)  a //<function __main__.fun()> 	a() // 1

- 使用`__getattr__`做函数多态性

  

#### python装饰器

```python
def sum1():
  sum = 1+2
  print(sum)
```

在这个函数的基础上，多加一些操作，比如查看函数运行的时间。

```python
import time
def timeit(fun):
  start = time.clock()
  fun()
  end = time.clock()
  print("timed use",end-start)

timeit(sum1)
```

但我们希望还是用原函数的表示，即sum1().

```python
def timeit(fun):
  def wrapper():
    start = time.clock()
    fun()
    end = time.clock()
    print("timed use",end-start)
  return wrapper

#sum1=timeit(sum1)

@timeit
def sum1():
  sum =1 + 2
  print(sum)

sum1()
```

这样就行了。　这里语法糖@timeit  相当于在定义和调用之间(函数体要改)，加上一句　sum1=timeit(sum1)

#### python \__call__()

让类实例当做函数调用.

```python
class Bar():
  def __call__(self, *args, **kwargs):
    print("i am instance method")
b=Bar() #创建实例
b() # 类实例当做函数调用.
```



>  os.system()

- os.system(cmd) 执行linux命令，成功，返回０

> ####  conda pip 切换国内源

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

$设置搜索时显示通道地址
conda config --set show_channel_urls yes
#linux
将以上配置文件写在~/.condarc中
vim ~/.condarc

channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true

> pip换源

在使用pip的时候加参数-i，如清华源：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple　后接其他
豆瓣源　https://pypi.douban.com/simple/　
中科大源　https://pypi.mirrors.ustc.edu.cn/simple/

注意加http(s)

#安装pytorch 1.1.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.0.1 -f https://download.pytorch.org/whl/cpu/stable # CPU-only build

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.0.1 -f https://download.pytorch.org/whl/cu80/stable # CUDA 8.0 build

#jupyter notebook
直接在终端输入　jupyter notebook

### 学习pytorch-book

#### chapter 2
- 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
  y.add(x) # 普通加法，不改变y的内容
  y.add_(x) # inplace 加法，y变了
- grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
  x = t.ones(2, 2, requires_grad=True)
  y = x.sum()
  y.backward() # 反向传播,计算梯度
  x.grad  tensor([[1., 1.],
          [1., 1.]])
  y.backward()
  x.grad	tensor([[2., 2.],
          [2., 2.]])
  x.grad.data.zero_()	
  	tensor([[ 0.,  0.],
          [ 0.,  0.]])

#### 模型

- 模型参数的访问，初始化和共享 共享模型参数：
  - Module类的forward函数里多次调用同一个层。　
  - 创建net时，使用的是同一个对象，比如是工厂函数obj = nn.Linear().net= nn.Sequential(obj,obj).其实，层的实体只有一个，也就是说 `net[0]==net[1] //True` 所以共享参数。同时，共享梯度，也就是`net[0].weight.grad`,在反传的时候会累加，即使是不同的网络对象，如`net_1=nn.Sequential(obj,obj_another). grad += tilda //grad `是同一个内存。
- 自定义层
  - 直接继承Module类来进行模型构造。最主要重载__init__(),forward().在构造函数中，定义层，如self.hidden = nn.Linear(786,256)。在forward(),定义前向计算。如 a =self.act(self.hidden(x))
  - 使用官方继承Module类的Sequential、ModuleList和ModuleDict类:
    - Sequential- 前向计算是简单的串联各个层的计算。其实它接收的是一个有序字典（你可以传元组）。自动生成forward()
    - ModuleList:接收一个列表作为输入。可以进行append和extend操作。这些模块间没有顺序没有联系，可以不用保证输入输出维度匹配。要自己定义forward(),还有，它也不是简单的list。self.linears = nn.ModuleList([nn.Linear(10, 10)])　self.linears = [nn.Linear(10, 10)]　加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中
    - ModuleDict类：接受一个子模块的字典，可以进行字典操作。
  - 含模型参数的自定义层
    Parameter类，是Tensor子类。如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。如何让一个Tensor是Parameter？　①直接定义成Parameter类。②使用ParameterList,ParameterDict定义参数的列表和字典。	print(net(x, 'linear1'))

#### CIFAR-10分类

1. 使用torchvision进行数据的获取和预处理
2. 定义网络各层级
3. 定义损失函数和优化器
4. 输入数据进行训练
5. 在测试集上预测

#### 直接使用下载好的CIFAR10进行训练
```python
trainset = tv.datasets.CIFAR10(
                    root='./data', #cifar-10-python.tar.gz 所在的位置
                    train=True, 
                    download=True,
                    transform=transform)
```

#### Tensor

a = t.Tensor(2,3)  #指定tensor的形状	
	tensor(1.00000e-37 *
       [[-8.9677,  0.0003, -8.9884],
        [ 0.0003,  0.0000,  0.0000]])

b = t.Tensor([1,2,3],[4,5,6])  #用list
	tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]])

b_size = b.size()　#tensor.size()返回torch.Size对象， 
		  **#tensor.shape等价于tensor.size()**
	torch.Size([2, 3])

c = t.Tensor(b_size) #创建一个和b形状一样的tensor

d = t.Tensor((2, 3)) #创建一个元素为2和3的tensor,注意和a 区别

scalar = t.tensor(3.1415) # torch.tensor使用的方法，参数几乎和np.array完全一致

Tensor: torch的一些构造函数来创建：　x = torch.empty(5,3) .ones(*size) .tensor(data) .zeros(*size) .arange(s,e,step),从s到e，步长为step
.linspace(s,e,steps) 从s到e，均匀切分成steps份 .rand/randn(*sizes) 均匀/标准分布　等等
**索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改**　`y = x[0, :]　y[1]=1e2`
用 **view()** 来改变Tensor的形状：view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。
返回一个真正新的副本　推荐先用**clone**创造一个副本然后再使用view
另外一个常用的函数就是**item()**, 它可以将一个标量Tensor转换成一个Python number：
广播机制：当对两个形状不同的Tensor按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个Tensor形状相同后再按元素运算。

​	:question:  Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）每个Tensor都有一个.grad_fn属性，该Tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None。 (grad_fn=<AddBackward>)

#### Autograd

在创建tensor的时候指定requires_grad

`a = t.randn(3,4, requires_grad=True)`或者`a = t.randn(3,4).requires_grad_()`或者

`a = t.randn(3,4)  a.requires_grad=True`

x = t.ones(1)
b = t.rand(1, requires_grad = True)
w = t.rand(1, requires_grad = True)
y = w * x # 等价于y=w.mul(x)
z = y + b # 等价于z=y.add(b)

grad_fn可以查看这个variable的反向传播函数，

z是add函数的输出，所以它的反向传播函数是AddBackward

z.grad_fn.next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function

第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward

z.grad_fn.next_functions

z.grad_fn.next_functions[0] [0]== y.grad_fn
True

with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad = True)
    y = x * w

y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False

x.requires_grad, w.requires_grad, y.requires_grad
(False, True, False)

**如果我们想要修改tensor的数值，但是又不希望被autograd记录，那么我么可以对tensor.data进行操作**
a = t.ones(3,4,requires_grad=True)
b = t.ones(3,4,requires_grad=True)
c = a * b

a.data # 还是一个tensor
d = a.data.sigmoid_() # sigmoid_ 是个inplace操作，会修改a自身的值

**在反向传播过程中非叶子节点的导数计算完之后即被清空**。若想查看这些变量的梯度，有两种方法：

    使用autograd.grad函数
    使用hook
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w

y依赖于w，而w.requires_grad = True

z = y.sum()

非叶子节点grad计算完之后自动清空，y.grad是None

z.backward()
(x.grad, w.grad, y.grad)



#### pytorch 常用工具模块　-数据

1. 自定义一个数据集的类，class DogCat(data.Dataset) 
    //继承自　data.Dataset 
    
2. 需要定义的方法: __getitem__(), __len__()

3. 对数据处理(比如归一化)
　　对PIL图像对象可以: Scale;　CenterCrop,　RandomResizeCrop;　Pad;　ToTensor
    对Tensor: Normalize  ToPILImage
    对图片同时进行多个操作　使用Compose 把多个操作连起来。

　　```python
　　from torchvision import transforms as T
　　transform = T.Compose([
　　		T.Resize(224),	#缩放，最短边为224,长宽比不变
　　		T.CenterCrop(224),#从中间切出224*224
　　		T.ToTensor(),
　　		T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
　　])
　　```
　　
　　

4. **Dataset 只是数据集的一个抽象，需要执行一次getitem()，才能读取一次数据。数据读取中的细节，比如数据打乱，并行加速等，使用DataLoader来做。**

```python
from torch.utils.data import DataLoader
```

   DataLoader函数定义：

   ```python
   DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)
   ```

 dataset：加载的数据集(Dataset对象)
    batch_size：batch size
    shuffle:：是否将数据打乱
    sampler： 样本抽样，后续会详细介绍
    num_workers：使用多进程加载的进程数，*0代表不使用多进程*
    collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
    pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
   DataLoader 是一个可迭代对象
   `for batch_datas,batch_labels in dataloader:`

DataLoader 封装了多进程库，所以不要在里面使用可变对象(比如计数器)　另外，高负载的操作应该放在__getitem__(),因为pytorch会并行的执行__getitem__().

5. 还提供了sampler模块。当shuffle=True时，使用RandomSampler,默认使用SequentialSampler.还有一个WeightedRandomSampler.参数:weight,num_samples,replacement.

6. torchvision
    包含三部分：models,datasets,transforms.

### 计算机小知识

- ssh-keygen 用来生成ssh公钥认证所需的文件
  		ssh-copy-id -p 15679 mig-01@122.51.251.31

- source ~/.bashrc  	source 是重新载入后面的文件，再全部执行，　~/ 当前用户的home文件夹，.bashrc文件保存的是环境变量。所以这句话是修改环境变量。

- vscode插件　remote-ssh　实现在VSCode上的终端直接访问，直接在VSCode上新建保存文件

- conda create -n pytorch python==3.7　自定义一个python环境
  　　　　source activate enname 激活该环境

​				conda info -e 查看有哪些环境

- SSH 为 Secure Shell 的缩写。SSH 为建立在应用层基础上的安全协议。SSH提供两种级别的安全验证。
  	第一种级别（基于口令的安全验证）
    	第二种级别（基于密匙的安全验证）	需要依靠密匙，也就是你必须为自己创建一对密匙，并把公用密匙放在需要访问的服务器上。客户端软件就会向服务器发出请求，请求用你的密匙进行安全验证。服务器收到请求之后，先在该服务器上你的主目录下寻找你的公用密匙，然后把它和你发送过来的公用密匙进行比较。如果两个密匙一致，服务器就用公用密匙加密“质询”（challenge）并把它发送给客户端软件。客户端软件收到“质询”之后就可以用你的私人密匙解密再把它发送给服务器。
    	它的命令格式是👉 ssh [-p port] user@remote　
    	有关SSH配置信息都保存在用户家目录下的.ssh目录下

- Linux scp命令用于Linux之间复制文件和目录。
  	scp是 secure copy的缩写, scp是linux系统下基于ssh登陆进行安全的远程文件拷贝命令。
    	它的地址格式与ssh基本相同，需要注意的是在指定端口时用的是大写的-P而不是小写

  ```shell
  scp -P 22 tx@10.170.55.151:/home/tx/hhua/Bert-BiLSTM-CRF-pytorch-master/result/2020-07-29-18\:11\:36--epoch\:34 /home/hh/Documents/temp_code/	 从服务器上复制文件到本地，在本地shell下输入。
  scp -P 22 02.txt deepin2@192.168.56.132:Desktop  客户端发送文件到服务器，在本地要发的文件的目录下输入。
  ```

１．model = getattr(models, opt.model)()   　　
    Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y. When a default argument is given, it is returned when the  attribute doesn't exist; without it, an exception is raised in that case.
    <2-2> import torchnet as tnt tnt.Meter 有三个基本方法　.add()　.value() .reset() tnt.AverageValueMeter(self)
      <https://blog.csdn.net/a362682954/article/details/82926499>

    <2-3> confusion matrix 　用来呈现算法性能的可视化效果。其每一列代表预测值(的分布)，每一行代表的是实际的类别(的分布)。这个名字来源于它可以非常容易的表明多个类别是否有混淆。
      <https://blog.csdn.net/vesper305/article/details/44927047>

#### ray

-  安装ray
    	先确保更新了numpy 和scipy, 以及redis,再进行pip install -U ray 之后有一个警告，可以 pip install psutil解决。
- Ray /import ray　　/ https://ray.readthedocs.io/en/latest/walkthrough.html
      定义一个函数时，加上装饰器@ray.remote 就可以将一个python函数变成remote functions

```python
#　A regular Python function.
def regular_function():
    return 1

# A Ray remote function.
@ray.remote
def remote_function():
    return 1
```

　　　　调用方法　一般的函数就是　regular_function() / 远程函数: remote_function.remote()
	返回值：　一般的函数会立即被执行并返回１．whereas remote_function immediately returns an object ID (a future) and then creates a task that will be executed on a worker process. The result can be retrieved with ray.get.（远程函数会返回一个对象ＩＤ标识（可以是多个），创建了一个任务，将在未来被执行，结果可以通过ray.get() 得到）(那到底什么时候执行呢？应该是调用.get的时候执行，如果执行了就算了，没执行放入队列，马上执行。)

```python
assert regular_function() == 1

object_id = remote_function.remote()

# The value of the original `regular_function`
assert ray.get(object_id) == 1

对象ＩＤ可以作为参数传递给另一个远程函数，如果你这么做，调用方只能在第一个任务被完成时开始执行。

@ray.remote
def remote_chain_function(value):
    return value + 1
y1_id = remote_function.remote()
assert ray.get(y1_id) == 1

chained_id = remote_chain_function.remote(y1_id)
assert ray.get(chained_id) == 2
```

对象ＩＤ可以是多个：


```python
@ray.remote(num_return_vals=3)
def return_multiple():
    return 1, 2, 3

a_id, b_id, c_id = return_multiple.remote()

对象ＩＤ可以用ray.put 获得

y = 1
object_id = ray.put(y)

取得结果：
ray.get(x_id, timeout=None)　用ｉｄ对应的远程对象创建一个python对象。
y = 1
obj_id = ray.put(y)
assert ray.get(obj_id) == 1
```
- After launching a number of tasks, you may want to know which ones have finished executing. This can be done with ray.wait. The function works as follows.

`ready_ids, remaining_ids = ray.wait(object_ids, num_returns=1, timeout=None)`

远程类Actors（被装饰器@ray.remote装饰的类）An actor is essentially a stateful worker. Each actor runs in its own Python process.

```python
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
```

```python
def increment(self):
    self.value += 1
    return self.value

要得到两个工人，就实例化两次。
a1 = Counter.remote()
a2 = Counter.remote()
```

​	当一个actor被实例化时，首先一个进程被创建，然后一个Counter类被实例化在其上。

Create ten Counter actors.

`counters = [Counter.remote() for _ in range(10)]`

Increment each Counter once and get the results. These tasks all happen in parallel.

```python
results = ray.get([c.increment.remote() for c in counters])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Increment the first Counter five times. These tasks are executed serially and share state.

```python
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)  # prints [2, 3, 4, 5, 6]
```

```python
Actor Methods
ray.init(memeory=<bytes>,object_store_memory=<bytes>)来设置给ray分配多少资
```

3. 读取和存储
	net.state_dict() 返回有序字典，包含的是可学习的参数，weigh,bias. (网络net和layer其实是等价的)net[1].state_dict()
	而且，优化器也有一个state_dict,包含的是优化器的状态和超参数

	保存和加载模型:
	1. 仅保存和加载模型参数(state_dict);(推荐)
		保存: torch.save(model.state_dict,PATH)
		加载: model = TheModelClass(*args, **kwargs)
		model.load_state_dict(torch.load(PATH))
	2. 保存和加载整个模型。

4. GPU计算
		nvidia-smi　命令来查看显卡信息了。
		torch.cuda.is_available()　查看GPU是否可用:
		torch.cuda.device_count()　查看GPU数量
		torch.cuda.current_device()　查看当前GPU索引号，索引号从0开始

	使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，我们用.cuda(i)来表示第 ii 块GPU及相应的显存。
	
	我们可以通过Tensor的device属性来查看该Tensor所在的设备。
	我们可以直接在创建的时候就指定设备。	

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device) or x = torch.tensor([1, 2, 3]).to(device)同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。我们可以通过检查模型的参数的device属性来查看存放模型的设备。
5. 卷积神经网络
		多通道输入和多通道输出
	一个输入是多通道的，比如c*h*w 那么卷积核的形状也要是c*h*w,每个通道对应位置相乘，最后把各个通道相加，得到一个输出，注意，输出仍然是一通道的。如果你想得到一个多通道的输出，你可以将卷积核的形状设置为c_o*c*h*w.
		1x1卷积层
	输入和输出具有相同的高和宽。输出中的每个元素来自输入中在高和宽上相同位置的元素在不同通道之间的按权重累加。假设我们将通道维当作特征维，将高和宽维度上的元素当成数据样本，那么1×11×1卷积层的作用与全连接层等价。

	池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感
	池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加

#11.25
1. LeNet
是由卷积层模块和全连接层模块构成。卷积层模块又是由卷积层和最大池化层构成。卷积层的输出形状为（批量大小，通道，高，宽）。当卷积层块的输出传入全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输入形状将变成二维，其中第一维是小批量中的样本，第二维是每个样本变平后的向量表示，且向量长度为通道、高和宽的乘积。
		nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
		nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
        nn.MaxPool2d(2, 2), # kernel_size, stride

2. AlexNet
	第一层卷积核是11x11(因为处理的图像更大)，第二层使用5x5,之后全部是3x3.在每次改变卷积核大小之后接一个池化层，3x3，步长为2.一共是五层卷积层，两个全连接隐藏层，一个全连接输出层。 激活函数是RELU.使用丢弃法和图像增强。
	
	如果原图片的形状和ImageNet图像的大小不同的话，要先进行变换
trans = []
if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
trans.append(torchvision.transforms.ToTensor())

transform = torchvision.transforms.Compose(trans)

3. VGG:	提出了可以通过重复使用简单的基础块来构建深度模型的思路

对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核优于采用大的卷积核，因为可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。例如，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。
通过每一层，高和宽减半，同时，通道数翻倍，直到512.

4. NiN

AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽（增加通道数）和加深。网络中的网络（NiN）提出了另外一个思路，即串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络。
卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。因此，NiN使用1×1卷积层来替代全连接层
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。

5. 可以尝试用hexo写博客。

6. GoogLeNet

GoogLeNet吸收了NiN中网络串联网络的思想。GoogLeNet中的基础卷积块叫作Inception块。Inception块相当于一个有4条线路的子网络。它通过不同窗口形状的卷积层和最大池化层来并行抽取信息，并使用1x1卷积层减少通道数从而降低模型复杂度。
		前3条线路使用窗口大小分别是1x1、3x3和5x5的卷积层来抽取不同空间尺寸下的信息，其中中间2个线路会对输入先做1x1卷积来减少输入通道数，以降低模型复杂度。第四条线路则使用3x3最大池化层，后接1x1卷积层来改变通道数。最后我们将每条线路的输出在通道维上连结，并输入接下来的层中去。

	GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block）第一模块使用一个64通道的7ｘ7卷积层. 在第三，四，五个模块里使用多个Inception块串联构成。输出的通道由参数传入。网络还是串联的。

7. 批量归一化　BN
	标准化处理：处理后的任意一个特征在数据集中<所有>样本上的均值为0、标准差为1。标准化处理输入数据使各个特征的分布相近：这往往更容易训练出有效的模型。　数据标准化预处理对于浅层模型就足够有效了，但对深度网络不够。
	在模型训练时，批量归一化利用<小批量>上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

全连接层做批量归一化:批量归一化层置于全连接层中的仿射变换和激活函数之间
<看图> https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm

8. ResNet
	ResNet模型是由Residual块（残差块）和其他的层构成。一个普通卷积块对应的残差块的输出是将输出和输入相加得到的。设原模块想要得到的映射是f(x),残差块结构下的模块期望得到的映射就是f(x)-x (残差由此得名)，因为实际上残差函数更容易优化。这种结构就要求，输出的通道数与输入的通道数相同。如需改变输出通道数，可使用1x1卷积层来减少通道数。（1x1卷积层可以自定义输出通道数）ResNet沿用了VGG全3x3卷积层的设计。残差块里首先有2个有相同输出通道数的3x3卷积层。每个卷积层后接一个批量归一化层和ReLU激活函数。
	ResNet模型和GoogLeNet一样，在最前面使用一个64通道　7x7卷积核，之后使用3x3卷积，后接最大池化层，之后和GoogLeNet不同，讲四个Inception模块替换成Residual块。经过一个模块，通道数翻倍，高宽减半。每个模块４个卷积层，加上头尾的卷积层和全连接层，称为ResNet18.

9. DenseNet
	与ResNet不同，不是将输出和输入相加，而是在通道上连结（通道数相加）。DenseNet模型里有两个特有的模块，稠密层和过渡层。一个将通道数相加，一个控制通道数。
	稠密块由多个conv_block组成，conv_block包含了批量归一化、激活和卷积。每块使用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结，然后传入下一块。
	过渡层用来控制模型复杂度。它通过1x1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
	DenseNet模型，DenseNet首先使用同ResNet一样的单卷积层和最大池化层，DenseNet使用的4个稠密块。同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。这里我们设成4，稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。这里我们则使用过渡层来减半高和宽，并减半通道数。最后接上全局池化层和全连接层来输出。

		循环神经网络
10. 语言模型
	一段自然语言文本可以看作一段离散的时间序列。假设一段长度为TT的文本中的词依次为w1,w2,…,wT​，那么在离散的时间序列中，wt​（1≤t≤T）可看作在时间步（time step）t的输出或标签。
	N元语法是基于n−1阶马尔可夫链的概率语言模型，其中n权衡了计算复杂度和模型准确性。
11. RNN
	循环神经网络是为了解决N元语法当N很大的时候计算量很大的问题。它并非刚性地记忆所有固定长度的序列，而是通过隐藏状态来存储之前时间步的信息。
	当前时间步的输出，不仅和输入X有关，还和上一步的隐藏状态有关。H=XW+H1w+b.上式的矩阵相乘相加，等价于X H先连结再相乘。

#11.28

1. 从０实现ＲＮＮ
	scatter_
	每次采样的小批量的形状是(批量大小, 时间步数)，通过one-hot转换成数个（批量大小，词典大小）

#1.15

1. linux服务器没有.ssh文件夹　-->ssh localhost 输入上面命令,然后按照提示yes在输入密码就可以生成了, ssh是记录你密码信息的, 没有登录过root ,是没有.ssh 文件夹的 
2. 免密登录ssh  示例中的服务器ip地址为192.168.1.1，ssh端口为22。
	1. 客户端生成密钥　ssh-keygen -t rsa
	2. 服务器配置　scp -P 22 ~/.ssh/id_rsa.pub hh@192.168.1.1
			cat id_rsa.pub >> ~/ssh/authorized_keys
	3.客户端配置其他信息
			~/.ssh/config
	Host server
	Hostname 192.168.1.1
	Port 22
	User bingoli
	
	ssh server



#2.20
 假设有一个列表list 里面的元素都是array,但是大小各不相同。比如gradient_groups. 现在要对每一个数都加上一个噪声，或者是一种操作。　
　　有一个np.nditer(a) 迭代数组。
import numpy as np

a = np.arange(6).reshape(2,3)
print ('原始数组是：')
print (a)
print ('\n')
print ('迭代输出元素：')
for x in np.nditer(a):
    print (x, end=", " )
print ('\n')

或者直接对每一个array进行处理。
 for i in range(len(grad_list)):
            size = np.shape(grad_list[i])
            grad_list[i]+= self.theta * np.random.randn(*size)
讲元组作为参数，要加上*

#3.4 
pip install -e . 点很关键　前提是先clone 比如　git clone 
这叫从VCS　或者从local 安装。

sklearn --> scikit-learn

#4.21

>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]



1.百度ai　持久性安装

pip install -i https://pypi.douban.com/simple/  torchvision==0.2 -t /home/aistudio/external-libraries

> screen

本地关机，服务器仍然运行

screen -S train 第一次进去

　Ctrl +a +d 切出来

查看

​	screen -R train 进去

删除

​	先　screen -ls 

122128.test     (12/04/2017 08:35:43 PM)        (Attached)

​	删除它

​	screen -X -S 122128 quit	

​	再screen -ls就没了

上翻页

​	先ctrl+a 松开 加[ 就可以了。

​	退出 ctrl+c



jupyter notebook --config /root/.jupyter/jupyter_notebook_config.py

ssh -L8008:localhost:8888 tx@10.170.64.231



深度学习之参数初始化——Xavier初始化 https://blog.csdn.net/weixin_35479108/article/details/90694800



---

> matplotlib:

python plot　设置线属性

ax**.**plot(x,y, c='r', markersize = 8,  marker='>', markevery=10)　　// c 指定颜色，r,y,b marker 线上的标记　还有* o 等标记　markervery 隔几个点


<hr style=" border: 1px solid #blue;">
---

>markdown:

'>' 引用　+ 列表

---

> 流畅的python

+ 元组拆包

  b, a = a, b

  divmod(20,8) 	t=(20,8)	divmod(*t)

  用\*处理剩下的元素，不定长	a,b, \*rest =range(5) 


















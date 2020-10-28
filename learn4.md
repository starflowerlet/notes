# torch

- **torch.squeeze()** 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度。`squeeze(a)`就是将a中所有为1的维度删掉。`a.squeeze(N)` 就是去掉a中指定的维数为一的维度。也可以写成：`b=torch.squeeze(a，N) `

  **torch.unsqueeze()**这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度。

  a.unsqueeze(N) 就是在a中指定位置N加上一个维数为1的维度。也可以写成：b=torch.unsqueeze(a，N)

- torch.expand()函数返回张量在某一个维度扩展之后的张量，就是将张量广播到新形状

- torch.scatter_(dim,index,src) 放置元素，修改元素。依照index, scr来。

  https://www.cnblogs.com/dogecheng/p/11938009.html

  一般来说index, scr同形状，scr放着准备放置的元素值，index放着准备放置的（部分）索引值,看dim参数。参考下式：

  `self[index[i][j] [j]] = src[i][j]` 

  **src 除了可以是张量外，也可以是一个标量**，scatter() **一般可以用来对标签进行 one-hot 编码**，这就是一个典型的用标量来修改张量的一个例子。

  ```python
  class_num = 10
  batch_size = 4
  label = torch.LongTensor(batch_size, 1).random_() % class_num
  torch.zeros(batch_size, class_num).scatter_(1, label, 1)
  
  ```

# tips



互相的注意力：

<img src="https://img-blog.csdn.net/20170424214300557?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvam9zaHVheHgzMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center" alt="img" style="zoom:25%;" />

# 流畅的python



## 装饰器

- 装饰器是可调用对象，参数是另一个函数，用来处理，增强被装饰的函数。它可以返回这个函数，也可以返回另一个函数。同时把被装饰的函数变成装饰器函数的引用。`target = decorate(target)` 
- 函数装饰器在导入模块时立即执行。
- 变量作用域

```python
b = 6
def f2(a):
    print(a)
    print(b)
    b = 9
此时，会报错local variable 'b' reference before assignment
```

- 闭包：闭包是一种延伸了作用域的函数，其中包含了在函数定义体中引用，但不在函数定义体中定义的非全局变量。（例子：移动平均值）｜　自由变量　未在本地作用域中绑定的变量。　｜ nonlocal 关键字 用来标记自由变量。
- 基本例子：

```python
import time
def decorate(func):
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed =time.perf_counter()-t0
        name = func.__name__
        arg_str = ','.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked
使用：
@decorate
def snooze(seconds):
    time.sleep(seconds)
```

- 一个更好的方式是使用 `functools.wraps`装饰器（多加一层）
- 标准库中两个装饰器：`functools.lru.cache`和 `functools.singledispatch` 
- 带参数的装饰器：参数跟在@后面。`@clock('{name}:{elapsed}s')` 加一层工厂函数。

```python 
def clock(fmt='default fmt'):
    def decorate(func):
        def clocked(*_args):
            ---
            return --
        return clocked
    return decorate
```

## 对象引用

- 变量不是盒子，是标注。贴多个标注，就是别名。
- is 比较两个对象的标识（地址），==比较值。
- 元组的不变性是元组的各个元素的标识（保存的引用）不变，被引用的对象可以变。
- 浅复制的方法：`list()` 或者`li[:]` 得到副本，其实是复制了最外层容器的每个元素的引用，如果每个元素的引用都是不可变的，那没事，如果可变，考虑要深拷贝。copy模块中copy是浅复制，deepcopy是深复制。
- 函数的参数传递的唯一方式是共享传递即形参获得实参中各个引用的副本。
- 不要使用可变类型作为参数的默认值。因为通过默认值实例化出来的对象都绑定了那个可变对象，会对后面的产生影响。
- 不单单是默认值，可变类型作为实参都要注意，直接在函数里赋值，会使这一变化延伸到函数体外。一个防御方法是将实参的副本绑定到类属性。
- del语句删除的是引用，而不是对象本身。对象只有在引用计数为0，或者因循环引用而无法获得时，由垃圾回收程序删除。
- 弱引用不会影响对象的引用计数。weakref模块
- 在交互环境下，没有赋值的语句会绑定到`_` 变量。



## python风格的对象

获取对象的字符串表示形式的标准方式：repr():便于开发者理解 str() 便于用于理解 

- 定义`__iter__` 方法可以使实例变成可迭代的对象，这样才可以拆包。

  ```python
  def __iter__(sefl):
      return (i for i in (self.x,self.y))
  ```

- 两个装饰器 @classmethod @staticmethod 定义类方法，定义静态方法（普通），第一个参数始终是cls |  第一个参数始终是空

- 格式化显示：内置的format函数和str.format()方法 内部是相应的`__format__(format_spec)`

  ```python
  format(42,'b') b二进制,x十六进制，f小数，%百分数
  ```

  如果类没有定义--format--()方法，会返回str(my_object).这时不能使用`format(v1,'.3f')` 解决方案是增加该方法的定义，实现可以迭代每一项（因为已经实现`__iter__`）,为每一项调用format(.)
  
- 你要把这个对象放入集合中，或者是作为字典的键，必须是可散列的，可以理解成不可变的，因为对其作hash必须返回同一个值。使其可散列，必须定义`__hash__(),__eq__()` 可以把对象的属性设成私有的，用@property定义一个读函数

  ```python
  def __init__(self,x,y):
      self.__x = x
      self.__y = y
  @porperty
  def x(self):
      return self.__x
  ```

  用__x 的方式，会触发名称改写，它实际的名称会加上 _classname. 这种叫私有属性

  用_x 的方式，一个下划线，本身没有任何机制，但是这是一种约定俗成的，受保护的属性，不会有人在类外部访问它。

## 序列的修改，散列和切片

`协议` 是非正式的接口，在文档中定义，代码无定义。如python 的序列协议只要`__len__`,`__getitem__` 只要实现了这两个方法，我们就可以说它是序列，因为它表现得像序列。最简单的方法可以把这两个方法委托给对象的序列属性（如它有一个属性是列表），现在它甚至可以切片了。但是这样的==问题1==是切片返回的是数组类型。

`切片原理` 实际上，它会把你的表示法，如 1:4 转化成一个`slice(1,4,None)` slice类是内置类型，有start, stop, step 属性，以及indices方法，主要是用来整顿参数元组，使其能够应对如负数这种不平常的参数。现在==问题1==的解决方案是用序列属性切片，然后再调用构造函数。

`动态存取属性 __getattr__,__setattr__`属性查找失败，会调用`__getattr__`方法，如执行`v.x`时。同时，如果对写属性有要求，如只读，要实现`__setattr__` 




































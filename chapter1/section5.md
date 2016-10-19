## Numpy性能: uint8 vs. float，乘法运算 vs. 除法运算？
### [Question](http://stackoverflow.com/questions/39104562/numpy-performance-uint8-vs-float-and-multiplication-vs-division):

我发现我的一个脚本的运行时间减半仅仅是因为我把其中的乘法运算改成了除法运算。

为了搞清楚这个问题，我写了一个小例子：

```python
import numpy as np                                                                                                                                                                                
import timeit

# uint8 array
arr1 = np.random.randint(0, high=256, size=(100, 100), dtype=np.uint8)

# float32 array
arr2 = np.random.rand(100, 100).astype(np.float32)
arr2 *= 255.0


def arrmult(a):
    """ 
    mult, read-write iterator
    """
    b = a.copy()
    for item in np.nditer(b, op_flags=["readwrite"]):
        item[...] = (item + 5) * 0.5

def arrmult2(a):
    """ 
    mult, index iterator
    """
    b = a.copy()
    for i, j in np.ndindex(b.shape):
        b[i, j] = (b[i, j] + 5) * 0.5

def arrmult3(a):
    """
    mult, vectorized
    """
    b = a.copy()
    b = (b + 5) * 0.5

def arrdiv(a):
    """ 
    div, read-write iterator 
    """
    b = a.copy()
    for item in np.nditer(b, op_flags=["readwrite"]):
        item[...] = (item + 5) / 2

def arrdiv2(a):
    """ 
    div, index iterator
    """
    b = a.copy()
    for i, j in np.ndindex(b.shape):
           b[i, j] = (b[i, j] + 5)  / 2                                                                                 

def arrdiv3(a):                                                                                                     
    """                                                                                                             
    div, vectorized                                                                                                 
    """                                                                                                             
    b = a.copy()                                                                                                    
    b = (b + 5) / 2                                                                                               




def print_time(name, t):                                                                                            
    print("{: <10}: {: >6.4f}s".format(name, t))                                                                    

timeit_iterations = 100                                                                                             

print("uint8 arrays")                                                                                               
print_time("arrmult", timeit.timeit("arrmult(arr1)", "from __main__ import arrmult, arr1", number=timeit_iterations))
print_time("arrmult2", timeit.timeit("arrmult2(arr1)", "from __main__ import arrmult2, arr1", number=timeit_iterations))
print_time("arrmult3", timeit.timeit("arrmult3(arr1)", "from __main__ import arrmult3, arr1", number=timeit_iterations))
print_time("arrdiv", timeit.timeit("arrdiv(arr1)", "from __main__ import arrdiv, arr1", number=timeit_iterations))  
print_time("arrdiv2", timeit.timeit("arrdiv2(arr1)", "from __main__ import arrdiv2, arr1", number=timeit_iterations))
print_time("arrdiv3", timeit.timeit("arrdiv3(arr1)", "from __main__ import arrdiv3, arr1", number=timeit_iterations))

print("\nfloat32 arrays")                                                                                           
print_time("arrmult", timeit.timeit("arrmult(arr2)", "from __main__ import arrmult, arr2", number=timeit_iterations))
print_time("arrmult2", timeit.timeit("arrmult2(arr2)", "from __main__ import arrmult2, arr2", number=timeit_iterations))
print_time("arrmult3", timeit.timeit("arrmult3(arr2)", "from __main__ import arrmult3, arr2", number=timeit_iterations))
print_time("arrdiv", timeit.timeit("arrdiv(arr2)", "from __main__ import arrdiv, arr2", number=timeit_iterations))  
print_time("arrdiv2", timeit.timeit("arrdiv2(arr2)", "from __main__ import arrdiv2, arr2", number=timeit_iterations))
print_time("arrdiv3", timeit.timeit("arrdiv3(arr2)", "from __main__ import arrdiv3, arr2", number=timeit_iterations))
```

运行这个脚本，得到如下运行时间测量结果：
```txt
uint8 arrays
arrmult   : 2.2004s
arrmult2  : 3.0589s
arrmult3  : 0.0014s
arrdiv    : 1.1540s
arrdiv2   : 2.0780s
arrdiv3   : 0.0027s

float32 arrays
arrmult   : 1.2708s
arrmult2  : 2.4120s
arrmult3  : 0.0009s
arrdiv    : 1.5771s
arrdiv2   : 2.3843s
arrdiv3   : 0.0009s
```

我一直认为乘法运算的复杂度要低于除法运算，但是对于`uint8`，除法运算的性能是乘法运算的两倍。这会不会是因为`* 0.5`是需要计算浮点数的乘法，之后再将结果转换为一个整型？

至少再上面的浮点运算测量结果中，乘法运算看起来要比除法运算更快，这是正确的吧？

为什么`uint8`的乘法运算要比`float32`的更耗？我觉得8比特的无符号整型在运算时应该远远快于32比特的浮点数呀。

哪位能阐明其中的道理？

*EDIT*: 为了获得更多的数据，我依建议添加了向量化(vectorized)的测试函数(译者著：arrmult3, numpy重载了矩阵的标量四则运算，具体见代码)，还添加了index iterator的版本(译者著：看起来是numpy提供的矩阵版enumerate)。向量化的函数快多了，因此，没有真正可比性。但是，如果将`timeit_iterations`设定为一个比较大的值，向量化的测试函数显示，无论对于`uint8`还是`float32`，乘法运算都要比除法运算快，这让我更疑惑了。

也许乘法运算总是比除法运算要快，这里性能出现差异的原因会不会是来自for-loop本身，而不是来自算数运算？尽管这不能解释为什么for-loop会不同类型的算数运算影响。

*EDIT2* 正如@jotasi所说的，我们正在探讨的是除法运算 vs. 乘法运算，int(或者uint8) vs. float (或者float32)。另外，对向量化的方案或者迭代器的介绍也会很有意思，因为在向量化的方案中，（相比较乘法运算）除法运算更慢，而在迭代器的方案中，除法运算更快。

### Answer:
| rank | ▲    | url  |
| :--- | :--: | :--: |
| 1	   | 5    | [url](http://stackoverflow.com/a/39174057/763878) |

你的假设有问题，你想要测乘法运算和除法运算的耗时，但是实际上你的测试代码测量的时间可不仅仅是乘法运算和除法运算。

你需要仔细研究代码才能搞清楚究竟发生了什么，并且这一切因版本而异，这个回答只是试图给你提供一些必须要要考虑到的问题。

问题在于，`int`在python中的并不简单：它是被垃圾回收机制管理着的对象，这使得它需要消耗更多的内存：比如对于8比特的int数据，实际在内存中可能会占据24比特的空间！对于python的`float`类型也同样如此。

另一方面，`numpy`数组有`c-style`的整型和浮点型数据构成，并没有这些额外的开销，节省了一些内存，但是当你需要在python中访问`numpy`数组中的元素的时候就需要付出额外的代价。`a[i]`意味着：一个python的`int`对象需要被构造，需要像垃圾回收机制去注册——只有这样，才能在python中使用——这就存在着额外的开销。

考虑下面的代码：
```python
li1=[x%256 for x in xrange(10**4)]
arr1=np.array(li1, np.uint8)

def arrmult(a):    
    for i in xrange(len(a)):
        a[i]*=5;
```

`arrmult(li1)`比`arrmult(arr1)`快25倍，因为列表中的元素已经是python的`int`对象，不会造成大量的int对象需要被创建的情况(除此之外的开销基本上可以被忽略)。
